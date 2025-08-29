import numpy as np
from rltaskoffloading.environment.offloading_task_graph import OffloadingTaskGraph
from collections import defaultdict


import numpy as np

class ServerSpecs:
    """A data class to hold all specifications for a server type, including a dynamic power model."""
    def __init__(self, mips, cost_per_second, cost_per_byte, 
                 min_freq_ghz, max_freq_ghz, min_volt, max_volt):
        self.mips = mips
        self.cost_per_second = cost_per_second
        self.cost_per_byte = cost_per_byte
        
        # Convert GHz to Hz for calculations
        self.min_freq_hz = min_freq_ghz * 1e9
        self.max_freq_hz = max_freq_ghz * 1e9
        
        self.min_volt = min_volt
        self.max_volt = max_volt

    def get_power(self, running_freq_hz):
        """Calculates the dynamic power in Watts for a given frequency."""
        if self.max_freq_hz == self.min_freq_hz:
            return self.min_volt * self.min_volt * running_freq_hz

        # Linearly interpolate voltage based on frequency
        running_volt = self.min_volt + (self.max_volt - self.min_volt) * \
                       (running_freq_hz - self.min_freq_hz) / (self.max_freq_hz - self.min_freq_hz)
        
        # Power formula: P = f * V^2
        power = running_freq_hz * running_volt * running_volt
        return power

class ProcessingNode:
    def __init__(self, node_id, node_type, spec: ServerSpecs):
        self.id = node_id
        self.type = node_type
        self.spec = spec
        self.processing_power = spec.mips * 1e6 # Convert MIPS to IPS
        self.available_time = 0.0

    def execution_cost(self, instruction_size):
        return instruction_size / self.processing_power

    def execution_energy(self, execution_time):
        # Active energy is calculated based on the power at MAX frequency
        active_power = self.spec.get_power(self.spec.max_freq_hz)
        return active_power * execution_time
    
    def execution_monetary_cost(self, execution_time):
        return self.spec.cost_per_second * execution_time

    def reset(self):
        self.available_time = 0.0

class ResourceCluster:
    def __init__(self):
        self.nodes = {}
        self.uplink_bandwidth_mbps = 100.0
        self.downlink_bandwidth_mbps = 100.0
        self.network_cost_per_byte = 1e-9
        self._initialize_nodes()

    def _initialize_nodes(self):
        # Define all server specifications, including the new physical parameters
        specs = {
            'edge_small':  ServerSpecs(mips=5000,  cost_per_second=0.01, cost_per_byte=2e-9, 
                                      min_freq_ghz=1.0, max_freq_ghz=2.5, min_volt=0.9, max_volt=1.1),
            'edge_medium': ServerSpecs(mips=10000, cost_per_second=0.02, cost_per_byte=2e-9, 
                                      min_freq_ghz=1.2, max_freq_ghz=3.0, min_volt=0.9, max_volt=1.2),
            'edge_large':  ServerSpecs(mips=15000, cost_per_second=0.03, cost_per_byte=2e-9, 
                                      min_freq_ghz=1.5, max_freq_ghz=3.5, min_volt=1.0, max_volt=1.25),
            'cloud_small': ServerSpecs(mips=20000, cost_per_second=0.05, cost_per_byte=1e-9, 
                                      min_freq_ghz=2.0, max_freq_ghz=4.0, min_volt=1.0, max_volt=1.3),
            'cloud_medium':ServerSpecs(mips=40000, cost_per_second=0.10, cost_per_byte=1e-9, 
                                      min_freq_ghz=2.2, max_freq_ghz=4.2, min_volt=1.1, max_volt=1.35),
            'cloud_large': ServerSpecs(mips=60000, cost_per_second=0.15, cost_per_byte=1e-9, 
                                      min_freq_ghz=2.4, max_freq_ghz=4.5, min_volt=1.1, max_volt=1.4)
        }
        
        node_id_counter = 1
        edge_counts = {'small': 5, 'medium': 5, 'large': 5}
        for type_name, count in edge_counts.items():
            for _ in range(count):
                key = f'edge_{type_name}'
                self.nodes[node_id_counter] = ProcessingNode(node_id_counter, key, specs[key])
                node_id_counter += 1

        cloud_counts = {'small': 3, 'medium': 3, 'large': 3}
        for type_name, count in cloud_counts.items():
            for _ in range(count):
                key = f'cloud_{type_name}'
                self.nodes[node_id_counter] = ProcessingNode(node_id_counter, key, specs[key])
                node_id_counter += 1
        
        self.num_nodes = len(self.nodes)

    def transmission_cost(self, from_node_id, to_node_id, data_size_bytes):
        if from_node_id == to_node_id: return 0.0
        rate_mbps = self.uplink_bandwidth_mbps
        rate_bytes_per_sec = rate_mbps * 1e6 / 8.0
        return data_size_bytes / rate_bytes_per_sec if rate_bytes_per_sec > 0 else float('inf')

    def reset(self):
        for node in self.nodes.values():
            node.reset()

class OffloadingEnvironment(object):
    def __init__(self, resource_cluster, batch_size, graph_number, graph_file_paths, time_major, lambda_t=1.0, lambda_e=0.0, encode_dependencies=True):
        self.resource_cluster = resource_cluster
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e
        self.task_graphs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.heft_avg_run_time = -1
        self.encode_dependencies = encode_dependencies

        for graph_file_path in graph_file_paths:
            encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths = \
                self.generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major)
            self.encoder_batchs += encoder_batchs
            self.encoder_lengths += encoder_lengths
            self.task_graphs += task_graph_batchs
            self.decoder_full_lengths += decoder_full_lengths
        self.input_dim = np.array(encoder_batchs[0]).shape[-1]

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
        encoder_list, task_graph_list = [], []
        for i in range(graph_number):
            task_graph = OffloadingTaskGraph(graph_file_path + str(i) + '.gv')
            task_graph_list.append(task_graph)
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)
            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(scheduling_sequence,
                                                                                          self.resource_cluster,
                                                                                          self.encode_dependencies))
            encoder_list.append(task_encode)
        
        encoder_batchs, task_graph_batchs = [], []
        encoder_lengths, decoder_full_lengths = [], []
        for i in range(int(graph_number / batch_size)):
            start, end = i * batch_size, (i + 1) * batch_size
            task_encode_batch = np.array(encoder_list[start:end])
            if time_major:
                task_encode_batch = task_encode_batch.swapaxes(0, 1)
            
            shape = task_encode_batch.shape
            sequence_length = [shape[1]] * shape[0] if not time_major else [shape[0]] * shape[1]
            
            encoder_batchs.append(task_encode_batch)
            task_graph_batchs.append(task_graph_list[start:end])
            encoder_lengths.append(sequence_length)
            decoder_full_lengths.append(sequence_length)
        return encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        self.resource_cluster.reset()
        task_finish_times, task_locations = {}, {}
        total_energy = 0.0
        total_monetary_cost = 0.0
        node_busy_time = defaultdict(float)

        # First pass: Calculate active time, active energy, and makespan
        for task_id, target_node_id in plan:
            task = task_graph.task_list[task_id]
            target_node = self.resource_cluster.nodes[target_node_id]
            ready_time = 0.0
            
            for pred_id in task_graph.pre_task_sets[task_id]:
                pred_finish_time = task_finish_times[pred_id]
                pred_location_id = task_locations[pred_id]
                data_to_transfer = task_graph.get_edge_data_size(pred_id, task_id)

                tx_time = self.resource_cluster.transmission_cost(pred_location_id, target_node_id, data_to_transfer)
                arrival_time = pred_finish_time + tx_time
                ready_time = max(ready_time, arrival_time)
                
                if tx_time > 0:
                    total_monetary_cost += data_to_transfer * self.resource_cluster.network_cost_per_byte
            
            start_time = max(ready_time, target_node.available_time)
            execution_time = target_node.execution_cost(task.processing_data_size)
            finish_time = start_time + execution_time
            
            # Add active computation energy and monetary cost
            total_energy += target_node.execution_energy(execution_time)
            total_monetary_cost += target_node.execution_monetary_cost(execution_time)
            
            # Track node busy time
            node_busy_time[target_node_id] += execution_time
            
            target_node.available_time = finish_time
            task_finish_times[task_id] = finish_time
            task_locations[task_id] = target_node_id

        final_makespan = max(task_finish_times.values()) if task_finish_times else 0.0
        
        # Second pass: Add idle energy for all *used* nodes
        for node_id, busy_time in node_busy_time.items():
            idle_time = final_makespan - busy_time
            if idle_time > 0:
                node = self.resource_cluster.nodes[node_id]
                # Idle power is calculated at the minimum frequency
                idle_power = node.spec.get_power(node.spec.min_freq_hz)
                idle_energy = idle_power * idle_time
                total_energy += idle_energy
        
        # Calculate average resource utilization
        utilization = 0.0
        if final_makespan > 0 and node_busy_time:
            sum_of_ratios = sum((busy_time / final_makespan) for busy_time in node_busy_time.values())
            utilization = (sum_of_ratios / len(node_busy_time)) * 100

        return final_makespan, total_energy, total_monetary_cost, utilization

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        latency_batch, energy_batch, cost_batch, util_batch = [], [], [], []
        
        for action_sequence, task_graph in zip(action_sequence_batch, task_graph_batch):
            plan_sequence = []
            for action_node_id, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action_node_id))
            
            makespan, energy, monetary_cost, utilization = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
            
            latency_batch.append(makespan)
            energy_batch.append(energy)
            cost_batch.append(monetary_cost)
            util_batch.append(utilization)
            
        return latency_batch, energy_batch, cost_batch, util_batch

    def step(self, action_sequence_batch, task_graph_batch):
        
        latency_batch, energy_batch, monetary_cost, utilization = self.get_running_cost(action_sequence_batch, task_graph_batch)
        
        # Reward calculation  based on latency and energy
        cost = self.lambda_t * np.array(latency_batch) + self.lambda_e * np.array(energy_batch)
        
        reward = -cost
        return reward

    def random_solution(self):
        running_cost = []
        for task_graph_batch in self.task_graphs:
            num_tasks = task_graph_batch[0].task_number
            # Node IDs are 1 to num_nodes, so randint needs to be (1, num_nodes + 1)
            plan = np.random.randint(1, self.resource_cluster.num_nodes + 1, size=(len(task_graph_batch), num_tasks))
            running_cost.append(self.get_running_cost(plan, task_graph_batch))
        return running_cost

    def greedy_solution(self, heft=True):
        result_plans, finish_times = [], []
        for task_graph_batch in self.task_graphs:
            batch_plans, batch_finish_times = [], []
            for task_graph in task_graph_batch:
                self.resource_cluster.reset()
                task_finish_times, task_locations, plan = {}, {}, []
                task_sequence = task_graph.prioritize_sequence if heft else np.arange(task_graph.task_number)
                for task_id in task_sequence:
                    task = task_graph.task_list[task_id]
                    earliest_finish_times = []
                    for node_id, node in self.resource_cluster.nodes.items():
                        ready_time = 0.0
                        for pred_id in task_graph.pre_task_sets[task_id]:
                            pred_finish_time = task_finish_times[pred_id]
                            pred_location_id = task_locations[pred_id]
                            data_to_transfer = task_graph.get_edge_data_size(pred_id, task_id)
                            tx_time = self.resource_cluster.transmission_cost(pred_location_id, node_id, data_to_transfer)
                            ready_time = max(ready_time, pred_finish_time + tx_time)
                        start_time = max(ready_time, node.available_time)
                        execution_time = node.execution_cost(task.processing_data_size)
                        finish_time = start_time + execution_time
                        earliest_finish_times.append((finish_time, node_id))
                    
                    best_finish_time, best_node_id = min(earliest_finish_times)
                    task_finish_times[task_id] = best_finish_time
                    task_locations[task_id] = best_node_id
                    self.resource_cluster.nodes[best_node_id].available_time = best_finish_time
                    plan.append((task_id, best_node_id))
                
                batch_plans.append(plan)
                batch_finish_times.append(max(task_finish_times.values()) if task_finish_times else 0)
            result_plans.append(batch_plans)
            finish_times.append(batch_finish_times)
        return result_plans, finish_times

    def calculate_heft_cost(self):
        plans, _ = self.greedy_solution(heft=True)
        # Assuming a single batch for simplicity, hence plans[0] and task_graphs[0]
        heft_latency_batch = self.get_running_cost_by_plan_batch(plans[0], self.task_graphs[0])
        self.heft_avg_run_time = np.mean(heft_latency_batch)
        return self.heft_avg_run_time