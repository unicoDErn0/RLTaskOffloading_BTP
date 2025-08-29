import numpy as np
from graphviz import Digraph
import pydotplus

class OffloadingTask:
    def __init__(self, id_name, process_data_size, transmission_data_size, type_name, depth=0):
        self.id_name = id_name
        self.processing_data_size = process_data_size
        self.transmission_data_size = transmission_data_size
        self.type_name = type_name
        self.depth = depth

class OffloadingDotParser:
    def __init__(self, file_name, is_matrix=False):
        self.succ_task_for_ids = {}
        self.pre_task_for_ids = {}
        self.dot_ob = pydotplus.graphviz.graph_from_dot_file(file_name)
        self._parse_task()
        self._parse_dependecies()
        self._calculate_depth_and_transimission_datasize()

    def _parse_task(self):
        jobs = self.dot_ob.get_node_list()
        self.task_list = [0] * len(jobs)
        for job in jobs:
            job_id = job.get_name()
            data_size = int(eval(job.obj_dict['attributes']['size']))
            communication_data_size = int(eval(job.obj_dict['attributes']['expect_size']))
            task = OffloadingTask(job_id, data_size, 0, "compute")
            task.transmission_data_size = communication_data_size
            id = int(job_id) - 1
            self.task_list[id] = task

    def _parse_dependecies(self):
        edge_list = self.dot_ob.get_edge_list()
        dependencies = []
        task_number = len(self.task_list)
        for i in range(len(self.task_list)):
            self.pre_task_for_ids[i] = []
            self.succ_task_for_ids[i] = []
        for edge in edge_list:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            self.pre_task_for_ids[destination_id].append(source_id)
            self.succ_task_for_ids[source_id].append(destination_id)

    def _calculate_depth_and_transimission_datasize(self):
        ids_to_depth = dict()
        def caluclate_depth_value(id):
            if id in ids_to_depth.keys():
                return ids_to_depth[id]
            else:
                if len(self.pre_task_for_ids[id]) != 0:
                    depth = 1 + max([caluclate_depth_value(pre_task_id) for pre_task_id in self.pre_task_for_ids[id]])
                else:
                    depth = 0
                ids_to_depth[id] = depth
            return ids_to_depth[id]
        for id in range(len(self.task_list)):
            ids_to_depth[id] = caluclate_depth_value(id)
        for id, depth in ids_to_depth.items():
            self.task_list[id].depth = depth

class OffloadingTaskGraph(object):
    def __init__(self, file_name, is_matrix=False):
        self._parse_from_dot(file_name, is_matrix)

    def _parse_from_dot(self, file_name, is_matrix):
        parser = OffloadingDotParser(file_name, is_matrix)
        task_list = parser.generate_task_list()
        self.task_number = len(task_list)
        self.dependency = np.zeros((self.task_number, self.task_number))
        self.task_list = []
        self.prioritize_sequence = []
        self.pre_task_sets = []
        self.succ_task_sets = []
        self.edge_set = []

        for i in range(self.task_number):
            self.pre_task_sets.append(set(parser.pre_task_for_ids[i]))
            self.succ_task_sets.append(set(parser.succ_task_for_ids[i]))
            
        self.add_task_list(task_list)

        edge_list = parser.dot_ob.get_edge_list()
        for edge in edge_list:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            data_size = int(eval(edge.obj_dict['attributes']['size']))
            self.add_dependency(source_id, destination_id, data_size)

    def add_task_list(self, task_list):
        self.task_list = task_list
        for i in range(len(self.task_list)):
            self.dependency[i][i] = task_list[i].processing_data_size

    def add_dependency(self, pre_task_index, succ_task_index, transmission_cost):
        self.dependency[pre_task_index][succ_task_index] = transmission_cost
        edge = [pre_task_index, self.task_list[pre_task_index].depth, self.task_list[pre_task_index].processing_data_size,
                transmission_cost, succ_task_index, self.task_list[succ_task_index].depth, self.task_list[succ_task_index].processing_data_size]
        self.edge_set.append(edge)

    def get_edge_data_size(self, pre_task_index, succ_task_index):
        return self.dependency[pre_task_index][succ_task_index]

    def encode_point_sequence_with_cost(self, resource_cluster, encode_dependencies=True):
        point_sequence = []
        edge_nodes = [node for node in resource_cluster.nodes.values() if 'edge' in node.type]
        cloud_nodes = [node for node in resource_cluster.nodes.values() if 'cloud' in node.type]

        for i in range(self.task_number):
            task = self.task_list[i]
            
            edge_costs = [node.execution_cost(task.processing_data_size) for node in edge_nodes]
            avg_edge_cost = np.mean(edge_costs) if edge_costs else 0
            min_edge_cost = np.min(edge_costs) if edge_costs else 0

            cloud_costs = [node.execution_cost(task.processing_data_size) for node in cloud_nodes]
            avg_cloud_cost = np.mean(cloud_costs) if cloud_costs else 0
            min_cloud_cost = np.min(cloud_costs) if cloud_costs else 0

            task_embeding_vector = [i, avg_edge_cost, min_edge_cost, avg_cloud_cost, min_cloud_cost]

            pre_task_index_set = list(self.pre_task_sets[i])
            succs_task_index_set = list(self.succ_task_sets[i])

            while len(pre_task_index_set) < 6:
                pre_task_index_set.append(-1.0)
            while len(succs_task_index_set) < 6:
                succs_task_index_set.append(-1.0)

            if encode_dependencies:
                point_vector = task_embeding_vector + pre_task_index_set[:6] + succs_task_index_set[:6]
            else:
                point_vector = task_embeding_vector
            
            point_sequence.append(point_vector)
        return point_sequence

    def encode_point_sequence_with_ranking_and_cost(self, sorted_task, resource_cluster, encode_dependencies=True):
        point_sequence = self.encode_point_sequence_with_cost(resource_cluster, encode_dependencies=encode_dependencies)
        prioritize_point_sequence = [point_sequence[task_id] for task_id in sorted_task]
        return prioritize_point_sequence

    def prioritize_tasks(self, resource_cluster):
        w = [0] * self.task_number
        for i, task in enumerate(self.task_list):
            costs = [node.execution_cost(task.processing_data_size) for node in resource_cluster.nodes.values()]
            w[i] = np.mean(costs)

        rank_dict = {}
        def rank(task_index):
            if task_index in rank_dict:
                return rank_dict[task_index]
            if not self.succ_task_sets[task_index]:
                rank_dict[task_index] = w[task_index]
                return rank_dict[task_index]
            else:
                max_succ_rank = max(rank(j) for j in self.succ_task_sets[task_index])
                rank_dict[task_index] = w[task_index] + max_succ_rank
                return rank_dict[task_index]

        for i in range(self.task_number):
            rank(i)
        
        sorted_tasks = sorted(range(self.task_number), key=lambda i: rank_dict[i], reverse=True)
        self.prioritize_sequence = sorted_tasks
        return self.prioritize_sequence

    def render(self, path):
        dot = Digraph(comment='DAG')
        for i in range(self.task_number):
            dot.node(str(i), str(i) + ":" + str(self.task_list[i].processing_data_size))
        for e in self.edge_set:
            dot.edge(str(e[0]), str(e[4]), constraint='true', label="%.2f" % e[3])
        dot.render(path, view=False)