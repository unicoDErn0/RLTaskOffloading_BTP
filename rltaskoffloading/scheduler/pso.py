import random
import numpy as np
import yaml
import networkx as nx  # Used for topological sort
from .base import MetaheuristicScheduler
from typing import Dict, Any, List, Optional, Tuple


class Particle:
    """A helper class to represent a single particle in the swarm."""
    def __init__(self, num_tasks: int, target_vm_ids: List[int]):
        self.num_tasks = num_tasks
        self.target_vm_ids = target_vm_ids
        
        # Position represents a schedule (task_id -> vm_id mapping)
        # The index of the array is the task_id.
        self.position = np.array([random.choice(self.target_vm_ids) for _ in range(self.num_tasks)])
        
        # Initialize velocity with small random values
        self.velocity = np.random.uniform(-1, 1, self.num_tasks)
        
        # Personal best trackers
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')


class ParticleSwarmOptimizationScheduler(MetaheuristicScheduler):
    """
    An implementation of the Particle Swarm Optimization (PSO) algorithm for
    task scheduling.
    """
    def __init__(self, env: Any, config: Dict[str, Any], target_vm_ids: List[int]):
        """
        Initializes the PSO scheduler.

        Args:
            env (Any): The simulation environment, providing access to system state.
            config (Dict[str, Any]): The global configuration dictionary.
            target_vm_ids (List[int]): A filtered list of VM IDs to schedule on.
        """
        # Correctly call the superclass constructor
        super().__init__(config, target_vm_ids=target_vm_ids)
        self.env = env
        self.target_vm_ids = target_vm_ids
        
        pso_config = config.get("pso", {})
        self.num_particles = pso_config.get("num_particles", 30)
        self.iterations = pso_config.get("num_iterations", 100)
        self.w = pso_config.get("w", 0.7)       # Inertia weight
        self.c1 = pso_config.get("c1", 1.5)     # Cognitive coefficient
        self.c2 = pso_config.get("c2", 1.5)     # Social coefficient
        self.alpha = pso_config.get("alpha", 0.5)  # Weight for cost vs makespan

    def _prepare_for_scheduling(self, dag: Any, node_states: Any):
        """Sets instance variables needed for the scheduling process."""
        self.dag = dag
        self.node_states = node_states
        # Assuming task IDs are integers from 0 to n-1
        self.num_tasks = len(dag.nodes)
        self.task_order = list(nx.topological_sort(dag))

    def schedule(self, dag: Any, node_states: Any) -> (Dict[int, int], Dict[str, float]):
        """Runs the PSO algorithm to find an optimal schedule."""
        self._prepare_for_scheduling(dag, node_states)
        
        swarm = [Particle(self.num_tasks, self.target_vm_ids) for _ in range(self.num_particles)]
        
        global_best_position = None
        global_best_fitness = float('inf')

        print(f"\nRunning PSO with {self.num_particles} particles for {self.iterations} iterations...")
        for i in range(self.iterations):
            for particle in swarm:
                # The particle's position is the schedule to be evaluated
                schedule = particle.position
                fitness = self._calculate_fitness(schedule, dag, node_states)
                
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(schedule)
                
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = np.copy(schedule)

            # Update particle velocities and positions
            for particle in swarm:
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

                # Update position based on new velocity
                new_position_float = particle.position + particle.velocity
                # Map the continuous position back to the discrete set of available VM IDs
                new_position_discrete = [
                    min(self.target_vm_ids, key=lambda vm_id: abs(vm_id - pos))
                    for pos in new_position_float
                ]
                particle.position = np.array(new_position_discrete)

            if (i + 1) % 10 == 0:
                print(f"  -> PSO Iteration {i+1}/{self.iterations}, Global Best Fitness: {global_best_fitness:.4f}")

        final_metrics = self.evaluate_schedule(global_best_position, dag, node_states)
        # Convert final schedule from numpy array to a dictionary for standardized output
        final_schedule_dict = {task_id: int(vm_id) for task_id, vm_id in enumerate(global_best_position)}
        
        return final_schedule_dict, final_metrics

    def evaluate_schedule(self, schedule: np.ndarray, dag: Any, node_states: Any) -> Dict[str, float]:
        """
        Simulates the execution of a schedule to calculate makespan and cost.
        This is the concrete implementation of the abstract method.
        """
        task_finish_times = {}
        vm_available_times = {vm_id: 0.0 for vm_id in self.target_vm_ids}
        total_monetary_cost = 0.0

        for task_id in self.task_order:
            vm_id = int(schedule[task_id])
            
            # --- Calculate Data Reception Time (Task Ready Time) ---
            # The task is ready to start only after all its parent tasks have
            # finished and transferred their output data.
            ready_time = 0.0
            for parent_id in dag.predecessors(task_id):
                parent_vm_id = int(schedule[parent_id])
                
                # Data transfer time = data_size / bandwidth
                data_size = dag.edges[parent_id, task_id]['weight']
                bandwidth = self.env.get_bandwidth(parent_vm_id, vm_id)
                transfer_time = data_size / bandwidth if bandwidth > 0 else float('inf')
                
                # The data arrives at the finish time of the parent + transfer time
                arrival_time = task_finish_times[parent_id] + transfer_time
                if arrival_time > ready_time:
                    ready_time = arrival_time

            # --- Calculate Execution Time ---
            task_workload = dag.nodes[task_id]['workload']
            vm_mips = self.env.get_vm(vm_id).mips
            execution_time = task_workload / vm_mips if vm_mips > 0 else float('inf')

            # --- Determine Start and Finish Times ---
            # Task starts at the maximum of its data ready time and VM available time
            start_time = max(ready_time, vm_available_times[vm_id])
            finish_time = start_time + execution_time
            
            # Update tracking dictionaries
            task_finish_times[task_id] = finish_time
            vm_available_times[vm_id] = finish_time

            # --- Accumulate Monetary Cost ---
            # Cost = execution_time * vm_cost_per_second
            vm_cost_rate = self.env.get_vm(vm_id).cost_per_sec
            total_monetary_cost += execution_time * vm_cost_rate

        # Makespan is the finish time of the last task
        makespan = max(task_finish_times.values()) if task_finish_times else 0.0
        
        return {"makespan": makespan, "monetary_cost": total_monetary_cost}


    def _calculate_fitness(self, schedule: np.ndarray, dag: Any, node_states: Any) -> float:
        """
        Fitness function combining monetary cost and makespan.
        The objective is to minimize this value.
        """
        metrics = self.evaluate_schedule(schedule, dag, node_states)
        cost_component = self.alpha * metrics["monetary_cost"]
        makespan_component = (1 - self.alpha) * metrics["makespan"]
        return cost_component + makespan_component