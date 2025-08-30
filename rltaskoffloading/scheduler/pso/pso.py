import random
import numpy as np
import yaml
from .base_scheduler import MetaheuristicScheduler
from typing import Dict, Any, List, Optional


class Particle:
    """A helper class to represent a single particle in the swarm."""
    def __init__(self, num_tasks: int, target_vm_ids: List[int]):
        self.num_tasks = num_tasks
        self.target_vm_ids = target_vm_ids
        
        # Position represents a schedule (task -> vm_id mapping)
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
    def __init__(self, env: Any, config: Dict[str, Any], target_vm_ids: Optional[List[int]] = None):
        """
        Initializes the PSO scheduler.

        Args:
            env (Any): The simulation environment.
            config (Dict[str, Any]): The 'pso' section of the global config.
            target_vm_ids (Optional[List[int]], optional): A filtered list of VM IDs to schedule on.
        """
        super().__init__(env, target_vm_ids)
        
        pso_config = config.get("pso", {})
        self.num_particles = pso_config.get("num_particles", 30)
        self.iterations = pso_config.get("num_iterations", 100)
        self.w = pso_config.get("w", 0.7)       # Inertia weight
        self.c1 = pso_config.get("c1", 1.5)     # Cognitive coefficient
        self.c2 = pso_config.get("c2", 1.5)     # Social coefficient
        self.alpha = pso_config.get("alpha", 0.5)  # Weight for cost vs makespan

    def schedule(self):
        self._prepare_for_scheduling()
        
        swarm = [Particle(self.num_tasks, self.target_vm_ids) for _ in range(self.num_particles)]
        
        global_best_position = None
        global_best_fitness = float('inf')

        print(f"\nRunning PSO with {self.num_particles} particles for {self.iterations} iterations...")
        for i in range(self.iterations):
            for particle in swarm:
                schedule = particle.position
                fitness = self._calculate_fitness(schedule)
                
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(schedule)
                
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = np.copy(schedule)

            for particle in swarm:
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

                new_position_float = particle.position + particle.velocity
                new_position_discrete = [
                    min(self.target_vm_ids, key=lambda vm_id: abs(vm_id - pos))
                    for pos in new_position_float
                ]
                particle.position = np.array(new_position_discrete)

            if (i + 1) % 10 == 0: # Print progress every 10 iterations
                print(f"  -> PSO Iteration {i+1}/{self.iterations}, Global Best Fitness: {global_best_fitness:.4f}")

        final_metrics = self.evaluate_schedule(global_best_position)
        return global_best_position, final_metrics

    def _calculate_fitness(self, schedule: np.ndarray) -> float:
        """
        Fitness function combining monetary cost and makespan.
        Objective: Minimize -> (alpha * monetary_cost) + ((1 - alpha) * makespan)
        """
        metrics = self.evaluate_schedule(schedule)
        # Note: The original code used metrics["cost"], which can be ambiguous.
        # We assume it refers to monetary cost, as this is a common objective.
        cost_component = self.alpha * metrics["monetary_cost"]
        makespan_component = (1 - self.alpha) * metrics["makespan"]
        return cost_component + makespan_component