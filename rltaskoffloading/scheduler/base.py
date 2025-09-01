from abc import ABC, abstractmethod
from typing import Dict, Any

class MetaheuristicScheduler(ABC):
    """
    Abstract base class for all metaheuristic scheduling algorithms.
    This class defines a standardized interface for schedulers, ensuring
    they can be used interchangeably within the evaluation framework.
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the scheduler with its configuration and
        algorithm-specific hyperparameters.
        
        Args:
            config (Dict[str, Any]): The global configuration dictionary.
            **kwargs: Additional algorithm-specific hyperparameters.
        """
        self.config = config
        # Store any other kwargs for the specific algorithm
        self.hyperparameters = kwargs

    @abstractmethod
    def schedule(self, dag: Any, node_states: Any) -> (Dict[int, int], Dict[str, float]):
        """
        Computes a complete offloading schedule for a given workflow (DAG).
        This method must be implemented by all concrete scheduler subclasses.

        Args:
            dag (Any): A data structure representing the workflow, its tasks,
                       and their dependencies.
            node_states (Any): A data structure representing the current real-time
                               state of all compute nodes (e.g., load, queue length).

        Returns:
            A tuple containing:
            - A dictionary mapping each task ID in the DAG to a target node ID.
              Example: {0: 2, 1: 5, 2: 2, ...}
            - A dictionary containing the final performance metrics for the schedule.
              Example: {'makespan': 15.7, 'monetary_cost': 0.04}
        """
        pass

    @abstractmethod
    def evaluate_schedule(self, schedule: Any, dag: Any, node_states: Any) -> Dict[str, float]:
        """
        Evaluates a given schedule and returns performance metrics.
        This is often used as the fitness function in metaheuristics.

        Args:
            schedule (Any): A mapping of tasks to compute nodes.
            dag (Any): The workflow's Directed Acyclic Graph.
            node_states (Any): The current state of the compute nodes.

        Returns:
            A dictionary of calculated metrics (e.g., 'makespan', 'monetary_cost').
        """
        pass