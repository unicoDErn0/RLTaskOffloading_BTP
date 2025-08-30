import argparse
import yaml
from typing import Dict, Any

from rltaskoffloading.environment.offloading_env import ResourceCluster, OffloadingEnvironment
from rltaskoffloading.schedulers.drl_offloader import DRLOffloader, OffloadingTarget
from rltaskoffloading.schedulers.pso import ParticleSwarmOptimizationScheduler
# Import other schedulers like GA, Greedy as you implement them
# from rltaskoffloading.schedulers.ga import GeneticAlgorithmScheduler
# from rltaskoffloading.schedulers.greedy import GreedyScheduler

def print_results(scheduler_name: str, offloading_target: OffloadingTarget, metrics: Dict[str, float]):
    """Helper function to print the final metrics in a formatted way."""
    print("\n" + "="*55)
    print(f"    Results for {scheduler_name} in '{offloading_target.value}' Layer")
    print("="*55)
    print(f"-> Makespan              : {metrics['makespan']:.4f} seconds")
    print(f"-> Total Energy          : {metrics['energy']:.4f} Joules")
    print(f"-> Monetary Cost         : ${metrics['monetary_cost']:.4f}")
    print(f"-> Avg Resource Utilization: {metrics['resource_utilization']:.2f}%")
    print("="*55 + "\n")

def main(config: Dict[str, Any]):
    """Main execution function driven by a config dictionary."""
    
    app_cfg = config.get("app", {})
    
    # --- 1. Environment and Offloader Setup ---
    print("Initializing environment and DRL Offloader...")
    resource_cluster = ResourceCluster()
    env = OffloadingEnvironment(resource_cluster, graph_file_path=app_cfg.get("graphs_path"))
    
    # The DRL Offloader is initialized with the full config
    drl_offloader = DRLOffloader(config)
    
    env.reset()
    task_graph = env.task_graph
    print(f"Loaded workflow with {len(task_graph.task_list)} tasks.")

    # --- 2. Stage 1: DRL Offloading Decision ---
    offloading_target = drl_offloader.decide_offloading_target(task_graph)

    # --- 3. Prepare for Stage 2 Scheduling ---
    if offloading_target == OffloadingTarget.FOG:
        target_vm_ids = resource_cluster.get_vm_ids_by_type('edge')
    else: # CLOUD
        target_vm_ids = resource_cluster.get_vm_ids_by_type('cloud')

    if not target_vm_ids:
        print(f"Error: No VMs found for the target layer '{offloading_target.value}'. Exiting.")
        return

    # --- 4. Stage 2: Metaheuristic Scheduling ---
    scheduler_name = config.get("scheduler", "pso") # Default to pso if not specified
    scheduler = None
    
    print(f"\nInitializing '{scheduler_name.upper()}' scheduler for the '{offloading_target.value}' layer...")
    if scheduler_name == 'pso':
        scheduler = ParticleSwarmOptimizationScheduler(env, config=config, target_vm_ids=target_vm_ids)
    # elif scheduler_name == 'ga':
    #     scheduler = GeneticAlgorithmScheduler(env, config=config, target_vm_ids=target_vm_ids)
    # elif scheduler_name == 'greedy':
    #     scheduler = GreedyScheduler(env, config=config, target_vm_ids=target_vm_ids)
    else:
        print(f"Error: Scheduler '{scheduler_name}' is not supported. Exiting.")
        return

    # The scheduler runs its optimization on the filtered set of resources
    best_schedule, final_metrics = scheduler.schedule()
    print_results(scheduler.__class__.__name__, offloading_target, final_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a two-stage DRL Offloading + Metaheuristic Scheduling process.")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help="Path to the main configuration file."
    )
    parser.add_argument(
        '--scheduler', type=str, choices=['pso', 'ga', 'greedy'],
        help="Override the scheduler specified in the config file."
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, "r") as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
        
    # Allow command-line override for the scheduler
    if args.scheduler:
        main_config['scheduler'] = args.scheduler

    main(main_config)