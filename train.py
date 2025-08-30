import argparse
import yaml
from typing import Dict, Any

from rltaskoffloading.offloading_ddqn.lstm_ddqn import DDQNTO_number, DDQNTO_trans
from rltaskoffloading.offloading_ppo.offloading_ppo import DRLTO_number, DRLTO_trans

def train(config: Dict[str, Any]):
    """Main training function driven by a config dictionary."""
    
    train_cfg = config.get("training", {})
    app_cfg = config.get("app", {})

    algo = train_cfg.get("algo")
    scenario = train_cfg.get("scenario")
    goal = train_cfg.get("goal")
    dependency = train_cfg.get("dependency")
    
    # Construct the log path from config values
    logpath = f"{app_cfg.get('log_path')}-{algo}-{scenario}-{goal}-dependency-{dependency}"
    print(f"Starting training for Algo={algo}, Scenario={scenario}, Goal={goal}")
    print(f"Logs will be saved to: {logpath}")

    # Select the correct training and test graph paths based on the scenario
    if scenario == "Number":
        train_paths = train_cfg.get("graph_paths_train_number")
        test_paths = train_cfg.get("graph_paths_test_number")
        trainer_function = DDQNTO_number if algo == "DDQNTO" else DRLTO_number
    elif scenario == "Trans":
        train_paths = train_cfg.get("graph_paths_train_trans")
        test_paths = train_cfg.get("graph_paths_test_trans")
        trainer_function = DDQNTO_trans if algo == "DDQNTO" else DRLTO_trans
    else:
        raise ValueError(f"Invalid scenario specified in config: {scenario}")

    # Set lambda weights based on the goal
    lambda_t = 1.0 if goal == "LO" else 0.5
    lambda_e = 0.0 if goal == "LO" else 0.5

    # Prepare arguments for the trainer function
    trainer_args = {
        "lambda_t": lambda_t,
        "lambda_e": lambda_e,
        "logpath": logpath,
        "encode_dependencies": dependency,
        "train_graph_file_paths": train_paths,
        "test_graph_file_paths": test_paths,
    }
    
    # Add bandwidths argument only for the 'Trans' scenario
    if scenario == "Trans":
        trainer_args["bandwidths"] = train_cfg.get("bandwidths")

    # Call the selected trainer function with the prepared arguments
    trainer_function(**trainer_args)
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DRL model training process.")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help="Path to the main configuration file."
    )
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    try:
        with open(args.config, "r") as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
        
    train(main_config)