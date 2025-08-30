import random
import yaml
from enum import Enum
from typing import Dict, Any

# Using an Enum makes the code safer and more readable than using raw strings.
class OffloadingTarget(Enum):
    FOG = "FOG"
    CLOUD = "CLOUD"

class DRLOffloader:
    """
    Represents the DRL agent responsible for making the high-level
    offloading decision: whether to place the workflow in the Fog or the Cloud.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DRL Offloader with a configuration dictionary.

        Args:
            config (Dict[str, Any]): The application configuration.
        """
        self.config = config.get("drl_offloader", {})
        self.model_path = config.get("app", {}).get("model_path")
        self._load_model()

    def _load_model(self):
        """
        Loads the trained DRL model.
        
        NOTE: This is a placeholder. In a real implementation, you would use
        a library like PyTorch or TensorFlow to load the model weights from
        the `self.model_path` file.
        """
        print(f"DRL Offloader: Attempting to load model from '{self.model_path}'...")
        # self.model = YourDRLModelClass.load(self.model_path)
        self.model = "FAKE_MODEL" # Placeholder for the loaded model object
        if self.model:
            print("DRL Offloader: Model loaded successfully (simulation).")
        else:
            print("DRL Offloader: ⚠️ Model not found, will make random decisions.")

    def decide_offloading_target(self, task_graph: Any) -> OffloadingTarget:
        """
        Makes the offloading decision for an entire workflow.

        Args:
            task_graph (Any): The workflow's task graph, used to create a state
                              representation for the DRL agent.

        Returns:
            OffloadingTarget: The target layer enum (FOG or CLOUD).
        """
        print("\nDRL Offloader: Making a high-level decision (Fog vs. Cloud)...")

        # 1. Check for a forced decision from the config file (for testing/debugging)
        bias_str = self.config.get("decision_bias")
        if bias_str and bias_str in [target.value for target in OffloadingTarget]:
            target = OffloadingTarget(bias_str)
            print(f"DRL Offloader: Config bias forces decision => {target.value}")
            return target

        # 2. Use the DRL model for inference (if it was loaded)
        if self.model:
            print("DRL Offloader: Using DRL model for inference...")
            # state = self._create_state_from_graph(task_graph)
            # action, _ = self.model.predict(state, deterministic=True)
            # decision_index = action_result # Replace with your model's output
            
            # --- SIMULATION of model prediction ---
            decision_index = random.choice([0, 1]) 
            # ------------------------------------
            
            target = OffloadingTarget.FOG if decision_index == 0 else OffloadingTarget.CLOUD
            print(f"DRL Offloader: Model chose => Execute workflow in the '{target.value}' layer.")
            return target
        
        # 3. Fallback to a random choice if no model is available
        print("DRL Offloader: ⚠️ No model loaded. Making a random decision.")
        target = random.choice(list(OffloadingTarget))
        print(f"DRL Offloader: Random choice => Execute workflow in the '{target.value}' layer.")
        return target