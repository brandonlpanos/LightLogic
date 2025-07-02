import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Callback to record the environment's parameters at each step
# Assumes the environment puts the parameters in the info dict under the key 'params'.
class ParameterHistoryCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(ParameterHistoryCallback, self).__init__(verbose)
        self.save_path = save_path
        self.param_history = [] # Store parameters from all environments concatenated together
        self.total_env_steps = 0 # Keep track of steps across all environments

    # Method is called by the model after each call to env.step()
    # For SAC, this represents N steps (N environments) per call
    def _on_step(self) -> bool:
        # self.locals contains information about the training process
        # infos is a list of info dicts, one for each environment
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                # Check if 'params' key exists
                # Check for terminal observation info 
                if 'params' in info:
                    params = info['params']
                    # Append parameters (numpy array)
                    self.param_history.append(params)
                    self.total_env_steps += 1
        # Return True to continue training
        return True

    # This event is triggered before exiting the learn() method
    # Save the recorded parameter history
    def _on_training_end(self) -> None:
        print(f"\nSaving parameter history ({len(self.param_history)} steps recorded) to {self.save_path}")
        param_array = np.array(self.param_history) # Convert list of arrays into a single 2D (numpy array)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.save(self.save_path, param_array)
        print(f"Parameter history saved with shape: {param_array.shape}")