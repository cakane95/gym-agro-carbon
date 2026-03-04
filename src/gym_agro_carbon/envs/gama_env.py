"""
GamaAgroCarbonEnv: Gymnasium interface compatible with GAML GymAgent.
Handles GAML-to-JSON string conversion for robust data exchange.
"""
import logging
import numpy as np
import re
import json
from typing import Any, Dict, List, Optional, Tuple

from gym_agro_carbon.envs.grid import AgroCarbonGridEnv, EnvSpec
from gym_agro_carbon.models.context import ContextSpec
from gym_agro_carbon.models.reward import RewardModel

from gym_agro_carbon.envs.gama_client_wrapper import GamaClientWrapper
from gym_agro_carbon.envs.exceptions import GamaCommandError, GamaConnectionError

logger = logging.getLogger(__name__)

class GamaAgroCarbonEnv(AgroCarbonGridEnv):
    """
    Gym environment coupled with GAMA using the high-level GamaClientWrapper.
    """

    def __init__(
        self, 
        env_spec: EnvSpec, 
        context_spec: ContextSpec, 
        reward_model: RewardModel, 
        gaml_file_path: str,
        experiment_name: str = "AgroCarbonSimulation",
        host: str = "localhost", 
        port: int = 6868
    ):
        super().__init__(env_spec, context_spec, reward_model)
        
        self.gaml_file = gaml_file_path
        self.experiment_name = experiment_name
        self.host = host
        self.port = port
        
        # Initialize Wrapper
        self.gama = GamaClientWrapper(host, port)
        self.current_step = 0

    def _ensure_dict(self, raw_data: Any) -> Dict[str, Any]:
        """
        Converts GAML-formatted string {Key=Value} to a Python Dictionary.
        """
        if isinstance(raw_data, dict):
            return raw_data
        
        if isinstance(raw_data, str):
            try:
                # 1. Nettoyage des assignations GAML (= -> :)
                # On utilise regex pour entourer les clés par des guillemets
                clean = re.sub(r'([a-zA-Z_]+)=', r'"\1":', raw_data)
                
                # 2. Conversion des séparateurs de matrices GAML (; -> ,)
                clean = clean.replace(';', ',')
                
                # 3. Conversion des booléens
                clean = clean.replace('false', 'false').replace('true', 'true')
                
                return json.loads(clean)
            except Exception as e:
                logger.error(f"Failed to parse GAML string: {raw_data}")
                raise GamaCommandError(f"Data format error: {e}")
        
        return {}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment and retrieves the initial state from GAMA GymAgent.
        """
        # Parameters matching your GAML experiment
        params = [
            {"name": "S", "value": self.env_spec.S, "type": "int"},
            {"name": "M", "value": self.env_spec.M, "type": "int"},
            {"name": "Alpha (Reward Trade-off): ", "value": self.env_spec.alpha, "type": "float"}
        ]

        try:
            self.gama.close() # Ensure clean restart
            self.gama.load_experiment(self.gaml_file, self.experiment_name, params)
            
            if seed is not None:
                self.gama._execute_expression(f"seed <- {seed};")
            
            # Read initial data from the agent
            raw_response = self.gama._execute_expression("gym_interface.data")
            data = self._ensure_dict(raw_response)
            
            # Reshape observation (10x10)
            obs_flat = np.array(data.get("State", []), dtype=np.int32)
            if obs_flat.size == 0:
                obs = np.zeros((self.env_spec.H, self.env_spec.W), dtype=np.int32)
            else:
                obs = obs_flat.reshape((self.env_spec.H, self.env_spec.W))
            
            self.current_step = 0
            info = {"status": "Reset Done", "exp_id": self.gama.experiment_id}
            
            return obs, info

        except Exception as e:
            logger.error(f"GAMA Reset Error: {e}")
            raise GamaConnectionError(f"Reset failed: {e}")

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step: sends actions, steps GAMA, and returns new state.
        """
        # 1. Flatten actions for GAMA (10x10 -> 100)
        actions_list = actions.flatten().tolist()

        try:
            # 2. Sync call: Set next_action -> Step -> Get data
            raw_data = self.gama.execute_step(actions_list)
            data = self._ensure_dict(raw_data)
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise GamaCommandError(f"Step failed: {e}")

        # 3. Extract and Reshape Observations
        obs_flat = np.array(data["State"], dtype=np.int32)
        obs = obs_flat.reshape((self.env_spec.H, self.env_spec.W))
        
        reward = float(data["Reward"])
        
        # 4. Extract spatial Info maps
        info_raw = data.get("Info", {})
        info = {
            "soc_map": np.array(info_raw.get("soc_map", [])).reshape((self.env_spec.H, self.env_spec.W)),
            "yield_map": np.array(info_raw.get("yield_map", [])).reshape((self.env_spec.H, self.env_spec.W)),
        }

        self.current_step += 1
        terminated = bool(data.get("Terminated", False)) or (self.current_step >= self.env_spec.T)
        truncated = bool(data.get("Truncated", False))
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Properly shutdown the GAMA connection."""
        if self.gama:
            self.gama.close()
        super().close()