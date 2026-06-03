import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class OrbitalManeuverEnv(gym.Env):
    """
    Custom Gym Environment for training satellite collision avoidance policies.
    Supports LEO (impulsive/discrete delta-V) and GEO (continuous low-thrust electric).
    """
    def __init__(self, regime="LEO"):
        super(OrbitalManeuverEnv, self).__init__()
        self.regime = regime
        
        # State vector: [x, y, z, vx, vy, vz, distance_to_debris, fuel_left]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, -2.0, -2.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        if regime == "LEO":
            # Action space LEO: Impulsive burn [delta_v_radial, delta_v_transverse, delta_v_normal] (m/s)
            self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(3,), dtype=np.float32)
        else:
            # Action space GEO: Continuous thrust direction and magnitude [u_r, u_t, u_n, thrust_magnitude]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 5.0, 1.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}
        
    def step(self, action):
        self.steps += 1
        
        # Simple dynamics propagation simulation
        if self.regime == "LEO":
            # LEO: Burn instantly increases relative separation and burns fuel
            dv_norm = np.linalg.norm(action)
            separation = self.state[6] + dv_norm * 5.0  # simple drift scaling
            fuel_used = dv_norm * 0.1
        else:
            # GEO: Continuous low thrust builds separation slowly over time steps
            thrust_magnitude = action[3]
            separation = self.state[6] + thrust_magnitude * 0.25
            fuel_used = thrust_magnitude * 0.02
            
        self.state[6] = min(100.0, separation)
        self.state[7] = max(0.0, self.state[7] - fuel_used)
        
        # Reward function: maximize separation, minimize fuel, penalize close approaches
        reward = self.state[6] * 10.0 - fuel_used * 100.0
        if self.state[6] < 0.2:  # Collision threshold
            reward -= 1000.0
            done = True
        else:
            done = self.steps >= 20
            
        return self.state, reward, done, False, {}

def train_policies():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(base_dir, "orchid_v4_best.zip")
    
    # 1. Train LEO Impulsive Model
    print("Initializing training for LEO Impulsive policy...")
    leo_env = OrbitalManeuverEnv(regime="LEO")
    
    # Transfer learning: attempt to load from existing best model base weights
    if os.path.exists(best_model_path):
        print("Transfer learning active: loading weights from orchid_v4_best...")
        leo_model = PPO.load(best_model_path, env=leo_env)
    else:
        leo_model = PPO("MlpPolicy", leo_env, verbose=1, learning_rate=3e-4)
        
    leo_model.learn(total_timesteps=10000)
    leo_model.save(os.path.join(base_dir, "orchid_leo_impulsive"))
    print("LEO Impulsive policy trained and saved successfully.")
    
    # 2. Train GEO Continuous Electric Model
    print("Initializing training for GEO Continuous Electric policy...")
    geo_env = OrbitalManeuverEnv(regime="GEO")
    
    # Transfer learning: load base weights
    if os.path.exists(best_model_path):
        print("Transfer learning active: loading weights from orchid_v4_best for GEO...")
        geo_model = PPO.load(best_model_path, env=geo_env)
    else:
        geo_model = PPO("MlpPolicy", geo_env, verbose=1, learning_rate=1e-4)
        
    geo_model.learn(total_timesteps=10000)
    geo_model.save(os.path.join(base_dir, "orchid_geo_electric"))
    print("GEO Continuous Electric policy trained and saved successfully.")

if __name__ == "__main__":
    train_policies()
