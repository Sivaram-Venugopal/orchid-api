import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class SatelliteAvoidanceEnv(gym.Env):
    """
    Custom Gymnasium Environment simulating an active satellite avoiding an incoming debris threat.
    State: [sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz, tca_min, threat_distance_km]
    Action: [delta_v_radial, delta_v_transverse, delta_v_normal] (m/s scale -1 to +1)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(SatelliteAvoidanceEnv, self).__init__()
        
        # Action space: 3 continuous values representing thrust RTN vector (m/s, clamped from -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 8 continuous variables
        self.observation_space = spaces.Box(
            low=np.array([-1e7, -1e7, -1e7, -10000, -10000, -10000, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e7, 1e7, 1e7, 10000, 10000, 10000, 1440.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state parameters: satellite starts in a LEO-like orbit
        r_sat = np.array([7000000.0, 0.0, 0.0], dtype=np.float32)
        v_sat = np.array([0.0, 7500.0, 0.0], dtype=np.float32)
        
        # Conjunction parameters: threat is incoming in 90 minutes with 50 meters separation
        self.tca_min = 90.0
        self.initial_miss_km = 0.05
        
        self.state = np.zeros(8, dtype=np.float32)
        self.state[0:3] = r_sat
        self.state[3:6] = v_sat
        self.state[6] = self.tca_min
        self.state[7] = self.initial_miss_km
        
        return self.state, {}

    def step(self, action):
        # Scale action from [-1, 1] to physical delta-V (max 0.15 m/s)
        dv = action * 0.15
        dv_magnitude = np.linalg.norm(dv)
        
        # Drift time in seconds between burn (t=0) and TCA (t=90 mins = 5400 seconds)
        drift_sec = self.tca_min * 60.0
        omega = 0.0011 # LEO mean motion in rad/s
        
        # Calculate displacement at TCA using Clohessy-Wiltshire (CW) equations
        wt = omega * drift_sec
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        
        # Displacement in Radial-Transverse-Normal local satellite frame
        dr_radial = (dv[0] / omega) * sin_wt + (2 * dv[1] / omega) * (1 - cos_wt)
        dr_transverse = (2 * dv[0] / omega) * (cos_wt - 1) + (dv[1] / omega) * (4 * sin_wt - 3 * wt)
        dr_normal = (dv[2] / omega) * sin_wt
        dr_magnitude = np.sqrt(dr_radial**2 + dr_transverse**2 + dr_normal**2) / 1000.0 # convert to km
        
        # New miss distance at TCA
        final_miss_km = self.initial_miss_km + dr_magnitude
        
        # Reward calculation:
        # 1. Penalize fuel/delta-V mass cost to keep burns optimal
        reward = -50.0 * dv_magnitude
        
        # 2. Reward achieving a safe separation window (> 2.0 km)
        if final_miss_km > 2.0:
            reward += 100.0
        # 3. Heavy penalty for collision course (final miss < 0.2 km)
        elif final_miss_km < 0.2:
            reward -= 1000.0
        else:
            # Linear scaling reward for partial clearance
            reward += (final_miss_km - 0.2) * 50.0
            
        # Update simulation state
        self.tca_min = 0.0
        self.state[6] = self.tca_min
        self.state[7] = final_miss_km
        
        terminated = True # Episode finishes after evaluating the single burn-maneuver step
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

def train():
    print("Initializing Satellite Avoidance RL Gymnasium Environment...")
    env = SatelliteAvoidanceEnv()
    
    print("\nStarting PPO policy training (1500 timesteps)...")
    # Using small network for rapid demonstration deployment
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=256,
        batch_size=64,
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    )
    model.learn(total_timesteps=1500)
    
    model_name = "orchid_leo_impulsive"
    print(f"\nTraining complete. Saving RL weights to {model_name}.zip...")
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name))
    print("[SUCCESS] Reinforcement Learning training pipeline verification complete.")

if __name__ == "__main__":
    train()
