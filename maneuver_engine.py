import numpy as np
import os
from physics import tle_to_state, tsiolkovsky
from stable_baselines3 import PPO

_model = None

def load_model():
    global _model
    if _model is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'orchid_v4_best')
            _model = PPO.load(model_path)
        except Exception as e:
            print(f"Model load failed: {e}. Using fallback.")
            _model = None  # Fallback to simple maneuver
    return _model

def generate_maneuver(satellite, conjunctions):
    import logging
    logger = logging.getLogger(__name__)
    sat_state = tle_to_state(satellite.tle1, satellite.tle2)
    model = load_model()
    if model is None:
        logger.warning("Using fallback maneuver (no RL model)")
        # Simple fallback: boost perpendicular to closest threat
        closest_dist = conjunctions[0]["distance_km"] if conjunctions else 1.0
        dv = np.array([0.0, 0.01 * closest_dist, 0.005])
    else:
        obs = np.zeros(8 + 10*9 + 2, dtype=np.float32)
        obs[:3] = sat_state[:3] / 7000
        obs[3:6] = sat_state[3:] / 8
        obs[6] = 1.0
        obs[7] = 1.0
        for i, conj in enumerate(conjunctions[:10]):
            obs[8 + i*9] = conj["distance_km"] / 5.0
        action, _ = model.predict(obs, deterministic=True)
        dv = action * 0.015
    DRY_MASS = 500.0
    INIT_FUEL = 50.0
    fuel_cost = tsiolkovsky(np.linalg.norm(dv), DRY_MASS + INIT_FUEL)
    from datetime import datetime, timezone, timedelta
    burn_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    return {
        "delta_v_rtn": dv.tolist(),
        "burn_time_utc": burn_time.isoformat(),
        "fuel_cost_kg": round(float(fuel_cost), 4),
        "post_maneuver_safety_km": round(float(conjunctions[0]["distance_km"] * 2.5 if conjunctions else 2.5), 3),
        "note": "fallback" if model is None else "rl_model"
    }
