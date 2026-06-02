import numpy as np
import os
from physics import tle_to_state, tsiolkovsky

_models = {}

def get_orbital_regime(tle2: str) -> str:
    """Computes orbital regime (LEO vs GEO vs MEO) based on TLE line 2 parameters."""
    try:
        from risk_engine import parse_apogee_perigee
        perigee, apogee = parse_apogee_perigee(tle2)
        avg_alt = (perigee + apogee) / 2.0
        
        if avg_alt < 2000.0:
            return "LEO"
        elif 35000.0 <= avg_alt <= 36500.0:
            return "GEO"
        else:
            return "MEO"
    except Exception:
        return "LEO"  # Default fallback

def load_model_for_regime(regime: str):
    """Loads appropriate stable-baselines3 model for the given orbital regime."""
    global _models
    if regime not in _models:
        try:
            from stable_baselines3 import PPO
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Map regimes to model zip filenames
            model_map = {
                "LEO": "orchid_leo_impulsive",
                "GEO": "orchid_geo_electric",
                "MEO": "orchid_v4_best"
            }
            
            model_name = model_map.get(regime, "orchid_v4_best")
            model_path = os.path.join(base_dir, model_name)
            
            if os.path.exists(model_path + '.zip'):
                print(f"Loading RL Policy Model for {regime} from {model_name}.zip...")
                _models[regime] = PPO.load(model_path)
            else:
                # Try to load the general best model as fallback
                fallback_path = os.path.join(base_dir, "orchid_v4_best")
                if os.path.exists(fallback_path + '.zip'):
                    print(f"Specialized model ({model_name}.zip) not found for {regime}. Falling back to default best model.")
                    _models[regime] = PPO.load(fallback_path)
                else:
                    _models[regime] = None
        except Exception as e:
            print(f"Model for {regime} not loaded: {e}")
            _models[regime] = None
            
    return _models[regime]

def generate_maneuver(satellite, conjunctions):
    sat_state = tle_to_state(satellite.tle1, satellite.tle2)
    regime = get_orbital_regime(satellite.tle2)
    model = load_model_for_regime(regime)

    # Default fallback thrust RTN delta-V (m/s) if no model loaded
    dv = np.array([0.0018, 0.0005, 0.0012])

    if model is not None:
        try:
            obs = np.zeros(8 + 10*9 + 2, dtype=np.float32)
            obs[:3] = sat_state[:3] / 7000
            obs[3:6] = sat_state[3:] / 8
            obs[6] = 1.0
            obs[7] = 1.0
            for i, conj in enumerate(conjunctions[:10]):
                obs[8 + i*9] = conj["distance_km"] / 5.0
            action, _ = model.predict(obs, deterministic=True)
            dv = action * 0.015
        except Exception as e:
            print(f"Failed to predict using RL model: {e}. Using default delta-v.")

    # Determine time to closest approach (TCA) of the primary conjunction
    tca_min = 30.0
    if conjunctions and len(conjunctions) > 0:
        tca_min = conjunctions[0].get("time_to_closest_approach_min", 30.0)

    # Optimization: Calculate the best time to execute the burn to maximize separation and minimize fuel
    # We want to burn as early as possible before TCA.
    # For LEO, 1-2 orbits before TCA (90 to 180 mins) is highly optimal.
    if tca_min > 180.0:
        # Burn 120 minutes before TCA (leaving 120 minutes of drift time)
        burn_lead_time_min = 120.0
    elif tca_min > 60.0:
        # Burn 45 minutes before TCA
        burn_lead_time_min = 45.0
    else:
        # Critical encounter: burn immediately (lead time is the entire remaining time to TCA)
        burn_lead_time_min = max(5.0, tca_min)

    # Calculate time of burn relative to now (minutes from now)
    minutes_until_burn = max(0.0, tca_min - burn_lead_time_min)
    
    # Scale delta-V based on drift time. 
    # The separation achieved is proportional to delta_v * drift_time.
    # Standard nominal delta_v is calibrated for a 30-minute drift time.
    # Therefore: dv_required = dv_nominal * (30.0 / drift_time)
    drift_time_ratio = 30.0 / burn_lead_time_min
    # Clamp scale factor to keep delta-V physically realistic [0.1x to 2.5x]
    dv_scale_factor = min(2.5, max(0.1, drift_time_ratio))
    
    dv = dv * dv_scale_factor

    DRY_MASS = 500.0
    INIT_FUEL = 50.0
    fuel_cost = tsiolkovsky(np.linalg.norm(dv), DRY_MASS + INIT_FUEL)
    from datetime import datetime, timezone, timedelta
    burn_time = datetime.now(timezone.utc) + timedelta(minutes=minutes_until_burn)
    
    return {
        "delta_v_rtn": dv.tolist(),
        "burn_time_utc": burn_time.isoformat(),
        "fuel_cost_kg": round(float(fuel_cost), 4),
        "post_maneuver_safety_km": round(float(conjunctions[0]["distance_km"] * (1.5 / dv_scale_factor)), 3) if conjunctions else 0.0,
        "policy_regime": regime,
        "burn_lead_time_min": round(float(burn_lead_time_min), 1)
    }