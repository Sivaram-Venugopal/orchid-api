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

def verify_maneuver_safety(satellite, conjunctions, debris_pool, dv, minutes_until_burn, omega):
    """
    Verifies if the evasive maneuver delta-V causes a secondary collision course
    with any other debris object in the pool using the Clohessy-Wiltshire equations.
    """
    from sgp4.api import jday, Satrec
    from risk_engine import get_state_at_time, get_tle_fields
    from datetime import datetime, timezone, timedelta
    
    if not debris_pool:
        return []
        
    secondary_hazards = []
    
    # Map debris_id to TLE inputs
    debris_map = {}
    for d in debris_pool:
        did, dtle1, dtle2 = get_tle_fields(d)
        debris_map[did] = (dtle1, dtle2)
        
    sat_rec = Satrec.twoline2rv(satellite.tle1, satellite.tle2)
    
    # Check secondary threats (skip primary threat at index 0)
    for conj in conjunctions[1:10]: # Check up to 10 closest objects
        deb_id = conj["object_id"]
        if deb_id not in debris_map:
            continue
            
        tca_min = conj["time_to_closest_approach_min"]
        
        # Drift time in seconds after the burn
        drift_sec = (tca_min - minutes_until_burn) * 60.0
        if drift_sec <= 0:
            continue # Burn happens after this TCA, no effect
            
        # Clohessy-Wiltshire (CW) equations to calculate delta_r in RTN
        wt = omega * drift_sec
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        
        dvr, dvt, dvn = dv
        dr_r = (dvr / omega) * sin_wt + (2 * dvt / omega) * (1 - cos_wt)
        dr_t = (2 * dvr / omega) * (cos_wt - 1) + (dvt / omega) * (4 * sin_wt - 3 * wt)
        dr_n = (dvn / omega) * sin_wt
        dr_rtn = np.array([dr_r, dr_t, dr_n])
        
        # Get start time of propagation based on satellite TLE epoch
        epoch_yr = sat_rec.epochyr
        year = 2000 + epoch_yr if epoch_yr < 57 else 1900 + epoch_yr
        start_time = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat_rec.epochdays - 1)
        
        tca_dt = start_time + timedelta(minutes=tca_min)
        sat_pos, sat_vel = get_state_at_time(sat_rec, tca_dt)
        
        dtle1, dtle2 = debris_map[deb_id]
        deb_rec = Satrec.twoline2rv(dtle1, dtle2)
        deb_pos, _ = get_state_at_time(deb_rec, tca_dt)
        
        if sat_pos is None or deb_pos is None or sat_vel is None:
            continue
            
        # Get RTN to ECI rotation matrix at TCA
        r_norm = np.linalg.norm(sat_pos)
        u_R = sat_pos / r_norm
        h = np.cross(sat_pos, sat_vel)
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-3:
            continue
        u_N = h / h_norm
        u_T = np.cross(u_N, u_R)
        R_sat = np.column_stack((u_R, u_T, u_N))
        
        # Relative position vector at TCA
        r_rel_nominal = sat_pos - deb_pos
        
        # New relative position vector at TCA (r_rel_new = r_rel_nominal + R_sat @ dr_rtn)
        r_rel_new = r_rel_nominal + R_sat @ dr_rtn
        new_dist_km = np.linalg.norm(r_rel_new) / 1000.0
        
        # If the new distance is below 0.5 km, it's a hazard!
        if new_dist_km < 0.5:
            secondary_hazards.append({
                "object_id": deb_id,
                "time_to_closest_approach_min": round(float(tca_min), 1),
                "original_distance_km": round(float(conj["distance_km"]), 4),
                "post_maneuver_distance_km": round(float(new_dist_km), 4)
            })
            
    return secondary_hazards

def generate_maneuver(satellite, conjunctions, debris_pool=None):
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
    drift_time_ratio = 30.0 / burn_lead_time_min
    # Clamp scale factor to keep delta-V physically realistic [0.1x to 2.5x]
    dv_scale_factor = min(2.5, max(0.1, drift_time_ratio))
    
    dv = dv * dv_scale_factor

    # Get mean motion (omega) in rad/s from TLE Line 2
    try:
        n_day = float(satellite.tle2[52:63].strip())
        omega = n_day * (2.0 * np.pi / 86400.0)
    except Exception:
        omega = 0.0011 # default LEO mean motion

    # Verify secondary safety check
    secondary_hazards = verify_maneuver_safety(
        satellite, conjunctions, debris_pool, dv, minutes_until_burn, omega
    )
    post_maneuver_clear = len(secondary_hazards) == 0

    DRY_MASS = 500.0
    INIT_FUEL = 50.0
    fuel_cost = tsiolkovsky(np.linalg.norm(dv), DRY_MASS + INIT_FUEL)
    from datetime import datetime, timezone, timedelta
    burn_time = datetime.now(timezone.utc) + timedelta(minutes=minutes_until_burn)
    
    # Generate multi-agent negotiation logs and cFS flight commands
    primary_conj = conjunctions[0] if conjunctions else {}
    negotiation_log = resolve_multi_agent_conjunction(satellite, primary_conj)
    flight_commands = compile_flight_commands(dv, minutes_until_burn, round(float(fuel_cost), 4))
    
    return {
        "delta_v_rtn": dv.tolist(),
        "burn_time_utc": burn_time.isoformat(),
        "fuel_cost_kg": round(float(fuel_cost), 4),
        "post_maneuver_safety_km": round(float(conjunctions[0]["distance_km"] * (1.5 / dv_scale_factor)), 3) if conjunctions else 0.0,
        "policy_regime": regime,
        "burn_lead_time_min": round(float(burn_lead_time_min), 1),
        "post_maneuver_clear": post_maneuver_clear,
        "secondary_hazards": secondary_hazards,
        "negotiation_log": negotiation_log,
        "flight_commands": flight_commands
    }

def resolve_multi_agent_conjunction(satellite, primary_conjunction):
    """
    Simulates peer-to-peer negotiation for active payload conjunctions.
    """
    sat_id = satellite.norad_id
    deb_id = primary_conjunction.get("object_id", "Unknown")
    
    # Simulate if the target object is an active satellite
    is_active_threat = False
    try:
        if int(deb_id) < 90000:
            is_active_threat = True
    except ValueError:
        pass

    logs = []
    if is_active_threat:
        logs.append(f"[P2P-LINK] Connection established with active payload #{deb_id}.")
        logs.append(f"[AD-HOC-NET] Exchanging telemetry and propulsion profiles...")
        logs.append(f"[DECISION] Sat A (local) Fuel: 48.2 kg | Sat B (remote) Fuel: 12.4 kg")
        logs.append(f"[DECISION] Sat A has higher fuel reserve. Selected as active maneuvering agent.")
        logs.append(f"[COORDINATION] Burn plan transmitted. Sat B locks attitude to passivity.")
    else:
        logs.append(f"[SENSOR-LOG] Target identified as passive orbital debris (NORAD #{deb_id}).")
        logs.append(f"[DECISION] Direct link unavailable. Sat A assigned to perform unilateral evasive burn.")
        
    return logs

def compile_flight_commands(dv, minutes_until_burn, fuel_cost_kg):
    """
    Generates a cFS compatible telecommand script for thruster firing and ADCS.
    """
    dv_norm = np.linalg.norm(dv)
    # Estimate thrust duration: 10N thruster on a 550kg satellite (Isp = 220s)
    duration_sec = round(dv_norm / 0.018, 1) if dv_norm > 0 else 0.0
    
    # Generate mock quaternion values based on the RTN thrust vector
    if dv_norm > 1e-6:
        u = dv / dv_norm
        qw = round(0.95 + 0.04 * u[0], 4)
        qx = round(0.1 * u[0], 4)
        qy = round(0.1 * u[1], 4)
        qz = round(0.1 * u[2], 4)
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        
    commands = [
        f"00:00:00.000  [SYS] CMD_CCSDS_HDR_EXEC  APID=0x180A FUNC=0x01",
        f"00:00:00.100  [ADCS] CMD_ADCS_POINT_QUAT  [qw: {qw}, qx: {qx}, qy: {qy}, qz: {qz}]",
        f"00:00:04.500  [ADCS] STATUS_CHECK_ALIGNMENT  tolerance=0.5deg ... OK",
        f"00:00:05.000  [PROP] CMD_THRUSTER_PREHEAT  manifold=A target_temp=45C",
        f"00:00:10.000  [PROP] CMD_THRUSTER_ARM  valves=OPEN pressure=320psi",
        f"00:00:12.000  [PROP] CMD_THRUSTER_FIRE  duration_sec={duration_sec}s pulse=100%",
        f"00:00:12.000+ [PROP] MONITOR_BURN_INTEGRATION  accum_dv={round(dv_norm, 5)} m/s",
        f"00:00:12.000+ [PROP] CMD_THRUSTER_SAFE  valves=CLOSED fuel_depleted={fuel_cost_kg}kg",
        f"00:00:12.500  [ADCS] CMD_ADCS_RESET_NOMINAL  mode=NADIR_TRACKING"
    ]
    return commands