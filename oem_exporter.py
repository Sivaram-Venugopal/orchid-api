import numpy as np
from datetime import datetime, timezone, timedelta
from physics import tle_to_state, propagate_rk4_trajectory

def generate_oem_text(satellite_name: str, norad_id: str, tle1: str, tle2: str, 
                      dv_rtn: list, minutes_until_burn: float, time_horizon_hrs: float = 24.0) -> str:
    """
    Generates a CCSDS Orbit Ephemeris Message (OEM) in Key-Value Notation (KVN) format.
    Includes the pre-burn trajectory, the impulsive burn event, and the post-burn avoidance flight path.
    """
    # 1. Get initial state in ECI (TEME) frame (meters, meters/second)
    init_state = tle_to_state(tle1, tle2)
    r_init = np.array(init_state[:3])
    v_init = np.array(init_state[3:])
    
    start_time = datetime.now(timezone.utc)
    total_seconds = time_horizon_hrs * 3600.0
    burn_sec = minutes_until_burn * 60.0
    
    # 2. Propagate trajectory up to the burn event
    if burn_sec > 0:
        traj_pre = propagate_rk4_trajectory(r_init.tolist(), v_init.tolist(), burn_sec, step_size_sec=10.0)
        # Get state at burn event
        r_burn, v_burn = traj_pre[int(burn_sec)]
        r_burn = np.array(r_burn)
        v_burn = np.array(v_burn)
    else:
        traj_pre = {}
        r_burn = r_init
        v_burn = v_init
        
    # 3. Apply the evasive burn (rotate RTN delta-V vector to ECI frame)
    r_norm = np.linalg.norm(r_burn)
    u_R = r_burn / r_norm
    h = np.cross(r_burn, v_burn)
    h_norm = np.linalg.norm(h)
    
    if h_norm > 1e-3:
        u_N = h / h_norm
        u_T = np.cross(u_N, u_R)
        R_sat = np.column_stack((u_R, u_T, u_N)) # Rotation matrix RTN to ECI
    else:
        R_sat = np.eye(3)
        
    # dv_rtn is in m/s, apply it to the ECI velocity
    dv_eci = R_sat @ np.array(dv_rtn)
    v_burn_post = v_burn + dv_eci
    
    # 4. Propagate the post-burn safe trajectory
    remaining_sec = max(0.0, total_seconds - burn_sec)
    if remaining_sec > 0:
        traj_post = propagate_rk4_trajectory(r_burn.tolist(), v_burn_post.tolist(), remaining_sec, step_size_sec=10.0)
    else:
        traj_post = {}
        
    # 5. Assemble and sample points at 60-second intervals for the OEM output
    oem_points = []
    
    # Pre-burn points
    for t in range(0, int(burn_sec) + 1, 60):
        state = traj_pre.get(t)
        if state:
            oem_points.append((t, state[0], state[1]))
            
    # Post-burn points
    for t_rel in range(60, int(remaining_sec) + 1, 60):
        state = traj_post.get(t_rel)
        if state:
            t_abs = int(burn_sec) + t_rel
            oem_points.append((t_abs, state[0], state[1]))
            
    # 6. Format to CCSDS OEM KVN String
    stop_time = start_time + timedelta(seconds=total_seconds)
    
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    stop_time_str = stop_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {creation_date}",
        "ORIGINATOR = ORCHID",
        "",
        "META_START",
        f"OBJECT_NAME = {satellite_name}",
        f"OBJECT_ID = {norad_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = TEME",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {start_time_str}",
        f"STOP_TIME = {stop_time_str}",
        "EPHEMERIS_TYPE = DETERMINISTIC",
        "INTERPOLATION = LAGRANGE",
        "INTERPOLATION_ORDER = 7",
        "META_STOP",
        ""
    ]
    
    # Ephemeris Data block: Epoch, X, Y, Z, X_dot, Y_dot, Z_dot
    # Spacecraft positions must be in kilometers, velocities in km/s
    for t, pos, vel in oem_points:
        epoch = start_time + timedelta(seconds=t)
        epoch_str = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] # milliseconds
        
        # Position in km, velocity in km/s
        px, py, pz = pos[0]/1000.0, pos[1]/1000.0, pos[2]/1000.0
        vx, vy, vz = vel[0]/1000.0, vel[1]/1000.0, vel[2]/1000.0
        
        lines.append(f"{epoch_str} {px:f} {py:f} {pz:f} {vx:f} {vy:f} {vz:f}")
        
    return "\n".join(lines)
