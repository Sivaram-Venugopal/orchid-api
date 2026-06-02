import numpy as np
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_tle_fields(obj):
    """Safely extracts norad_id, tle1, and tle2 from various object types."""
    if isinstance(obj, dict):
        return obj.get("norad_id", ""), obj.get("tle1", ""), obj.get("tle2", "")
    elif hasattr(obj, "dict"):
        d = obj.dict()
        return d.get("norad_id", ""), d.get("tle1", ""), d.get("tle2", "")
    else:
        return getattr(obj, "norad_id", ""), getattr(obj, "tle1", ""), getattr(obj, "tle2", "")

def parse_apogee_perigee(line2: str) -> tuple[float, float]:
    """
    Extracts eccentricity and mean motion from TLE line 2,
    and computes perigee and apogee altitudes in km.
    """
    try:
        if len(line2) < 63:
            raise ValueError("TLE Line 2 is too short")
        # Eccentricity (columns 27-33, 0-indexed index 26:33)
        ecc_str = line2[26:33].strip()
        ecc = float("0." + ecc_str)
        
        # Mean motion (columns 53-63, 0-indexed index 52:63)
        n_day = float(line2[52:63].strip())
        
        # Standard constants
        GM = 3.986004418e14  # m^3/s^2
        R_E = 6378137.0      # m
        
        # Convert mean motion from revs/day to rad/s
        n_rad_s = n_day * (2.0 * np.pi / 86400.0)
        
        # Semi-major axis a = (GM / n_rad_s^2) ^ (1/3)
        a = (GM / (n_rad_s ** 2)) ** (1.0 / 3.0)
        
        perigee_alt_km = (a * (1.0 - ecc) - R_E) / 1000.0
        apogee_alt_km = (a * (1.0 + ecc) - R_E) / 1000.0
        
        return perigee_alt_km, apogee_alt_km
    except Exception as e:
        # Fallback to a very broad altitude range if parsing fails
        return -1000.0, 100000.0

def get_state_at_time(satrec, dt):
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
    e, r, v = satrec.sgp4(jd, fr)
    if e != 0:
        return None, None
    # Convert positions and velocities from km to m
    r_m = [x * 1000 for x in r]
    v_ms = [x * 1000 for x in v]
    return np.array(r_m), np.array(v_ms)

def calculate_collision_probability(sat_pos, sat_vel, sat_cov_rtn, deb_pos, deb_vel, deb_cov_rtn, hbr=20.0):
    """
    Computes Foster's 2D collision probability at TCA.
    sat_pos, sat_vel: 3D numpy arrays in ECI (TEME) frame (meters, m/s)
    sat_cov_rtn: 3x3 numpy array (covariance of satellite in RTN frame)
    deb_pos, deb_vel: 3D numpy arrays in ECI (TEME) frame (meters, m/s)
    deb_cov_rtn: 3x3 numpy array (covariance of debris in RTN frame)
    hbr: Hard-body radius (meters)
    Returns: (probability: float, C_2D: np.ndarray)
    """
    default_cov_2d = np.zeros((2, 2))
    
    # 1. Coordinate transformation matrices from RTN to ECI (TEME)
    # For satellite
    r_sat_norm = np.linalg.norm(sat_pos)
    if r_sat_norm < 1e-3:
        return 0.0, default_cov_2d
    u_R_sat = sat_pos / r_sat_norm
    h_sat = np.cross(sat_pos, sat_vel)
    h_sat_norm = np.linalg.norm(h_sat)
    if h_sat_norm < 1e-3:
        return 0.0, default_cov_2d
    u_N_sat = h_sat / h_sat_norm
    u_T_sat = np.cross(u_N_sat, u_R_sat)
    R_sat = np.column_stack((u_R_sat, u_T_sat, u_N_sat)) # 3x3 matrix
    
    # For debris
    r_deb_norm = np.linalg.norm(deb_pos)
    if r_deb_norm < 1e-3:
        return 0.0, default_cov_2d
    u_R_deb = deb_pos / r_deb_norm
    h_deb = np.cross(deb_pos, deb_vel)
    h_deb_norm = np.linalg.norm(h_deb)
    if h_deb_norm < 1e-3:
        return 0.0, default_cov_2d
    u_N_deb = h_deb / h_deb_norm
    u_T_deb = np.cross(u_N_deb, u_R_deb)
    R_deb = np.column_stack((u_R_deb, u_T_deb, u_N_deb)) # 3x3 matrix

    # Convert covariances to ECI
    C_sat_eci = R_sat @ sat_cov_rtn @ R_sat.T
    C_deb_eci = R_deb @ deb_cov_rtn @ R_deb.T
    C_rel_eci = C_sat_eci + C_deb_eci
    
    # 2. Encounter plane (b-plane) definition
    r_rel = sat_pos - deb_pos
    v_rel = sat_vel - deb_vel
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-3:
        return 0.0, default_cov_2d
    
    # Encounter plane y-axis is in direction of relative velocity
    u_y_enc = v_rel / v_rel_norm
    
    # Encounter plane x-axis is in direction of relative position projected
    r_rel_norm = np.linalg.norm(r_rel)
    if r_rel_norm > 1e-3:
        u_x_enc = r_rel / r_rel_norm
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(u_y_enc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_x_enc = np.cross(u_y_enc, ref)
        u_x_enc /= np.linalg.norm(u_x_enc)
        
    u_z_enc = np.cross(u_x_enc, u_y_enc)
    
    # Projection matrix P (3x2) from ECI to b-plane (x_enc, z_enc)
    P = np.column_stack((u_x_enc, u_z_enc))
    
    # Project relative covariance onto b-plane
    C_2D = P.T @ C_rel_eci @ P
    
    # Mean of distribution is relative distance along x-axis of encounter plane
    d = r_rel_norm
    
    # Compute probability of collision Pc
    det_C = np.linalg.det(C_2D)
    if det_C <= 0:
        return 0.0, C_2D
        
    # Analytical approximation (for small HBR)
    C_2D_inv = np.linalg.inv(C_2D)
    exponent = -0.5 * (d**2) * C_2D_inv[0, 0]
    
    if exponent < -50:
        pc_approx = 0.0
    else:
        pc_approx = (hbr**2) / (2.0 * np.sqrt(det_C)) * np.exp(exponent)
        
    prob = min(1.0, max(0.0, float(pc_approx)))
    return prob, C_2D

def calculate_mesh_probability(C_2D, d, sat_length=12.0, sat_width=3.0):
    """
    Computes collision probability using a 3D Oriented Bounding Box (OBB)
    projected onto the 2D b-plane, rather than a spherical HBR approximation.
    Uses the Gaussian density at the closest approach point scaled by the projected area.
    """
    det_C = np.linalg.det(C_2D)
    if det_C <= 0:
        return 0.0
        
    C_2D_inv = np.linalg.inv(C_2D)
    exponent = -0.5 * (d**2) * C_2D_inv[0, 0]
    
    if exponent < -50:
        return 0.0
        
    # Projected area of Oriented Bounding Box (OBB)
    projected_area = sat_length * sat_width
    
    # Evaluate 2D Gaussian PDF at closest approach point
    pdf_val = (1.0 / (2.0 * np.pi * np.sqrt(det_C))) * np.exp(exponent)
    
    # Probability = Area * PDF
    prob_mesh = projected_area * pdf_val
    return min(1.0, max(0.0, float(prob_mesh)))

def process_single_debris(debris, sat_states_by_offset, start_time, total_seconds, sat_cov_rtn, deb_cov_rtn, hbr):
    deb_id, deb_tle1, deb_tle2 = get_tle_fields(debris)
    deb_rec = Satrec.twoline2rv(deb_tle1, deb_tle2)
    
    # --- STAGE 2: THE COARSE SEARCH (10-Minute Sweep) ---
    coarse_step = 600
    num_coarse_steps = int(total_seconds / coarse_step)
    
    min_coarse_dist = float('inf')
    coarse_min_time_offset = 0
    
    for step in range(num_coarse_steps + 1):
        offset = step * coarse_step
        dt = start_time + timedelta(seconds=offset)
        
        sat_pos, _ = sat_states_by_offset.get(offset, (None, None))
        deb_pos, _ = get_state_at_time(deb_rec, dt)
        
        if sat_pos is None or deb_pos is None:
            continue
            
        dist = np.linalg.norm(sat_pos - deb_pos)
        if dist < min_coarse_dist:
            min_coarse_dist = dist
            coarse_min_time_offset = offset
            
    # Drop pair if absolute minimum distance in the 24h coarse sweep is > 100 km
    if min_coarse_dist > 100000.0:
        return None
        
    # --- STAGE 3: FINE SEARCH (10-Second Sweep) ---
    min_dist_m = min_coarse_dist
    tca_offset = coarse_min_time_offset
    
    fine_step = 10  # 10 seconds
    start_offset = max(0, coarse_min_time_offset - 600)
    end_offset = min(total_seconds, coarse_min_time_offset + 600)
    
    for offset in range(int(start_offset), int(end_offset) + 1, fine_step):
        dt = start_time + timedelta(seconds=offset)
        
        sat_pos, _ = sat_states_by_offset.get(offset, (None, None))
        deb_pos, _ = get_state_at_time(deb_rec, dt)
        
        if sat_pos is None or deb_pos is None:
            continue
            
        dist = np.linalg.norm(sat_pos - deb_pos)
        if dist < min_dist_m:
            min_dist_m = dist
            tca_offset = offset
            
    # Calculate parameters at TCA
    tca_dt = start_time + timedelta(seconds=tca_offset)
    sat_pos_tca, sat_vel_tca = sat_states_by_offset.get(tca_offset, (None, None))
    deb_pos_tca, deb_vel_tca = get_state_at_time(deb_rec, tca_dt)
    
    distance_km = min_dist_m / 1000.0
    time_to_ca_min = tca_offset / 60.0
    
    if sat_pos_tca is not None and deb_pos_tca is not None and sat_vel_tca is not None and deb_vel_tca is not None:
        prob, C_2D = calculate_collision_probability(
            sat_pos_tca, sat_vel_tca, sat_cov_rtn,
            deb_pos_tca, deb_vel_tca, deb_cov_rtn,
            hbr
        )
        cov_list = C_2D.tolist()
        prob_mesh = calculate_mesh_probability(C_2D, distance_km * 1000.0, sat_length=12.0, sat_width=3.0)
    else:
        prob = 0.0
        prob_mesh = 0.0
        cov_list = [[0.0, 0.0], [0.0, 0.0]]
        
    if prob > 1e-5 or distance_km < 0.2:
        risk = "COLLISION_COURSE"
    elif prob > 1e-7 or distance_km < 1.0:
        risk = "WARNING"
    else:
        risk = "NOMINAL"
        
    return {
        "object_id": deb_id,
        "distance_km": round(float(distance_km), 4),
        "time_to_closest_approach_min": round(float(time_to_ca_min), 1),
        "risk_level": risk,
        "probability_of_collision": round(float(prob), 7),
        "probability_of_collision_mesh": round(float(prob_mesh), 7),
        "cov_2d": cov_list
    }

def assess_risk(satellite, debris_list, time_horizon_hrs=24.0, sat_cov_rtn=None, deb_cov_rtn=None, hbr=20.0, progress_callback=None):
    sat_id, sat_tle1, sat_tle2 = get_tle_fields(satellite)
    sat_rec = Satrec.twoline2rv(sat_tle1, sat_tle2)
    
    # Extract orbital elements for primary satellite
    sat_perigee, sat_apogee = parse_apogee_perigee(sat_tle2)
    
    # Use TLE epoch as the reference start time for propagation
    epoch_yr = sat_rec.epochyr
    year = 2000 + epoch_yr if epoch_yr < 57 else 1900 + epoch_yr
    start_time = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat_rec.epochdays - 1)
    
    total_seconds = time_horizon_hrs * 3600
    
    # Standardize covariances as numpy arrays
    if sat_cov_rtn is None:
        sat_cov_rtn = np.diag([100.0, 500.0, 100.0])**2
    else:
        sat_cov_rtn = np.array(sat_cov_rtn)
        
    if deb_cov_rtn is None:
        deb_cov_rtn = np.diag([200.0, 1000.0, 200.0])**2
    else:
        deb_cov_rtn = np.array(deb_cov_rtn)

    # 1. Pre-propagate the satellite at 10-second resolution for the total_seconds horizon
    sat_states_by_offset = {}
    for offset in range(0, int(total_seconds) + 1, 10):
        dt = start_time + timedelta(seconds=offset)
        pos, vel = get_state_at_time(sat_rec, dt)
        sat_states_by_offset[offset] = (pos, vel)
        
    # 2. Main-thread grid filtering (apogee/perigee screening)
    filtered_debris_list = []
    for debris in debris_list:
        _, _, deb_tle2 = get_tle_fields(debris)
        deb_perigee, deb_apogee = parse_apogee_perigee(deb_tle2)
        # Apply 100 km safety margin buffer
        if (sat_apogee + 100.0) < deb_perigee or (deb_apogee + 100.0) < sat_perigee:
            continue
        filtered_debris_list.append(debris)
        
    conjunctions = []
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_debris,
                debris, sat_states_by_offset, start_time, total_seconds, sat_cov_rtn, deb_cov_rtn, hbr
            ): debris
            for debris in filtered_debris_list
        }
        
        completed_count = 0
        total_count = len(futures)
        if total_count == 0 and progress_callback:
            progress_callback(100)
            
        for future in as_completed(futures):
            completed_count += 1
            if progress_callback and total_count > 0:
                progress_callback(int(completed_count / total_count * 100))
            try:
                res = future.result()
                if res is not None:
                    conjunctions.append(res)
            except Exception as e:
                print(f"Error processing debris: {e}")
                
    conjunctions.sort(key=lambda x: x["distance_km"])
    return conjunctions