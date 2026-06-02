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
    Returns: (probability: float, C_2D: np.ndarray, R_sat: np.ndarray, P: np.ndarray)
    """
    default_cov_2d = np.zeros((2, 2))
    dummy_R = np.eye(3)
    dummy_P = np.zeros((3, 2))
    
    # 1. Coordinate transformation matrices from RTN to ECI (TEME)
    # For satellite
    r_sat_norm = np.linalg.norm(sat_pos)
    if r_sat_norm < 1e-3:
        return 0.0, default_cov_2d, dummy_R, dummy_P
    u_R_sat = sat_pos / r_sat_norm
    h_sat = np.cross(sat_pos, sat_vel)
    h_sat_norm = np.linalg.norm(h_sat)
    if h_sat_norm < 1e-3:
        return 0.0, default_cov_2d, dummy_R, dummy_P
    u_N_sat = h_sat / h_sat_norm
    u_T_sat = np.cross(u_N_sat, u_R_sat)
    R_sat = np.column_stack((u_R_sat, u_T_sat, u_N_sat)) # 3x3 matrix
    
    # For debris
    r_deb_norm = np.linalg.norm(deb_pos)
    if r_deb_norm < 1e-3:
        return 0.0, default_cov_2d, dummy_R, dummy_P
    u_R_deb = deb_pos / r_deb_norm
    h_deb = np.cross(deb_pos, deb_vel)
    h_deb_norm = np.linalg.norm(h_deb)
    if h_deb_norm < 1e-3:
        return 0.0, default_cov_2d, dummy_R, dummy_P
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
        return 0.0, default_cov_2d, dummy_R, dummy_P
    
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
        return 0.0, C_2D, R_sat, P
        
    # Analytical approximation (for small HBR)
    C_2D_inv = np.linalg.inv(C_2D)
    exponent = -0.5 * (d**2) * C_2D_inv[0, 0]
    
    if exponent < -50:
        pc_approx = 0.0
    else:
        pc_approx = (hbr**2) / (2.0 * np.sqrt(det_C)) * np.exp(exponent)
        
    prob = min(1.0, max(0.0, float(pc_approx)))
    return prob, C_2D, R_sat, P

def calculate_mesh_probability(C_2D, d, R_sat, P):
    """
    Computes collision probability using a 3D Oriented Bounding Box (OBB) & Solar Arrays
    projected onto the 2D b-plane, rather than a spherical HBR approximation.
    Integrates the 2D Gaussian density function over the exact projected structural geometry.
    """
    from scipy.spatial import ConvexHull
    
    det_C = np.linalg.det(C_2D)
    if det_C <= 0:
        return 0.0
        
    C_2D_inv = np.linalg.inv(C_2D)
    
    # Define satellite body geometry boxes (Bus + 2 Solar Arrays) in Local Body Frame
    bus_min = np.array([-1.5, -1.0, -1.0])
    bus_max = np.array([1.5, 1.0, 1.0])
    p1_min = np.array([-0.1, -1.0, 1.0])
    p1_max = np.array([0.1, 1.0, 6.0])
    p2_min = np.array([-0.1, -1.0, -6.0])
    p2_max = np.array([0.1, 1.0, -1.0])
    
    def get_projected_box_hull(box_min, box_max, R_sat_mat, P_mat):
        vertices = []
        for x in [box_min[0], box_max[0]]:
            for y in [box_min[1], box_max[1]]:
                for z in [box_min[2], box_max[2]]:
                    vertices.append([x, y, z])
        vertices_body = np.array(vertices).T # 3x8
        vertices_eci = R_sat_mat @ vertices_body # 3x8
        vertices_b = P_mat.T @ vertices_eci # 2x8
        points_2d = vertices_b.T # 8x2
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except Exception:
            return points_2d

    hulls = [
        get_projected_box_hull(bus_min, bus_max, R_sat, P),
        get_projected_box_hull(p1_min, p1_max, R_sat, P),
        get_projected_box_hull(p2_min, p2_max, R_sat, P)
    ]
    
    # Find bounding box
    all_points = np.vstack(hulls)
    x_min, z_min = np.min(all_points, axis=0)
    x_max, z_max = np.max(all_points, axis=0)
    
    # Perform Monte Carlo Integration over the bounding box
    num_samples = 1500
    np.random.seed(42) # deterministic seed for consistency
    xs = np.random.uniform(x_min, x_max, num_samples)
    zs = np.random.uniform(z_min, z_max, num_samples)
    samples = np.column_stack((xs, zs))
    
    # Check which samples are inside the union of hulls
    inside_union = np.zeros(num_samples, dtype=bool)
    for hull in hulls:
        M = len(hull)
        if M < 3:
            continue
        v = np.vstack([hull, hull[0]])
        first_edge_signs = (v[1, 0] - v[0, 0]) * (samples[:, 1] - v[0, 1]) - (v[1, 1] - v[0, 1]) * (samples[:, 0] - v[0, 0])
        is_positive = np.mean(first_edge_signs >= 0) > 0.5
        
        inside_hull = np.ones(num_samples, dtype=bool)
        for i in range(M):
            dx = v[i+1, 0] - v[i, 0]
            dz = v[i+1, 1] - v[i, 1]
            cross = dx * (samples[:, 1] - v[i, 1]) - dz * (samples[:, 0] - v[i, 0])
            if is_positive:
                inside_hull &= (cross >= -1e-9)
            else:
                inside_hull &= (cross <= 1e-9)
        inside_union |= inside_hull
        
    inside_samples = samples[inside_union]
    if len(inside_samples) == 0:
        return 0.0
        
    dx_vals = inside_samples[:, 0] - d
    dz_vals = inside_samples[:, 1]
    
    quad_form = (
        dx_vals * (C_2D_inv[0, 0] * dx_vals + C_2D_inv[0, 1] * dz_vals) +
        dz_vals * (C_2D_inv[1, 0] * dx_vals + C_2D_inv[1, 1] * dz_vals)
    )
    
    exponent = -0.5 * quad_form
    pdf_vals = (1.0 / (2.0 * np.pi * np.sqrt(det_C))) * np.exp(exponent)
    
    box_area = (x_max - x_min) * (z_max - z_min)
    prob_mesh = box_area * np.sum(pdf_vals) / num_samples
    
    return min(1.0, max(0.0, float(prob_mesh)))

def process_single_debris(debris, sat_states_by_offset, start_time, total_seconds, sat_cov_rtn, deb_cov_rtn, hbr):
    from physics import propagate_rk4_trajectory
    import math
    from sgp4.api import jday
    
    deb_id, deb_tle1, deb_tle2 = get_tle_fields(debris)
    deb_rec = Satrec.twoline2rv(deb_tle1, deb_tle2)
    
    # Pre-calculate Julian Date of start_time to avoid slow datetime arithmetic in loop
    jd_epoch, fr_epoch = jday(start_time.year, start_time.month, start_time.day,
                              start_time.hour, start_time.minute, start_time.second + start_time.microsecond/1e6)
    
    # --- STAGE 2: THE COARSE SEARCH (10-Minute Sweep using fast SGP4) ---
    coarse_step = 600
    num_coarse_steps = int(total_seconds / coarse_step)
    
    min_coarse_dist = float('inf')
    coarse_min_time_offset = 0
    
    for step in range(num_coarse_steps + 1):
        offset = step * coarse_step
        
        sat_pos, _ = sat_states_by_offset.get(offset, (None, None))
        if sat_pos is None:
            continue
            
        e, r, v = deb_rec.sgp4(jd_epoch, fr_epoch + offset / 86400.0)
        if e != 0:
            continue
            
        dx = sat_pos[0] - r[0] * 1000.0
        dy = sat_pos[1] - r[1] * 1000.0
        dz = sat_pos[2] - r[2] * 1000.0
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < min_coarse_dist:
            min_coarse_dist = dist
            coarse_min_time_offset = offset
            
    # Drop pair if absolute minimum distance in the 24h coarse sweep is > 100 km
    if min_coarse_dist > 100000.0:
        return None
        
    # --- STAGE 3: FINE SEARCH (10-Second Sweep using High-Fidelity perturbed RK4 propagation) ---
    min_dist_m = float('inf')
    tca_offset = coarse_min_time_offset
    
    fine_step = 10  # 10 seconds
    start_offset = max(0, coarse_min_time_offset - 600)
    end_offset = min(total_seconds, coarse_min_time_offset + 600)
    
    deb_pos_tca = None
    deb_vel_tca = None
    
    # Initialize numerical integration from the SGP4 state at start_offset
    e, r, v = deb_rec.sgp4(jd_epoch, fr_epoch + start_offset / 86400.0)
    if e == 0:
        deb_pos_start = np.array([r[0] * 1000.0, r[1] * 1000.0, r[2] * 1000.0])
        deb_vel_start = np.array([v[0] * 1000.0, v[1] * 1000.0, v[2] * 1000.0])
    else:
        deb_pos_start, deb_vel_start = None, None
        
    if deb_pos_start is not None:
        deb_total_sec = end_offset - start_offset
        deb_traj = propagate_rk4_trajectory(
            deb_pos_start.tolist(), deb_vel_start.tolist(), deb_total_sec,
            mass=250.0, area=4.0, drag_coeff=2.2, step_size_sec=10.0
        )
        
        deb_pos_tca = deb_pos_start.copy()
        deb_vel_tca = deb_vel_start.copy()
        
        for offset in range(int(start_offset), int(end_offset) + 1, fine_step):
            deb_offset = offset - start_offset
            deb_state = deb_traj.get(deb_offset)
            if deb_state is None:
                continue
            deb_pos_current, deb_vel_current = deb_state
                
            sat_pos, _ = sat_states_by_offset.get(offset, (None, None))
            if sat_pos is None:
                continue
                
            dist = np.linalg.norm(sat_pos - deb_pos_current)
            if dist < min_dist_m:
                min_dist_m = dist
                tca_offset = offset
                deb_pos_tca = deb_pos_current.copy()
                deb_vel_tca = deb_vel_current.copy()
                
    sat_pos_tca, sat_vel_tca = sat_states_by_offset.get(tca_offset, (None, None))
    distance_km = min_dist_m / 1000.0
    time_to_ca_min = tca_offset / 60.0
    
    if sat_pos_tca is not None and deb_pos_tca is not None and sat_vel_tca is not None and deb_vel_tca is not None:
        prob, C_2D, R_sat, P = calculate_collision_probability(
            sat_pos_tca, sat_vel_tca, sat_cov_rtn,
            deb_pos_tca, deb_vel_tca, deb_cov_rtn,
            hbr
        )
        cov_list = C_2D.tolist()
        prob_mesh = calculate_mesh_probability(C_2D, distance_km * 1000.0, R_sat, P)
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

    # 1. Pre-propagate the satellite at 10-second resolution for the total_seconds horizon (RK4 perturbed integration)
    from physics import propagate_rk4_trajectory
    pos_epoch, vel_epoch = get_state_at_time(sat_rec, start_time)
    if pos_epoch is not None:
        sat_states_by_offset = propagate_rk4_trajectory(
            pos_epoch.tolist(), vel_epoch.tolist(), total_seconds,
            mass=550.0, area=12.0, drag_coeff=2.2, step_size_sec=10.0
        )
    else:
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