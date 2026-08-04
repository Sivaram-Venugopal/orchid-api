import os
import numpy as np
import logging
import sqlite3
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from ukf import ukf_propagate, ukf_update
from doppler_engine import get_station_eci, calculate_doppler_shift, extract_range_rate_from_doppler
from anomaly_detector import detect_tle_drift

logger = logging.getLogger(__name__)

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conjunction_history.db")

def calculate_checksum(line: str) -> int:
    """Computes the standard TLE checksum for a line."""
    line = line[:68]
    total = 0
    for char in line:
        if char.isdigit():
            total += int(char)
        elif char == '-':
            total += 1
    return total % 10

def cartesian_to_keplerian(r_vec: np.ndarray, v_vec: np.ndarray) -> dict:
    """
    Converts Cartesian state (position r in meters, velocity v in m/s)
    to orbital Keplerian elements for TLE Line 2 representation.
    """
    mu = 3.986004418e14
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    if r < 1e-3:
        return {
            "inclination": 0.0, "raan": 0.0, "eccentricity": 0.0,
            "arg_perigee": 0.0, "mean_anomaly": 0.0, "mean_motion": 0.0
        }
        
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    if h < 1e-3:
        return {
            "inclination": 0.0, "raan": 0.0, "eccentricity": 0.0,
            "arg_perigee": 0.0, "mean_anomaly": 0.0, "mean_motion": 0.0
        }
        
    # 1. Inclination
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    
    # 2. Node vector
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)
    
    # 3. RAAN
    if n > 1e-6:
        raan = np.arctan2(n_vec[1], n_vec[0])
    else:
        raan = 0.0
        
    # 4. Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    ecc = np.linalg.norm(e_vec)
    ecc = min(0.999999, max(0.0, ecc))
    
    # 5. Argument of Perigee
    if n > 1e-6 and ecc > 1e-6:
        arg_p = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1.0, 1.0))
        if e_vec[2] < 0:
            arg_p = 2.0 * np.pi - arg_p
    else:
        arg_p = 0.0
        
    # 6. True anomaly
    if ecc > 1e-6:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2.0 * np.pi - nu
    else:
        if n > 1e-6:
            nu = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0))
            if r_vec[2] < 0:
                nu = 2.0 * np.pi - nu
        else:
            nu = np.arctan2(r_vec[1], r_vec[0])
            
    # 7. Eccentric anomaly
    if ecc < 1.0:
        sin_E = (np.sqrt(1.0 - ecc**2) * np.sin(nu)) / (1.0 + ecc * np.cos(nu))
        cos_E = (ecc + np.cos(nu)) / (1.0 + ecc * np.cos(nu))
        E = np.arctan2(sin_E, cos_E)
        M = E - ecc * np.sin(E)
    else:
        M = 0.0
        
    # 8. Semi-major axis
    a = mu * r / (2.0 * mu - r * v**2)
    if a <= 0:
        a = 6800000.0
        
    # 9. Mean motion
    mean_motion_rad = np.sqrt(mu / a**3)
    mean_motion_rev_day = mean_motion_rad * 86400.0 / (2.0 * np.pi)
    
    return {
        "inclination": float(np.degrees(inc) % 360.0),
        "raan": float(np.degrees(raan) % 360.0),
        "eccentricity": float(ecc),
        "arg_perigee": float(np.degrees(arg_p) % 360.0),
        "mean_anomaly": float(np.degrees(M) % 360.0),
        "mean_motion": float(mean_motion_rev_day)
    }

def format_tle_lines(line1_old: str, line2_old: str, epoch: datetime, elements: dict) -> tuple:
    """
    Constructs new TLE line 1 and line 2 strings with updated epoch
    and Keplerian elements, maintaining other metadata and checksum integrity.
    """
    # 1. Update Epoch on Line 1
    year = epoch.year % 100
    day_of_year = (epoch - datetime(epoch.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 1.0
    epoch_str = f"{year:02d}{day_of_year:12.8f}"
    
    line1_base = line1_old[:18]
    line1_tail = line1_old[32:68]
    line1_new = f"{line1_base}{epoch_str}{line1_tail}"
    line1_checksum = calculate_checksum(line1_new)
    line1_new = f"{line1_new[:68]}{line1_checksum}"
    
    # 2. Update Keplerian Elements on Line 2
    incl_str = f"{elements['inclination']:8.4f}"
    raan_str = f"{elements['raan']:8.4f}"
    ecc_str = f"{int(round(elements['eccentricity'] * 1e7)):07d}"
    argp_str = f"{elements['arg_perigee']:8.4f}"
    ma_str = f"{elements['mean_anomaly']:8.4f}"
    mm_str = f"{elements['mean_motion']:11.8f}"
    
    line2_prefix = line2_old[:8] # "2 25544 "
    rev_num = line2_old[63:68]
    line2_new = f"{line2_prefix}{incl_str} {raan_str} {ecc_str} {argp_str} {ma_str} {mm_str}{rev_num}"
    line2_checksum = calculate_checksum(line2_new)
    line2_new = f"{line2_new[:68]}{line2_checksum}"
    
    return line1_new, line2_new

def process_observation_and_correct_tle(
    norad_id: str,
    observation: dict,
    active_filters: dict
) -> dict:
    """
    Runs the full TLE Correction Pipeline:
    1. Parse observation metadata (timestamp, station, observed frequency).
    2. Retrieve current TLE and filter states.
    3. Update the UKF filter with the range-rate measurement.
    4. Convert the updated state vector to orbital elements and construct corrected TLE.
    5. Detect significant orbital deviations (>5km) via anomaly detector.
    6. Commit corrected TLE and covariance matrices back to the database.
    """
    # Parse observation parameters
    obs_time = datetime.fromisoformat(observation["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc)
    freq_obs = float(observation["frequency_hz"])
    freq_trans = float(observation.get("transmitted_frequency_hz", 437.5e6))
    
    # Station coordinates
    lat = float(observation.get("station_lat", 0.0))
    lng = float(observation.get("station_lng", 0.0))
    alt_m = float(observation.get("station_alt_m", 100.0))
    station_id = str(observation.get("station_id", "ST-1"))
    
    # Get station ECI position & velocity
    r_station, v_station = get_station_eci(lat, lng, alt_m, obs_time)
    sensor_location = np.concatenate([r_station, v_station])
    
    # 1. Fetch current TLE from Database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, tle1, tle2 FROM fleet_satellites WHERE norad_id = ?", (norad_id,))
    row = cursor.fetchone()
    if row:
        sat_name, tle1, tle2 = row
    else:
        # Fallback to cached catalog or defaults
        sat_name = f"SAT-{norad_id}"
        from catalog_manager import load_tle_catalog
        cat = load_tle_catalog()
        if norad_id in cat:
            sat_name = cat[norad_id]["name"]
            tle1 = cat[norad_id]["line1"]
            tle2 = cat[norad_id]["line2"]
        else:
            tle1 = "1 25544U 98067A   26154.50000000  .00010000  00000-0  10000-3 0  9999"
            tle2 = "2 25544  51.6400 120.0000 0005000  90.0000 270.0000 15.50000000000000"
            
    # 2. Get or initialize UKF filter state
    if norad_id in active_filters:
        sat_filter = active_filters[norad_id]
    else:
        from physics import tle_to_state
        state_init = tle_to_state(tle1, tle2)
        # Position uncertainty starts at km-level (e.g., 5000m radial, 10000m transverse, 5000m normal)
        std_r, std_t, std_n = 5000.0, 10000.0, 5000.0
        cov_rtn = np.diag([std_r, std_t, std_n, std_r * 0.01, std_t * 0.01, std_n * 0.01])**2
        sat_filter = {
            "state": np.array(state_init),
            "covariance": cov_rtn,
            "tle1": tle1,
            "tle2": tle2,
            "last_time": obs_time
        }
        active_filters[norad_id] = sat_filter
        
    x_prev = sat_filter["state"]
    P_prev = sat_filter["covariance"]
    last_time = sat_filter["last_time"]
    
    # Calculate propagation time
    dt_sec = (obs_time - last_time).total_seconds()
    if dt_sec < 0:
        # Ignore out of order historical observation updates or set to small step
        dt_sec = 0.0
        
    # 3. Propagate UKF filter
    Q = np.diag([0.5, 0.5, 0.5, 1e-4, 1e-4, 1e-4])**2
    x_pred, P_pred, sigmas_pred = ukf_propagate(x_prev, P_prev, dt_sec, Q)
    
    # 4. Extract range-rate from Doppler frequency
    range_rate_obs = extract_range_rate_from_doppler(freq_trans, freq_obs)
    z = np.array([range_rate_obs])
    
    # Calculate prediction and residual
    r_rel = x_pred[:3] - r_station
    r_norm = np.linalg.norm(r_rel)
    v_rel = x_pred[3:6] - v_station
    range_rate_pred = np.dot(r_rel, v_rel) / r_norm if r_norm > 1e-3 else 0.0
    residual = float(range_rate_obs - range_rate_pred)
    
    # Doppler measurement noise (e.g., 20 Hz frequency error maps to ~13.7 m/s range-rate uncertainty)
    # R represents the variance of this measurement
    freq_err_std = 20.0 # Hz
    c_light = 299792458.0
    rr_err_std = c_light * (freq_err_std / freq_trans)
    R = np.array([[rr_err_std**2]])
    
    # Store initial standard deviations (before update)
    std_pos_before = float(np.sqrt(np.trace(P_pred[:3, :3]) / 3.0))
    
    # 5. Perform UKF measurement update
    x_updated, P_updated = ukf_update(x_pred, P_pred, sigmas_pred, z, R, sensor_type="doppler", sensor_location=sensor_location)
    
    std_pos_after = float(np.sqrt(np.trace(P_updated[:3, :3]) / 3.0))
    logger.info(f"[TLE Corrector] UKF Updated for {norad_id}. Avg Position uncertainty reduced from {std_pos_before:.2f} m to {std_pos_after:.2f} m.")
    
    # Update active filter state
    sat_filter["state"] = x_updated
    sat_filter["covariance"] = P_updated
    sat_filter["last_time"] = obs_time
    
    # 6. Convert ECI Cartesian state back to TLE elements
    pos_m = x_updated[:3]
    vel_m = x_updated[3:6]
    elements = cartesian_to_keplerian(pos_m, vel_m)
    
    # Construct updated TLE
    tle1_new, tle2_new = format_tle_lines(tle1, tle2, obs_time, elements)
    sat_filter["tle1"] = tle1_new
    sat_filter["tle2"] = tle2_new
    
    # 7. Check for anomalies (drift > 5.0 km)
    drift_report = detect_tle_drift(norad_id, sat_name, tle1, tle2, tle1_new, tle2_new, threshold_km=5.0)
    
    # 8. Store results to database
    try:
        # Create table ground_observations if not exists, and add columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ground_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                norad_id TEXT,
                observation_id TEXT,
                station_id TEXT,
                timestamp TEXT,
                frequency_hz REAL,
                doppler_shift_hz REAL,
                range_rate REAL,
                residual REAL
            )
        """)
        
        # Check if last_observation_time exists in fleet_satellites
        try:
            cursor.execute("ALTER TABLE fleet_satellites ADD COLUMN last_observation_time TEXT")
        except sqlite3.OperationalError:
            pass
            
        # Update TLE & last observation time in database
        cursor.execute("""
            UPDATE fleet_satellites 
            SET tle1 = ?, tle2 = ?, last_observation_time = ?
            WHERE norad_id = ?
        """, (tle1_new, tle2_new, obs_time.isoformat(), norad_id))
        
        # Log this specific observation details
        obs_id = observation.get("observation_id", f"OBS-INGEST-{int(datetime.now().timestamp())}")
        cursor.execute("""
            INSERT INTO ground_observations (norad_id, observation_id, station_id, timestamp, frequency_hz, doppler_shift_hz, range_rate, residual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (norad_id, obs_id, station_id, obs_time.isoformat(), freq_obs, freq_obs - freq_trans, range_rate_obs, residual))
        
        conn.commit()
    except Exception as e:
        logger.error(f"[TLE Corrector] Database logging failed: {e}")
    finally:
        conn.close()
        
    return {
        "status": "success",
        "satellite_id": norad_id,
        "satellite_name": sat_name,
        "obs_time": obs_time.isoformat(),
        "doppler_shift_hz": freq_obs - freq_trans,
        "range_rate_m_s": range_rate_obs,
        "residual_m_s": residual,
        "std_position_before_m": round(std_pos_before, 2),
        "std_position_after_m": round(std_pos_after, 2),
        "tle1_old": tle1,
        "tle2_old": tle2,
        "tle1_new": tle1_new,
        "tle2_new": tle2_new,
        "anomaly_detected": drift_report.get("anomaly_detected", False),
        "drift_km": drift_report.get("deviation_km", 0.0)
    }
