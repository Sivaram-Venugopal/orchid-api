import numpy as np
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

C = 299792458.0  # Speed of light in m/s
OMEGA_E = 7.292115e-5  # Earth rotation rate in rad/s

def get_station_eci(lat: float, lng: float, alt_m: float, epoch: datetime) -> tuple:
    """
    Computes ECI (TEME) position and velocity vectors of a ground station.
    lat/lng in degrees, alt_m in meters.
    """
    # 1. Convert geodetic coordinates to ECEF (WGS84 approximation)
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    
    R_E = 6378137.0  # Earth equatorial radius in meters
    f = 1.0 / 298.257223563  # flattening
    e_sq = 2 * f - f**2
    
    N = R_E / np.sqrt(1.0 - e_sq * np.sin(lat_rad)**2)
    
    x_ecef = (N + alt_m) * np.cos(lat_rad) * np.cos(lng_rad)
    y_ecef = (N + alt_m) * np.cos(lat_rad) * np.sin(lng_rad)
    z_ecef = (N * (1.0 - e_sq) + alt_m) * np.sin(lat_rad)
    
    r_ecef = np.array([x_ecef, y_ecef, z_ecef])
    
    # 2. Estimate Greenwich Mean Sidereal Time (GMST) to rotate ECEF to ECI
    # Standard formula for GMST based on days since J2000.0
    J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    dt_days = (epoch.astimezone(timezone.utc) - J2000).total_seconds() / 86400.0
    
    # Greenwich Mean Sidereal Time in radians
    gmst_rad = np.radians(280.46061837 + 360.98564736629 * dt_days) % (2.0 * np.pi)
    
    c_g = np.cos(gmst_rad)
    s_g = np.sin(gmst_rad)
    
    # Rotate position about Z-axis
    x_eci = r_ecef[0] * c_g - r_ecef[1] * s_g
    y_eci = r_ecef[0] * s_g + r_ecef[1] * c_g
    z_eci = r_ecef[2]
    
    r_eci = np.array([x_eci, y_eci, z_eci])
    
    # 3. Compute station ECI velocity due to Earth's rotation: v_eci = omega_E x r_eci
    v_eci = np.array([
        -OMEGA_E * r_eci[1],
        OMEGA_E * r_eci[0],
        0.0
    ])
    
    return r_eci, v_eci

def calculate_doppler_shift(f_transmitted: float, r_sat: np.ndarray, v_sat: np.ndarray, 
                            r_station: np.ndarray, v_station: np.ndarray) -> float:
    """
    Computes Doppler shift (Hz) based on relative radial velocity between satellite and station.
    f_observed = f_transmitted * (1 - v_radial / c)   [v_radial > 0 means moving away]
    """
    r_rel = r_sat - r_station
    v_rel = v_sat - v_station
    
    range_dist = np.linalg.norm(r_rel)
    if range_dist < 1e-3:
        return 0.0
        
    # Project relative velocity onto relative position vector (meters/second)
    range_rate = np.dot(r_rel, v_rel) / range_dist
    
    # Doppler shift formula (first-order relativistic approximation)
    doppler_shift = -f_transmitted * (range_rate / C)
    return doppler_shift

def extract_range_rate_from_doppler(f_transmitted: float, f_observed: float) -> float:
    """
    Converts Doppler frequency measurement back to a range-rate (m/s).
    range_rate = -c * (f_observed - f_transmitted) / f_transmitted
    """
    doppler_shift = f_observed - f_transmitted
    range_rate = -C * (doppler_shift / f_transmitted)
    return range_rate

def predict_passes_24h(tle1: str, tle2: str, station_lat: float, station_lng: float, station_alt_m: float, 
                      station_id: str, f_transmitted: float = 437.5e6) -> list:
    """
    Predicts ground station passes over the next 24 hours.
    Returns list of passes with AOS, LOS, max elevation, and max Doppler shift.
    """
    from sgp4.api import Satrec
    from datetime import datetime, timezone, timedelta
    from sgp4.api import jday
    
    satrec = Satrec.twoline2rv(tle1, tle2)
    start_time = datetime.now(timezone.utc)
    
    passes = []
    in_pass = False
    aos_time = None
    max_el = 0.0
    max_doppler = 0.0
    
    # Check elevation at 30-second steps to get precise rise/set times
    steps = int(24 * 3600 / 30)
    for i in range(steps):
        current_time = start_time + timedelta(seconds=i * 30)
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
                      
        e, r, v = satrec.sgp4(jd, fr)
        if e != 0:
            continue
            
        r_sat = np.array(r) * 1000.0  # m
        v_sat = np.array(v) * 1000.0  # m/s
        
        r_station, v_station = get_station_eci(station_lat, station_lng, station_alt_m, current_time)
        
        r_rel = r_sat - r_station
        r_rel_norm = np.linalg.norm(r_rel)
        r_station_norm = np.linalg.norm(r_station)
        
        # Calculate elevation
        sin_el = np.dot(r_rel, r_station) / (r_rel_norm * r_station_norm)
        # Numerical safety clip
        sin_el = max(-1.0, min(1.0, sin_el))
        el_deg = np.degrees(np.arcsin(sin_el))
        
        # 10 degrees tracking threshold
        if el_deg >= 10.0:
            if not in_pass:
                in_pass = True
                aos_time = current_time
                max_el = el_deg
                max_doppler = 0.0
            else:
                max_el = max(max_el, el_deg)
                
            # Track max Doppler shift magnitude during pass
            d_shift = calculate_doppler_shift(f_transmitted, r_sat, v_sat, r_station, v_station)
            max_doppler = max(max_doppler, abs(d_shift))
        else:
            if in_pass:
                in_pass = False
                los_time = current_time
                passes.append({
                    "station_id": station_id,
                    "aos": aos_time.isoformat(),
                    "los": los_time.isoformat(),
                    "max_elevation": round(max_el, 2),
                    "doppler_shift_hz": round(max_doppler, 1)
                })
                
    return passes
