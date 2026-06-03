import numpy as np
import logging
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from alerts import trigger_alert

logger = logging.getLogger(__name__)

def detect_tle_drift(norad_id: str, sat_name: str, tle1_old: str, tle2_old: str, tle1_new: str, tle2_new: str, threshold_km: float = 5.0) -> dict:
    """
    Propagates both the old (cached) TLE and the new TLE to the current epoch
    to compute their spatial separation. If the deviation exceeds the threshold,
    it flags an anomaly (uncooperative drift rate or maneuver execution detection).
    """
    try:
        sat_old = Satrec.twoline2rv(tle1_old, tle2_old)
        sat_new = Satrec.twoline2rv(tle1_new, tle2_new)
        
        now = datetime.now(timezone.utc)
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
        
        # Propagate old
        e_old, pos_old, vel_old = sat_old.sgp4(jd, fr)
        # Propagate new
        e_new, pos_new, vel_new = sat_new.sgp4(jd, fr)
        
        if e_old != 0 or e_new != 0:
            logger.warning(f"SGP4 propagation failed during drift check for NORAD {norad_id}.")
            return {"anomaly_detected": False, "reason": "SGP4 propagation error"}
            
        # Compute distance in km
        r_old = np.array(pos_old)
        r_new = np.array(pos_new)
        deviation = np.linalg.norm(r_new - r_old)
        
        anomaly = bool(deviation > threshold_km)
        
        result = {
            "satellite_id": norad_id,
            "satellite_name": sat_name,
            "epoch_utc": now.isoformat(),
            "deviation_km": round(float(deviation), 4),
            "threshold_km": threshold_km,
            "anomaly_detected": anomaly
        }
        
        if anomaly:
            logger.warning(f"[ANOMALY DETECTED] NORAD {norad_id} has drifted {deviation:.4f} km from predicted path!")
            title = f"[ORBITAL ANOMALY] Unusual drift detected for {sat_name} ({norad_id})"
            text = (
                f"Satellite: {sat_name} ({norad_id})\n"
                f"Observed Deviation: {deviation:.4f} km\n"
                f"Allowed Threshold: {threshold_km} km\n"
                f"Detection Time: {now.isoformat()} UTC\n"
                f"Drift Analysis: Non-conforming orbit change detected. Satellite may have performed an unannounced maneuver or experienced thrust anomalies."
            )
            trigger_alert("P1", title, text)
            
        return result
    except Exception as e:
        logger.error(f"Error executing drift check: {e}")
        return {"anomaly_detected": False, "reason": str(e)}

def detect_sensor_anomaly(residuals: list, covariance: np.ndarray) -> bool:
    """
    Performs a 3-sigma statistical threshold check on UKF residuals.
    If the residuals exceed the uncertainty bounds, it flags an observation anomaly.
    """
    try:
        res = np.array(residuals)
        # Extract diagonal variances
        variances = np.diag(covariance)[:len(res)]
        std_devs = np.sqrt(variances)
        
        # Check if residuals exceed 3 times the standard deviation (99.7% confidence interval)
        for i in range(len(res)):
            if abs(res[i]) > 3.0 * std_devs[i]:
                logger.warning(f"[SENSOR ANOMALY] residual at index {i} ({res[i]:.2f}) exceeds 3-sigma variance bound ({3.0 * std_devs[i]:.2f})!")
                return True
    except Exception as e:
        logger.error(f"Error in sensor anomaly checks: {e}")
    return False
