import re
import math
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def parse_cdm_kvn(cdm_text: str) -> Dict[str, Any]:
    """
    Parses a CCSDS Conjunction Data Message (CDM) in Key-Value Notation (KVN).
    Extracts NORAD IDs, TLEs, covariance values, and TCA to construct a ManeuverRequest.
    """
    lines = cdm_text.splitlines()
    
    metadata = {}
    obj1_data = {}
    obj2_data = {}
    
    current_object = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("COMMENT"):
            continue
            
        # Parse KEY = VALUE (comments/units stripped)
        parts = line.split("=", 1)
        if len(parts) != 2:
            continue
            
        key = parts[0].strip().upper()
        value = parts[1].split(";", 1)[0].strip() # strip unit comments e.g. "0.452 ; [km]"
        
        # Track active object block
        if key == "OBJECT":
            current_object = value.upper() # "OBJECT1" or "OBJECT2"
            continue
            
        if current_object == "OBJECT1":
            obj1_data[key] = value
        elif current_object == "OBJECT2":
            obj2_data[key] = value
        else:
            metadata[key] = value

    # Extract global variables
    tca_str = metadata.get("TCA")
    collision_prob = float(metadata.get("COLLISION_PROBABILITY", 0.0))
    hbr = float(metadata.get("COLLISION_THRESHOLD", 20.0)) # default HBR
    
    # Helper to resolve TLE/metadata for an object
    def build_object_input(obj_data: dict) -> Tuple[dict, dict]:
        norad_id = obj_data.get("OBJECT_DESIGNATOR")
        name = obj_data.get("OBJECT_NAME", f"NORAD {norad_id}")
        
        tle1 = obj_data.get("TLE_LINE1")
        tle2 = obj_data.get("TLE_LINE2")
        
        # Parse covariances
        # Variance is in m**2, we need standard deviation (meters)
        try:
            r = math.sqrt(float(obj_data.get("CR_R", 10000.0)))
            t = math.sqrt(float(obj_data.get("CT_T", 250000.0)))
            n = math.sqrt(float(obj_data.get("CN_N", 10000.0)))
        except (ValueError, TypeError):
            r, t, n = 100.0, 500.0, 100.0
            
        cov = {"r": r, "t": t, "n": n}
        
        tle_payload = None
        if tle1 and tle2:
            tle_payload = {
                "norad_id": norad_id,
                "tle1": tle1,
                "tle2": tle2
            }
            
        return {
            "norad_id": norad_id,
            "name": name,
            "tle_payload": tle_payload
        }, cov

    sat_info, sat_cov = build_object_input(obj1_data)
    deb_info, deb_cov = build_object_input(obj2_data)
    
    # Compute relative time horizon from current time to TCA in hours
    time_horizon_hrs = 24.0
    if tca_str:
        try:
            from datetime import datetime, timezone
            # Support various ISO time formats (space track has decimal seconds)
            clean_tca = tca_str.replace("Z", "")
            if "." in clean_tca:
                clean_tca = clean_tca.split(".")[0]
            
            tca_dt = datetime.strptime(clean_tca, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff_sec = (tca_dt - now).total_seconds()
            if diff_sec > 0:
                time_horizon_hrs = max(1.0, diff_sec / 3600.0)
        except Exception as e:
            logger.warning(f"Failed to parse TCA timestamp '{tca_str}': {e}. Using default 24h horizon.")

    # Construct the ORCHID ManeuverRequest format
    # If explicit TLE details are missing from CDM, we return the IDs so catalog resolves them
    request_payload = {
        "time_horizon_hrs": round(time_horizon_hrs, 2),
        "hard_body_radius": hbr,
        "satellite_covariance": sat_cov,
        "debris_covariance": deb_cov
    }
    
    if sat_info["tle_payload"]:
        request_payload["satellite"] = sat_info["tle_payload"]
    else:
        request_payload["satellite_id"] = sat_info["norad_id"]
        
    if deb_info["tle_payload"]:
        request_payload["debris"] = [deb_info["tle_payload"]]
    else:
        # If we only have ID, debris_pool resolution logic in main.py will parse it
        pass
        
    return request_payload
