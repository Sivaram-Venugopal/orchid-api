import os
import json
import logging
import requests
import re
from datetime import datetime, timezone
from risk_engine import assess_risk

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_conjunctions.json")

def fetch_live_conjunctions_data():
    logger.info("Starting live SOCRATES TLE feed fetch from CelesTrak...")
    url = "https://celestrak.org/SOCRATES/table-socrates.php?NAME=,&ORDER=MAXPROB&MAX=20"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
        
        # Parse NORAD ID pairs from data links like data.php?CATNR=60099,14064
        pairs = re.findall(r"data\.php\?CATNR=(\d+),(\d+)", html)
        logger.info(f"Successfully parsed {len(pairs)} conjunction pairs from CelesTrak SOCRATES.")
        
        conjunctions_results = []
        unique_tles = {}
        
        # Limit to top 20
        active_pairs = pairs[:20]
        
        for idx, (p_id, s_id) in enumerate(active_pairs):
            logger.info(f"Retrieving TLEs and calculating risk for Pair {idx + 1}: {p_id} vs {s_id}")
            
            # Fetch TLE for primary
            if p_id not in unique_tles:
                unique_tles[p_id] = fetch_tle_for_id(p_id)
            # Fetch TLE for secondary
            if s_id not in unique_tles:
                unique_tles[s_id] = fetch_tle_for_id(s_id)
                
            p_tle = unique_tles[p_id]
            s_tle = unique_tles[s_id]
            
            if not p_tle or not s_tle:
                logger.warning(f"Could not retrieve TLEs for pair {p_id} vs {s_id}. Skipping.")
                continue
                
            try:
                # Wrap in format supported by risk_engine
                primary_input = {
                    "norad_id": p_id,
                    "tle1": p_tle["line1"],
                    "tle2": p_tle["line2"]
                }
                secondary_input = {
                    "norad_id": s_id,
                    "tle1": s_tle["line1"],
                    "tle2": s_tle["line2"]
                }
                
                # Execute risk assessment over 7 days (168 hours) - SOCRATES' horizon
                assessments = assess_risk(
                    primary_input, [secondary_input], time_horizon_hrs=168.0,
                    sat_cov_rtn=None, deb_cov_rtn=None, hbr=20.0
                )
                
                if assessments:
                    res = assessments[0]
                    # Classify risk level based on Foster probability & distance
                    prob = res.get("probability_of_collision", 0.0)
                    dist = res.get("distance_km", 999.0)
                    
                    if prob > 1e-4 or dist < 0.1:
                        risk_level = "CRITICAL"
                    elif prob > 1e-5 or dist < 0.5:
                        risk_level = "HIGH"
                    elif prob > 1e-7 or dist < 1.5:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"
                        
                    res["risk_level"] = risk_level
                    
                    conjunctions_results.append({
                        "pair_index": idx + 1,
                        "primary": {
                            "norad_id": p_id,
                            "name": p_tle["name"],
                            "tle1": p_tle["line1"],
                            "tle2": p_tle["line2"]
                        },
                        "secondary": {
                            "norad_id": s_id,
                            "name": s_tle["name"],
                            "tle1": s_tle["line1"],
                            "tle2": s_tle["line2"]
                        },
                        "risk_assessment": res
                    })
            except Exception as e:
                logger.error(f"Error calculating risk for pair {p_id} vs {s_id}: {e}")
                
        # Cache results locally to JSON file
        with open(DATA_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_monitored": len(unique_tles),
                "conjunctions": conjunctions_results
            }, f, indent=2)
            
        logger.info(f"Successfully cached {len(conjunctions_results)} live conjunction risks to local cache.")
    except Exception as e:
        logger.error(f"Failed to fetch live conjunctions feed: {e}")

def fetch_tle_for_id(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line.strip()]
            if len(lines) >= 3:
                return {
                    "name": lines[0],
                    "line1": lines[1],
                    "line2": lines[2]
                }
    except Exception as e:
        logger.error(f"Failed to fetch TLE for NORAD ID {norad_id}: {e}")
    return None

def get_cached_conjunctions():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_monitored": 0,
        "conjunctions": []
    }
