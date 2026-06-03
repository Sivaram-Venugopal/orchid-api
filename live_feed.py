import os
import json
import logging
import requests
import re
import sqlite3
from datetime import datetime, timezone
from risk_engine import assess_risk
from alerts import trigger_alert

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_conjunctions.json")
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conjunction_history.db")

def fetch_live_conjunctions_data():
    logger.info("Starting live SOCRATES TLE feed fetch from CelesTrak...")
    url = "https://celestrak.org/SOCRATES/table-socrates.php?NAME=,&ORDER=MAXPROB&MAX=20"
    
    conjunctions_results = []
    unique_tles = {}
    
    # 1. Fetch SOCRATES public conjunction table
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
        pairs = re.findall(r"data\.php\?CATNR=(\d+),(\d+)", html)
        logger.info(f"Successfully parsed {len(pairs)} conjunction pairs from CelesTrak SOCRATES.")
        
        active_pairs = pairs[:20]
        for idx, (p_id, s_id) in enumerate(active_pairs):
            if p_id not in unique_tles:
                unique_tles[p_id] = fetch_tle_for_id(p_id)
            if s_id not in unique_tles:
                unique_tles[s_id] = fetch_tle_for_id(s_id)
                
            p_tle = unique_tles[p_id]
            s_tle = unique_tles[s_id]
            
            if not p_tle or not s_tle:
                continue
                
            try:
                assessments = assess_risk(
                    {"norad_id": p_id, "tle1": p_tle["line1"], "tle2": p_tle["line2"]},
                    [{"norad_id": s_id, "tle1": s_tle["line1"], "tle2": s_tle["line2"]}],
                    time_horizon_hrs=168.0, sat_cov_rtn=None, deb_cov_rtn=None, hbr=20.0
                )
                if assessments:
                    res = assessments[0]
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
                        "primary": {"norad_id": p_id, "name": p_tle["name"], "tle1": p_tle["line1"], "tle2": p_tle["line2"]},
                        "secondary": {"norad_id": s_id, "name": s_tle["name"], "tle1": s_tle["line1"], "tle2": s_tle["line2"]},
                        "risk_assessment": res
                    })
            except Exception as e:
                logger.error(f"Error calculating risk for SOCRATES pair {p_id} vs {s_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to fetch public SOCRATES feed: {e}")

    # 2. Load fleet satellites and screen against debris catalog
    fleet_conjunctions = []
    fleet_sats = []
    
    try:
        if os.path.exists(DB_FILE):
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT norad_id, name, status, fuel_capacity_kg, current_fuel_kg, tle1, tle2 FROM fleet_satellites")
            rows = cursor.fetchall()
            conn.close()
            for r_row in rows:
                fleet_sats.append({
                    "norad_id": r_row[0],
                    "name": r_row[1],
                    "status": r_row[2],
                    "fuel_capacity_kg": r_row[3],
                    "current_fuel_kg": r_row[4],
                    "tle1": r_row[5],
                    "tle2": r_row[6]
                })
            logger.info(f"Loaded {len(fleet_sats)} fleet satellites from database for active screening.")
    except Exception as e:
        logger.error(f"Failed to read fleet_satellites table: {e}")

    if fleet_sats:
        try:
            from catalog_manager import load_tle_catalog
            catalog = load_tle_catalog()
            debris_pool = [
                {"norad_id": nid, "tle1": d["line1"], "tle2": d["line2"]}
                for nid, d in catalog.items()
                if d.get("type") == "debris"
            ]
            logger.info(f"Screening fleet against {len(debris_pool)} debris objects...")
            
            for idx, sat in enumerate(fleet_sats):
                # Execute 3-stage screening for the fleet satellite
                # Run over 120 hours (5 days)
                primary_input = {"norad_id": sat["norad_id"], "tle1": sat["tle1"], "tle2": sat["tle2"]}
                assessments = assess_risk(
                    primary_input, debris_pool, time_horizon_hrs=120.0,
                    sat_cov_rtn=None, deb_cov_rtn=None, hbr=20.0
                )
                
                # Filter down to close passes (under 15.0 km) to prevent bloating the cache file
                for res in assessments:
                    dist = res.get("distance_km", 999.0)
                    if dist <= 15.0:
                        prob = res.get("probability_of_collision", 0.0)
                        
                        # Classify risk
                        if prob > 1e-4 or dist < 0.15:
                            risk_level = "CRITICAL"
                            alert_level = "P0" # SMS + Email
                        elif prob > 1e-5 or dist < 0.5:
                            risk_level = "HIGH"
                            alert_level = "P1" # Email
                        elif prob > 1e-7 or dist < 1.5:
                            risk_level = "MEDIUM"
                            alert_level = "P2" # Log only
                        else:
                            risk_level = "LOW"
                            alert_level = "P2"
                            
                        res["risk_level"] = risk_level
                        
                        # Load secondary TLE details
                        sec_info = catalog.get(res["object_id"], {"name": "DEBRIS"})
                        
                        fleet_conjunctions.append({
                            "pair_index": len(fleet_conjunctions) + 1,
                            "primary": {"norad_id": sat["norad_id"], "name": sat["name"], "tle1": sat["tle1"], "tle2": sat["tle2"]},
                            "secondary": {"norad_id": res["object_id"], "name": sec_info.get("name"), "tle1": sec_info.get("line1"), "tle2": sec_info.get("line2")},
                            "risk_assessment": res
                        })
                        
                        # Trigger automated alert
                        if alert_level in ["P0", "P1"]:
                            title = f"[{risk_level} ALERT] Conjunction Threat for Fleet Sat {sat['name']} ({sat['norad_id']})"
                            text = (
                                f"Urgency Level: {alert_level}\n"
                                f"Primary Sat: {sat['name']} ({sat['norad_id']})\n"
                                f"Conjoins with Debris: {sec_info.get('name')} ({res['object_id']})\n"
                                f"Closest Approach: {dist:.4f} km\n"
                                f"Collision Probability: {prob:.4e}\n"
                                f"TCA: {res.get('time_to_closest_approach_min', 0.0):.1f} mins\n"
                                f"Maneuver Required: YES\n"
                                f"Fuel State: {sat['current_fuel_kg']:.1f} / {sat['fuel_capacity_kg']:.1f} kg ({sat['status']})"
                            )
                            trigger_alert(alert_level, title, text)
        except Exception as e:
            logger.error(f"Error screening fleet: {e}")

    # 3. Cache combined outputs
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_monitored": len(unique_tles) + len(fleet_sats),
                "conjunctions": conjunctions_results,
                "fleet_conjunctions": fleet_conjunctions
            }, f, indent=2)
        logger.info(f"Successfully cached {len(conjunctions_results)} public and {len(fleet_conjunctions)} fleet conjunctions.")
    except Exception as e:
        logger.error(f"Failed to write live_conjunctions cache: {e}")

def fetch_tle_for_id(norad_id):
    try:
        from spacetrack_client import SpaceTrackClient
        client = SpaceTrackClient()
        res = client.fetch_tle(norad_id)
        if res:
            return res
    except Exception as e:
        logger.error(f"Failed to fetch TLE via SpaceTrackClient for NORAD ID {norad_id}: {e}")
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
        "conjunctions": [],
        "fleet_conjunctions": []
    }
