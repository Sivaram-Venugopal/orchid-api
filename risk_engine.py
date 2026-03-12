import numpy as np
from physics import tle_to_state

def assess_risk(satellite, debris_list, time_horizon_hrs=24.0):
    sat_state = tle_to_state(satellite.tle1, satellite.tle2)
    conjunctions = []
    for debris in debris_list:
        deb_state = tle_to_state(debris.tle1, debris.tle2)
        rel_pos = np.array(sat_state[:3]) - np.array(deb_state[:3])
        distance_m = np.linalg.norm(rel_pos)
        distance_km = distance_m / 1000.0
        rel_vel = np.array(sat_state[3:]) - np.array(deb_state[3:])
        closing_speed = np.linalg.norm(rel_vel)
        if closing_speed > 0:
            time_to_ca = distance_m / closing_speed / 60.0
        else:
            time_to_ca = 999.0
        if distance_km < 1.0:
            prob = max(0.0, 1.0 - (distance_km / 1.0))
        else:
            prob = 0.0
        if distance_km < 0.1:
            risk = "CRITICAL"
        elif distance_km < 0.5:
            risk = "HIGH"
        elif distance_km < 2.0:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        conjunctions.append({
            "object_id": debris.norad_id,
            "distance_km": round(float(distance_km), 4),
            "time_to_closest_approach_min": round(float(time_to_ca), 1),
            "risk_level": risk,
            "probability_of_collision": round(float(prob), 7)
        })
    conjunctions.sort(key=lambda x: x["distance_km"])
    return conjunctions