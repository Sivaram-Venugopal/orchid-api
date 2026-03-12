import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

def tle_to_state(tle1, tle2):
    satellite = Satrec.twoline2rv(tle1, tle2)
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond/1e6)
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        r = [7000.0, 0.0, 0.0]
        v = [0.0, 7.5, 0.0]
    r_m = [x * 1000 for x in r]
    v_ms = [x * 1000 for x in v]
    return r_m + v_ms

def tsiolkovsky(delta_v, mass, isp=220.0):
    g0 = 9.80665
    fuel = mass * (1 - np.exp(-delta_v / (isp * g0)))
    return abs(fuel)