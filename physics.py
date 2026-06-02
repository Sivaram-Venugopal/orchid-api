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

def orbit_derivatives(t, state, mass, area, drag_coeff):
    """
    Computes derivative of state vector [x, y, z, vx, vy, vz] (meters, m/s).
    Models Keplerian gravity, J2 perturbation, and atmospheric drag.
    """
    x, y, z, vx, vy, vz = state
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    r = np.linalg.norm(r_vec)
    
    if r < 1e-3:
        return np.zeros(6)
        
    # Standard Keplerian gravity acceleration
    mu = 3.986004418e14  # m^3/s^2
    a_grav = -mu * r_vec / (r**3)
    
    # J2 gravity perturbation acceleration
    R_E = 6378137.0      # Earth radius in meters
    J2 = 1.08263e-3      # J2 harmonic coefficient
    z_sq = z**2
    r_sq = r**2
    factor = 1.5 * J2 * mu * (R_E**2) / (r**5)
    
    a_J2_x = factor * x * (5.0 * (z_sq / r_sq) - 1.0)
    a_J2_y = factor * y * (5.0 * (z_sq / r_sq) - 1.0)
    a_J2_z = factor * z * (5.0 * (z_sq / r_sq) - 3.0)
    a_J2 = np.array([a_J2_x, a_J2_y, a_J2_z])
    
    # Atmospheric Drag acceleration (rotating atmosphere model)
    omega_E = np.array([0.0, 0.0, 7.292115e-5])  # Earth rotation vector in rad/s
    v_rel = v_vec - np.cross(omega_E, r_vec)
    v_rel_norm = np.linalg.norm(v_rel)
    
    altitude = r - R_E
    # Scale density based on standard scale height model
    if 0.0 < altitude < 1000000.0:
        rho_0 = 3.614e-13  # Density at h_0 = 400km in kg/m^3
        h_0 = 400000.0     # Reference altitude in meters
        H = 58200.0        # Scale height in meters
        rho = rho_0 * np.exp(-(altitude - h_0) / H)
    else:
        rho = 0.0
        
    a_drag = -0.5 * rho * drag_coeff * (area / mass) * v_rel_norm * v_rel
    
    a_total = a_grav + a_J2 + a_drag
    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])

def propagate_rk4(r_init, v_init, dt_sec, mass=550.0, area=12.0, drag_coeff=2.2, step_size_sec=10.0):
    """
    Propagates orbit state forward/backward by dt_sec using RK4 solver.
    """
    state = np.array(r_init + v_init, dtype=float)
    t = 0.0
    steps = int(abs(dt_sec) / step_size_sec)
    dt = np.sign(dt_sec) * step_size_sec
    
    for _ in range(steps):
        k1 = orbit_derivatives(t, state, mass, area, drag_coeff)
        k2 = orbit_derivatives(t + dt/2.0, state + k1 * dt/2.0, mass, area, drag_coeff)
        k3 = orbit_derivatives(t + dt/2.0, state + k2 * dt/2.0, mass, area, drag_coeff)
        k4 = orbit_derivatives(t + dt, state + k3 * dt, mass, area, drag_coeff)
        state += (k1 + 2.0*k2 + 2.0*k3 + k4) * dt / 6.0
        t += dt
        
    # Remainder step for fractions
    rem_dt = dt_sec - (np.sign(dt_sec) * steps * step_size_sec)
    if abs(rem_dt) > 1e-3:
        k1 = orbit_derivatives(t, state, mass, area, drag_coeff)
        k2 = orbit_derivatives(t + rem_dt/2.0, state + k1 * rem_dt/2.0, mass, area, drag_coeff)
        k3 = orbit_derivatives(t + rem_dt/2.0, state + k2 * rem_dt/2.0, mass, area, drag_coeff)
        k4 = orbit_derivatives(t + rem_dt, state + k3 * rem_dt, mass, area, drag_coeff)
        state += (k1 + 2.0*k2 + 2.0*k3 + k4) * rem_dt / 6.0
        
    return state[:3].tolist(), state[3:].tolist()