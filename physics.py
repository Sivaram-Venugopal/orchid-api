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

def _derivatives_scalar(x_val, y_val, z_val, vx_val, vy_val, vz_val, J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const):
    import math
    r_sq = x_val*x_val + y_val*y_val + z_val*z_val
    r = math.sqrt(r_sq)
    if r < 1e-3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    r3 = r_sq * r
    g_factor = -mu / r3
    a_grav_x = g_factor * x_val
    a_grav_y = g_factor * y_val
    a_grav_z = g_factor * z_val
    
    z_sq = z_val*z_val
    factor = J2_mu_R2 / (r3 * r_sq)
    a_J2_x = factor * x_val * (5.0 * (z_sq / r_sq) - 1.0)
    a_J2_y = factor * y_val * (5.0 * (z_sq / r_sq) - 1.0)
    a_J2_z = factor * z_val * (5.0 * (z_sq / r_sq) - 3.0)
    
    v_rel_x = vx_val + omega_E * y_val
    v_rel_y = vy_val - omega_E * x_val
    v_rel_z = vz_val
    v_rel_norm = math.sqrt(v_rel_x*v_rel_x + v_rel_y*v_rel_y + v_rel_z*v_rel_z)
    
    altitude = r - R_E
    if 0.0 < altitude < 1000000.0:
        rho = rho_0 * math.exp(-(altitude - h_0) / H)
    else:
        rho = 0.0
        
    drag_factor = drag_const * rho * v_rel_norm
    a_drag_x = drag_factor * v_rel_x
    a_drag_y = drag_factor * v_rel_y
    a_drag_z = drag_factor * v_rel_z
    
    return (
        vx_val, vy_val, vz_val,
        a_grav_x + a_J2_x + a_drag_x,
        a_grav_y + a_J2_y + a_drag_y,
        a_grav_z + a_J2_z + a_drag_z
    )

def propagate_rk4(r_init, v_init, dt_sec, mass=550.0, area=12.0, drag_coeff=2.2, step_size_sec=10.0):
    """
    Propagates orbit state forward/backward by dt_sec using RK4 solver.
    """
    import math
    x, y, z = float(r_init[0]), float(r_init[1]), float(r_init[2])
    vx, vy, vz = float(v_init[0]), float(v_init[1]), float(v_init[2])
    
    steps = int(abs(dt_sec) / step_size_sec)
    dt = (1.0 if dt_sec >= 0 else -1.0) * step_size_sec
    half_dt = dt / 2.0
    sixth_dt = dt / 6.0
    
    mu = 3.986004418e14
    R_E = 6378137.0
    J2 = 1.08263e-3
    omega_E = 7.292115e-5
    rho_0 = 3.614e-13
    h_0 = 400000.0
    H = 58200.0
    
    J2_mu_R2 = 1.5 * J2 * mu * (R_E**2)
    drag_const = -0.5 * drag_coeff * (area / mass)
    
    for _ in range(steps):
        k1_x, k1_y, k1_z, k1_vx, k1_vy, k1_vz = _derivatives_scalar(
            x, y, z, vx, vy, vz, J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k2_x, k2_y, k2_z, k2_vx, k2_vy, k2_vz = _derivatives_scalar(
            x + k1_x * half_dt, y + k1_y * half_dt, z + k1_z * half_dt,
            vx + k1_vx * half_dt, vy + k1_vy * half_dt, vz + k1_vz * half_dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k3_x, k3_y, k3_z, k3_vx, k3_vy, k3_vz = _derivatives_scalar(
            x + k2_x * half_dt, y + k2_y * half_dt, z + k2_z * half_dt,
            vx + k2_vx * half_dt, vy + k2_vy * half_dt, vz + k2_vz * half_dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k4_x, k4_y, k4_z, k4_vx, k4_vy, k4_vz = _derivatives_scalar(
            x + k3_x * dt, y + k3_y * dt, z + k3_z * dt,
            vx + k3_vx * dt, vy + k3_vy * dt, vz + k3_vz * dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        
        x += (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) * sixth_dt
        y += (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) * sixth_dt
        z += (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) * sixth_dt
        vx += (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * sixth_dt
        vy += (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * sixth_dt
        vz += (k1_vz + 2.0 * k2_vz + 2.0 * k3_vz + k4_vz) * sixth_dt
        
    rem_dt = dt_sec - ((1.0 if dt_sec >= 0 else -1.0) * steps * step_size_sec)
    if abs(rem_dt) > 1e-3:
        k1_x, k1_y, k1_z, k1_vx, k1_vy, k1_vz = _derivatives_scalar(
            x, y, z, vx, vy, vz, J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k2_x, k2_y, k2_z, k2_vx, k2_vy, k2_vz = _derivatives_scalar(
            x + k1_x * rem_dt / 2.0, y + k1_y * rem_dt / 2.0, z + k1_z * rem_dt / 2.0,
            vx + k1_vx * rem_dt / 2.0, vy + k1_vy * rem_dt / 2.0, vz + k1_vz * rem_dt / 2.0,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k3_x, k3_y, k3_z, k3_vx, k3_vy, k3_vz = _derivatives_scalar(
            x + k2_x * rem_dt / 2.0, y + k2_y * rem_dt / 2.0, z + k2_z * rem_dt / 2.0,
            vx + k2_vx * rem_dt / 2.0, vy + k2_vy * rem_dt / 2.0, vz + k2_vz * rem_dt / 2.0,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k4_x, k4_y, k4_z, k4_vx, k4_vy, k4_vz = _derivatives_scalar(
            x + k3_x * rem_dt, y + k3_y * rem_dt, z + k3_z * rem_dt,
            vx + k3_vx * rem_dt, vy + k3_vy * rem_dt, vz + k3_vz * rem_dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        
        x += (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) * rem_dt / 6.0
        y += (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) * rem_dt / 6.0
        z += (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) * rem_dt / 6.0
        vx += (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * rem_dt / 6.0
        vy += (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * rem_dt / 6.0
        vz += (k1_vz + 2.0 * k2_vz + 2.0 * k3_vz + k4_vz) * rem_dt / 6.0
        
    return [x, y, z], [vx, vy, vz]

def propagate_rk4_trajectory(r_init, v_init, total_seconds, mass=550.0, area=12.0, drag_coeff=2.2, step_size_sec=10.0):
    """
    Propagates a full orbit trajectory forward in step_size_sec increments.
    Returns a dictionary mapping time offset (int seconds) to state tuple (pos, vel)
    without list-conversion overhead inside the integration loop.
    """
    import math
    x, y, z = float(r_init[0]), float(r_init[1]), float(r_init[2])
    vx, vy, vz = float(v_init[0]), float(v_init[1]), float(v_init[2])
    
    steps = int(total_seconds / step_size_sec)
    trajectory = {}
    trajectory[0] = (np.array([x, y, z]), np.array([vx, vy, vz]))
    
    dt = step_size_sec
    half_dt = dt / 2.0
    sixth_dt = dt / 6.0
    
    mu = 3.986004418e14
    R_E = 6378137.0
    J2 = 1.08263e-3
    omega_E = 7.292115e-5
    rho_0 = 3.614e-13
    h_0 = 400000.0
    H = 58200.0
    
    J2_mu_R2 = 1.5 * J2 * mu * (R_E**2)
    drag_const = -0.5 * drag_coeff * (area / mass)
    
    for step in range(1, steps + 1):
        k1_x, k1_y, k1_z, k1_vx, k1_vy, k1_vz = _derivatives_scalar(
            x, y, z, vx, vy, vz, J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k2_x, k2_y, k2_z, k2_vx, k2_vy, k2_vz = _derivatives_scalar(
            x + k1_x * half_dt, y + k1_y * half_dt, z + k1_z * half_dt,
            vx + k1_vx * half_dt, vy + k1_vy * half_dt, vz + k1_vz * half_dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k3_x, k3_y, k3_z, k3_vx, k3_vy, k3_vz = _derivatives_scalar(
            x + k2_x * half_dt, y + k2_y * half_dt, z + k2_z * half_dt,
            vx + k2_vx * half_dt, vy + k2_vy * half_dt, vz + k2_vz * half_dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        k4_x, k4_y, k4_z, k4_vx, k4_vy, k4_vz = _derivatives_scalar(
            x + k3_x * dt, y + k3_y * dt, z + k3_z * dt,
            vx + k3_vx * dt, vy + k3_vy * dt, vz + k3_vz * dt,
            J2_mu_R2, mu, R_E, omega_E, rho_0, h_0, H, drag_const
        )
        
        x += (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) * sixth_dt
        y += (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) * sixth_dt
        z += (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) * sixth_dt
        vx += (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * sixth_dt
        vy += (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * sixth_dt
        vz += (k1_vz + 2.0 * k2_vz + 2.0 * k3_vz + k4_vz) * sixth_dt
        
        trajectory[int(step * step_size_sec)] = (np.array([x, y, z]), np.array([vx, vy, vz]))
        
    return trajectory