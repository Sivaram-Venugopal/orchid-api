import numpy as np
from scipy.linalg import cholesky
from physics import propagate_rk4

def get_ukf_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Computes Unscented Transform weights for mean and covariance.
    n: state dimension
    """
    # Scaling parameter
    lambda_val = (alpha**2) * (n + kappa) - n
    
    w_m = np.zeros(2*n + 1)
    w_c = np.zeros(2*n + 1)
    
    w_m[0] = lambda_val / (n + lambda_val)
    w_c[0] = lambda_val / (n + lambda_val) + (1.0 - alpha**2 + beta)
    
    for i in range(1, 2*n + 1):
        w_m[i] = 0.5 / (n + lambda_val)
        w_c[i] = 0.5 / (n + lambda_val)
        
    return w_m, w_c, lambda_val

def generate_sigma_points(x, P, n, lambda_val):
    """
    Generates 2n+1 sigma points around the mean state x.
    """
    sigma_pts = np.zeros((2*n + 1, n))
    sigma_pts[0] = x
    
    # Compute matrix square root via Cholesky decomposition
    try:
        chol = cholesky((n + lambda_val) * P)
    except Exception:
        # regularize if covariance matrix is not strictly positive definite
        reg_P = P + np.eye(n) * 1e-5
        try:
            chol = cholesky((n + lambda_val) * reg_P)
        except Exception:
            # Fallback to simple diagonal square root
            chol = np.diag(np.sqrt(np.maximum(1e-9, np.diag((n + lambda_val) * reg_P))))
        
    for i in range(n):
        sigma_pts[i + 1] = x + chol[i]
        sigma_pts[i + 1 + n] = x - chol[i]
        
    return sigma_pts

def ukf_propagate(x, P, dt_sec, Q, mass=550.0, area=12.0, drag_coeff=2.2):
    """
    Propagates mean state x and covariance P over dt_sec using the Unscented Transform.
    Q: Process noise covariance matrix
    """
    n = len(x)
    w_m, w_c, lambda_val = get_ukf_weights(n)
    
    # Generate sigma points
    sigma_pts = generate_sigma_points(x, P, n, lambda_val)
    
    # Propagate sigma points through RK4 numerical dynamics
    propagated_sigmas = np.zeros((2*n + 1, n))
    for i in range(2*n + 1):
        r_init = sigma_pts[i, :3].tolist()
        v_init = sigma_pts[i, 3:].tolist()
        r_prop, v_prop = propagate_rk4(
            r_init, v_init, dt_sec,
            mass=mass, area=area, drag_coeff=drag_coeff, step_size_sec=10.0
        )
        propagated_sigmas[i] = np.array(r_prop + v_prop)
        
    # Recombine to predict mean
    x_pred = np.zeros(n)
    for i in range(2*n + 1):
        x_pred += w_m[i] * propagated_sigmas[i]
        
    # Recombine to predict covariance
    P_pred = np.zeros((n, n))
    for i in range(2*n + 1):
        diff = propagated_sigmas[i] - x_pred
        P_pred += w_c[i] * np.outer(diff, diff)
    P_pred += Q
    
    return x_pred, P_pred, propagated_sigmas

def measurement_model(state, sensor_type="cartesian", sensor_location=None):
    import math
    import numpy as np
    
    pos = state[:3]
    if sensor_location is not None:
        r_rel = pos - np.array(sensor_location)
    else:
        r_rel = pos
        
    rx, ry, rz = r_rel[0], r_rel[1], r_rel[2]
    r = math.sqrt(rx*rx + ry*ry + rz*rz)
    
    if sensor_type == "cartesian":
        return pos.copy()
        
    elif sensor_type == "laser":
        return np.array([r])
        
    elif sensor_type == "radar":
        if r < 1e-3:
            return np.array([0.0, 0.0, 0.0])
        az = math.atan2(ry, rx)
        el = math.asin(rz / r)
        return np.array([r, az, el])
        
    elif sensor_type == "optical":
        if r < 1e-3:
            return np.array([0.0, 0.0])
        ra = math.atan2(ry, rx)
        dec = math.asin(rz / r)
        return np.array([ra, dec])
        
    elif sensor_type == "doppler":
        if r < 1e-3:
            return np.array([0.0])
        if sensor_location is not None and len(sensor_location) >= 6:
            v_station = np.array(sensor_location[3:6])
        else:
            v_station = np.zeros(3)
        v_rel = state[3:6] - v_station
        range_rate = np.dot(r_rel, v_rel) / r
        return np.array([range_rate])
        
    else:
        return pos.copy()

def ukf_update(x_pred, P_pred, sigmas_pred, z, R, sensor_type="cartesian", sensor_location=None):
    """
    Updates predicted state mean and covariance with a new noisy measurement from any sensor.
    z: Measured vector (meters or radians)
    R: Measurement noise covariance matrix
    sensor_type: "cartesian", "radar", "optical", or "laser"
    sensor_location: Ground station ECI coordinates [X, Y, Z] (optional)
    """
    n = len(x_pred)
    w_m, w_c, lambda_val = get_ukf_weights(n)
    
    # Map sigma points through non-linear measurement model
    m_dim = len(z)
    sigmas_meas = np.zeros((2*n + 1, m_dim))
    for i in range(2*n + 1):
        sigmas_meas[i] = measurement_model(sigmas_pred[i], sensor_type, sensor_location)
        
    # Compute predicted measurement mean
    z_pred = np.zeros(m_dim)
    for i in range(2*n + 1):
        z_pred += w_m[i] * sigmas_meas[i]
        
    # Compute measurement covariance P_zz
    P_zz = np.zeros((m_dim, m_dim))
    for i in range(2*n + 1):
        diff_z = sigmas_meas[i] - z_pred
        P_zz += w_c[i] * np.outer(diff_z, diff_z)
    P_zz += R
    
    # Compute cross-covariance P_xz
    P_xz = np.zeros((n, m_dim))
    for i in range(2*n + 1):
        diff_x = sigmas_pred[i] - x_pred
        diff_z = sigmas_meas[i] - z_pred
        P_xz += w_c[i] * np.outer(diff_x, diff_z)
        
    # Compute Kalman Gain
    try:
        K = P_xz @ np.linalg.inv(P_zz)
    except Exception:
        K = P_xz @ np.linalg.pinv(P_zz)
        
    # Update mean and covariance
    x_updated = x_pred + K @ (z - z_pred)
    P_updated = P_pred - K @ P_zz @ K.T
    
    return x_updated, P_updated
