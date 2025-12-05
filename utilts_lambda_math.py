import numpy as np

def damped_pinv(J, lam):
    """Compute damped least-squares pseudo-inverse."""
    m, n = J.shape
    return J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(m))

def caccavale_damping(J, sigma_d1, lambda_M= 1.0):
    """
    Compute damping factor lambda^2 using the method from
    Caccavale et al. (1997).

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix (m x n)
    sigma_d1 : float
        Threshold singular value
    lambda_M : float
        Maximum damping value (default = 1.0)

    Returns
    -------
    float
        lambda^2 damping factor
    """
    # Compute singular values
    sigma = np.linalg.svd(J, compute_uv=False)
    sigma_min = np.min(sigma)

    # Apply Caccavale formula
    if sigma_min >= sigma_d1:
        return 0.0
    else:
        ratio = sigma_min / sigma_d1
        lambda_sq = (1 - ratio**2) * (lambda_M**2)
        return np.sqrt(lambda_sq)


def baerlocher_damping(J, xr_dot, qdot_max = 2.5):
    """
    Compute damping factor lambda using Baerlocher (2001).

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix (m x n)
    xr_dot : np.ndarray
        Task-space reference velocity (m-dimensional)
    qdot_max : float
        Max joint velocity magnitude

    Returns
    -------
    float
        Damping factor λ
    """

    # ---- Step 1: Compute sigma_min ----
    sigma = np.linalg.svd(J, compute_uv=False)
    sigma_min = np.min(sigma)

    # ---- Step 2: Compute dynamic threshold sigma_d2 ----
    xr_norm = np.linalg.norm(xr_dot)

    if qdot_max <= 1e-9:  # avoid divide by zero
        return 0.0

    sigma_d2 = xr_norm / qdot_max

    # ---- Step 3: Apply Baerlocher formula ----
    if sigma_min >= sigma_d2:
        lam = 0.0

    elif sigma_min >= sigma_d2 / 2.0:
        lam = sigma_min * (sigma_d2 - sigma_min)

    else:
        lam = sigma_d2 / 2.0

    return lam

def iterative_damping(J, xr_dot, qdot_max,
                             lambda_init=0.0,
                             alpha=0.01,
                             iters=10):
    """
    Newton-like iterative method to find optimal damping λ,
    satisfying |qdot| <= qdot_max.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix (m x n)
    xr_dot : np.ndarray
        Task velocity vector
    qdot_max : float
        Maximum allowed joint velocity norm
    lambda_init : float
        Initial guess for λ
    alpha : float
        Gain factor (tuning parameter)
    iters : int
        Number of iterations

    Returns
    -------
    float
        Final optimized λ
    """

    lam = lambda_init

    for k in range(iters):
        # Compute current qdot using DLS λ
        J_pinv = damped_pinv(J, lam)
        qdot = J_pinv @ xr_dot
        qnorm = np.linalg.norm(qdot)

        # Newton-like update (Eq. 11)
        lam = lam + alpha * (qdot_max - qnorm)

        # keep λ non-negative
        lam = max(lam, 0.0)

    return lam

def deoandwalker_damping(J, xr_dot, qdot_max, lambda0=0.01, max_iter=50, tol=1e-6):
    """
    Newton method to compute optimal damping factor λ
    such that ||qdot(λ)|| = qdot_max.

    Parameters
    ----------
    J : ndarray, shape (m, n)
        Jacobian matrix
    xr_dot : ndarray, shape (m,)
        Desired task-space velocity
    qdot_max : float
        Max allowed joint velocity norm
    """

    J = np.asarray(J)
    xr_dot = np.asarray(xr_dot).reshape(-1, 1)
    m, n = J.shape

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    # S is singular values σ_i
    # gamma_i = U^T * xr
    gamma = U.T @ xr_dot
    gamma = gamma.flatten()

    lam = lambda0

    for _ in range(max_iter):
        # Compute qdot norm using analytic expression
        q_norm_sq = np.sum((S * gamma)**2 / (S**2 + lam)**2)
        q_norm = np.sqrt(q_norm_sq)

        phi = q_norm - qdot_max
        if abs(phi) < tol:
            break

        # derivative phi'(λ)
        phi_prime = -np.sum((S**2) * (gamma**2) / (S**2 + lam)**3) / q_norm

        # Newton update
        lam = lam - phi / phi_prime

        # Enforce λ >= 0
        if lam < 0:
            lam = 0.0

    return lam

def sugihara_damping(J, xr_dot, link_length=0.8, WE=None):
    """
    Compute joint velocities using Sugihara (2011) bias-damped least squares.
    Parameters
    ----------
    J : ndarray (m x n)
        Jacobian matrix
    xr_dot : ndarray (m,)
        Task-space velocity
    link_length : float
        Typical kinematic link length (used to set wN)
    WE : ndarray (m x m)
        Task-space weight matrix (default = identity)

    Returns
    -------
    qdot : ndarray (n,)
        Joint velocity computed by Sugihara BDLS method
    WN_total : ndarray (n x n)
        Total damping matrix used (for analysis)
    """
    J = np.asarray(J)
    xr_dot = np.asarray(xr_dot).reshape(-1, 1)
    m, n = J.shape

    # Default WE = identity
    if WE is None:
        WE = np.eye(m)

    # --- step 1: compute E = 1/2 xr^T WE xr
    E = 0.5 * float(xr_dot.T @ WE @ xr_dot)

    # --- step 2: compute WN diagonal bias
    wN_small = (1e-2) * (link_length**2)  # Sugihara suggested 1e-2–1e-3 * l^2

    damping = wN_small + E
    return damping