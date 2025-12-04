import numpy as np
import matplotlib.pyplot as plt
from utilts_mdh_math import *
from utilts_uvms_math import *

from math import sin, cos
a = np.array([0,0,0,0.0825,-0.0825,0.0,0.088,0.0])
alpha = np.array([0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0.0])
d = np.array([0.333,0.0,0.316,0.0,0.384,0.0,0.0,0.107])
theta = np.zeros(8)   # 初始所有关节角

def get_ee_state(q):
    mdh_i = np.vstack([a, alpha, d, theta+np.append(q,0)]).T
    T = direct_kinematics_mdh(mdh_i)
    R = T[0:3,0:3]
    p = T[0:3,3]
    q = Rot2Quat(R)
    return np.hstack((p,q))

def get_error(eta_d,eta):
    res = np.zeros(6)
    res[0:3] = eta_d[0:3] - eta[0:3]
    q_d = eta_d[3:7]
    q   = eta[3:7]
    res[3:6] = quat_err(q_d,q)
    return res

def get_jacobian(q):
    mdh_i = np.vstack([a, alpha, d, theta+np.append(q,0)]).T
    return jacobian_mdh(mdh_i)[:,0:7]

# ---- Trajectory generation (end effector) ----
def circle_trajectory(center=np.array([0.4, 0.0, 0.4]), radius=0.05, duration=8.0, dt=0.02):
    t = np.arange(0, duration, dt)
    pos = []
    for ti in t:
        theta = 2*np.pi * ti / duration
        pos_i = center + np.array([radius*np.cos(theta), radius*np.sin(theta), 0.0])
        pos.append(pos_i)
    pos = np.array(pos)
    # velocities by finite differences (simple)
    vel_lin = np.vstack([np.zeros(3), (pos[1:] - pos[:-1]) / dt])

    orientation = []
    for ti in t:
        orie_i = np.array([1,0,0,0]) # always x y z w sequence
        orientation.append(orie_i)
    orientation = np.array(orientation)
    vel_ang = 0*orientation[:,0:3]

    return t, np.hstack((pos,orientation)), np.hstack((vel_lin,vel_ang))

# ---- Resolved-rate control simulation ----
def simulate(method_name, q0, ee_traj, ee_vel, dt=0.02, qdot_limit=2.5, dls_lambda=0.05, null_target=None):
    n_steps = ee_traj.shape[0]
    qs = np.zeros((n_steps, 7))
    q = q0.copy()
    qs[0,:] = q
    ee_errs = np.zeros((n_steps,))
    for k in range(1, n_steps):
        pd = ee_traj[k]
        vd = ee_vel[k]
        eta = get_ee_state(q)
        err = get_error(pd,eta)
        ee_errs[k] = np.linalg.norm(err)
        J = get_jacobian(q)
        if method_name == 'pinv':
            Jinv = j_pinv(J)
            qdot = Jinv @ vd + Jinv @ (2.0 * err)  # combine velocity + proportional feedback
        elif method_name == 'dls':
            Jinv = j_dls(J, lam=dls_lambda)
            qdot = Jinv @ (vd + 2.0 * err)
        elif method_name == 'transpose':
            # choose alpha adaptively using simple heuristic
            alpha = 0.8
            qdot = j_transpose(J, alpha=alpha) @ (vd + 2.0 * err)
        elif method_name == 'dls-ns':
            # DLS with nullspace projection to keep q near null_target
            if null_target is None:
                null_target = q0
            Jinv = j_dls(J, lam=dls_lambda)
            qdot_task = Jinv @ (vd + 2.0 * err)
            # nullspace projector: (I - J# J)
            J_p = np.eye(7) - Jinv @ J
            # gradient to move towards null_target (simple PD)
            q_null = 1.0 * (null_target - q)
            qdot = qdot_task + J_p @ q_null
        else:
            raise ValueError("Unknown method")
        # clamp velocities
        qdot = clamp(qdot, -qdot_limit, qdot_limit)
        q = q + qdot * dt
        qs[k,:] = q
    return qs, ee_errs

# ---- Run comparison ----
dt = 0.02
t, ee_traj, ee_vel = circle_trajectory(center=np.array([0.45, -0.05, 0.35]), radius=0.06, duration=10.0, dt=dt)
n_steps = t.shape[0]
q0 = np.array([0.0,   -0.7826,   0.0,   -2.3550,   0.0,    1.5768,    0.7922-0.7922])  # initial joint posture (home)
methods = ['pinv','dls']
results = {}
for method in methods:
    qs, errs = simulate(method, q0, ee_traj, ee_vel, dt=dt, dls_lambda=0.08, null_target=0.1*np.ones(7))
    results[method] = {'qs':qs, 'errs':errs}

# ---- Plots: End-effector error for each method ----
plt.figure(figsize=(8,4))
for method in methods:
    plt.plot(t, results[method]['errs'], label=method)
plt.xlabel('time [s]')
plt.ylabel('EE position error [m]')
plt.title('End-effector tracking error: comparison of inverse methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plots: Joint trajectories for each method (first 4 joints to keep plots readable) ----
for method in methods:
    qs = results[method]['qs']
    plt.figure(figsize=(8,4))
    for j in range(4):
        plt.plot(t, qs[:,j], label=f'joint{j+1}')
    plt.xlabel('time [s]')
    plt.ylabel('joint angle [rad]')
    plt.title(f'Joint trajectories (first 4 joints) - {method}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

