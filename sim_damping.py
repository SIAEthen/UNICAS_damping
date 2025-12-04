# Python code to run a Franka-like 7-DOF kinematics simulation and compare different Jacobian-inverse methods.
# The Jacobian is computed numerically (finite differences) from the forward kinematics function.
# Methods compared:
#  - Moore-Penrose pseudoinverse (pinv)
#  - Damped Least Squares (DLS)
#  - Jacobian Transpose scaled (JT)
#  - DLS + nullspace projection to maintain a preferred posture (DLS-NS)
#
# Outputs:
#  - Plots of end-effector tracking error for each method
#  - Joint trajectories (plotted)
#  - Dataframe of joint trajectories saved and displayed for the user
#
# Note: The kinematic parameters used here are approximate and chosen to give a plausible 7-DOF manipulator
# behavior similar in shape to research arms. If you have a specific URDF or exact DH parameters for Franka Emika Panda,
# you can replace the `forward_kinematics` implementation accordingly.
#
# This cell will run the simulation and produce plots.

import numpy as np
import matplotlib.pyplot as plt
from utilts_mdh_math import *
from math import sin, cos

# ---- Trajectory generation (end effector) ----
def circle_trajectory(center=np.array([0.4, 0.0, 0.4]), radius=0.05, duration=8.0, dt=0.02):
    t = np.arange(0, duration, dt)
    traj = []
    for ti in t:
        theta = 2*np.pi * ti / duration
        pos = center + np.array([radius*np.cos(theta), radius*np.sin(theta), 0.0])
        traj.append(pos)
    traj = np.array(traj)
    # velocities by finite differences (simple)
    vel = np.vstack([np.zeros(3), (traj[1:] - traj[:-1]) / dt])
    return t, traj, vel

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
        p = get_ee_position(q)
        err = pd - p
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
methods = ['pinv','dls','transpose','dls-ns']
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

