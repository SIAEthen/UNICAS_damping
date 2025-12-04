import numpy as np



# -------------------------------------------
# MDH 单步变换：a, alpha, d, theta
# -------------------------------------------
def TMDH(a, alpha, d, theta):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    T = np.array([
        [ ct,           -st,    0,         a ],
        [ st*ca,      ct*ca,    -sa,   -sa*d ],
        [ st*sa,      ct*sa,     ca,    ca*d ],
        [  0,         0,       0,            1 ]
    ], dtype=float)
    return T


# -------------------------------------------
# 正运动学（MDH）
# MDH: N×4 数组，每行 [a, alpha, d, theta]
# 返回 T_0_N
# -------------------------------------------
def direct_kinematics_mdh(MDH):
    T = np.eye(4)
    n = MDH.shape[0]
    for i in range(n):
        a, alpha, d, theta = MDH[i]
        Ti = TMDH(a, alpha, d, theta)
        T = T @ Ti
    return T


# -------------------------------------------
# 雅可比矩阵 J (6×n)
# revolute joints
# -------------------------------------------
def jacobian_mdh(MDH):
    n = MDH.shape[0]

    # 存 frame origin 和 z 轴
    p = np.zeros((3, n+1))
    z = np.zeros((3, n))

    T_i_0 = np.eye(4)

    # Forward kinematics and storing frames
    for i in range(n):
        a, alpha, d, theta = MDH[i]
        TT = TMDH(a, alpha, d, theta)
        T_i_0 = T_i_0 @ TT

        # 提取 z_i-1
        z[:, i] = T_i_0[0:3, 2]
        # 提取 p_i
        p[:, i+1] = T_i_0[0:3, 3]

    p_end = p[:, -1]

    # Jacobian
    J = np.zeros((6, n))

    for i in range(n):
        Jp = np.cross(z[:, i], p_end - p[:, i])  # 线速度部分
        Jo = z[:, i]                              # 角速度部分
        J[0:3, i] = Jp
        J[3:6, i] = Jo

    return J
# ---- Inverse methods ----
def j_pinv(J):
    return np.linalg.pinv(J)

def j_dls(J, lam=0.05):
    # Damped Least Squares solution matrix: J^T (J J^T + lambda^2 I)^{-1}
    JJt = J @ J.T
    reg = (lam**2) * np.eye(JJt.shape[0])
    inv = np.linalg.inv(JJt + reg)
    return J.T @ inv

def j_transpose(J, alpha=0.5):
    # simple transpose scaling
    # choose alpha small enough to keep stable; user may tune
    return alpha * J.T

def clamp(x, low, high):
    return np.maximum(np.minimum(x, high), low)

# -------------------------------------------
# Franka Emika Panda 的 MDH 参数
# （你提供的数据直接复现）
# -------------------------------------------
a = np.array([0,0,0,0.0825,-0.0825,0.0,0.088,0.0])
alpha = np.array([0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0.0])
d = np.array([0.333,0.0,0.316,0.0,0.384,0.0,0.0,0.107])
theta = np.zeros(8)   # 初始所有关节角

emika_mdh = np.vstack([a, alpha, d, theta]).T


# -------- 测试 -------
if __name__ == "__main__":
    T = direct_kinematics_mdh(emika_mdh)
    J = jacobian_mdh(emika_mdh)

    print("T(EE->0) =\n", T)
    print("\nJacobian =\n", J)