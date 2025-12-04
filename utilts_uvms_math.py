import numpy as np
from math import cos,sin,sqrt 
def Rot_dh(alpha,theta):   #pass test
    ct = cos(theta)
    st = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)

    R = np.array([[ct,-st*ca,st*sa],
                  [st ,ct*ca, -ct*sa],
                  [ 0 ,sa ,ca]])
    return R

def Homogeneous_dh(DH_i):  #pass test with  order a alpha d theta
# input: DH_i     dim 1x4     row i of the Denavit-Hartenberg table
# output: T      dim 4x4      Homogeneous transformation matrix  T_i^{i-1}
# Homogeneous transformation matrix between consecutive frames according to DH convention
    a = DH_i[0]
    alpha=DH_i[1]
    d=DH_i[2]
    theta=DH_i[3]
    R = Rot_dh(alpha,theta)
    #print("rotational matrix",R)
    p = np.array([a*cos(theta),a*sin(theta),d]).reshape(3,1)
    T = np.vstack( (np.hstack((R   ,  p)),
                    np.array([0,0,0,1])))
    #print("T",T)
    return T

def S(a):     # pass test

    a1=a[0]
    a2=a[1]
    a3=a[2]
    sa = np.array([ [0,-a3,a2],
                    [a3,0,-a1],
                    [-a2,a1,0] ]  )
    return sa

def J_man(DH):   #pass test  
    #np.dot()   is not the same as *, pay attention!
    # calcute base Jacobian matrix from frame 2 to frame 0, frame0 is the same with frame1 but 0 is static
    # if want to calcute ee to frame0, then DH need to be 3*4, the third line is static, is the ee frame to joint2 frame
    # one leg only have 2 joints.(neglect third joint)
    n = len(DH)   #(for array with 2 dimention, len() will return row of the array)
                     # for array with only 1 dimention, len() will return num of elements of the array
    J = np.zeros((6,n))
    p = np.zeros((3,n))  #i coloum of p matrix is position vector from base frame to frame i
    z = np.zeros((3,n))
    # compute homog. transf. from base frame
    T0 = np.zeros((4,4,n))
    T0[:,:,0] = Homogeneous_dh(DH[0,:])
    # p: 3xn matrix, the generic column is the position of frame i expressed in inertial frame
    p[:,0] = T0[0:3,3,0]
    # z: 3xn matrix, the generic column is the z-versor of frame i expressed in inertial frame
    z[:,0] = T0[0:3,2,0]
    
    for i in range(1,n):
        T_i = Homogeneous_dh(DH[i,:])
        T0[:,:,i] = np.dot(T0[:,:,i-1],T_i)
        p[:,i] = T0[0:3,3,i]
        z[:,i] = T0[0:3,2,i]
    z0 = np.array([0,0,1]).T
    p0 = np.array([0,0,0]).T
    J[:,0] = np.hstack( (np.cross(z0, p[:,n-1]-p0 )   ,
                                         z0    )).T
    for i in range(1,n):
        J[:,i] = np.hstack( [np.cross(z[:,i-1],p[:,n-1]-p[:,i-1]),
                                             z[:,i-1]     ]) .T                                    
    return J

def Rpy2Rot(rpy):  #pass test   if we want get R_i^b, just transpose this func is ok
    #R_I_B = Rpy2Rot(eta[3:6]).transpose()
    #R_B_I = Rpy2Rot(eta[3:6])
    phi   = rpy[0]
    theta = rpy[1]
    psi   = rpy[2]
    cp = cos(psi) ;sp = sin(psi);ct = cos(theta);st = sin(theta);cf = cos(phi);sf = sin(phi)
    R_B_I = np.array([  [ cp*ct, -sp*cf+cp*st*sf , sp*sf+cp*cf*st        ],
                        [ sp*ct , cp*cf+sf*st*sp ,-cp*sf+st*sp*cf         ],
                        [   -st  ,   ct*sf        ,      ct*cf        ]    ])
    return R_B_I  #R_b^i   

def Directkinematics(eta,DH,T_0_B):    # pass test
    #direct kinematics  
    # input:
    #       eta    dim 6x1     vehicle position/orientation
    #       DH     dim nx4     Denavit-Hartenberg table
    #       T_0_B  dim 4x4     Homogeneous transformation               T_0^B
    #                          matrix from vehicle to zero frame
    #output:T      dim 4x4    Homogeneous transformation  matrix from inertial to end-effector   
    n = len(DH)     # joint number
    eta1 = eta[0:3]
    eta2 = eta[3:6]
    RBI = Rpy2Rot(eta2)
    T = np.vstack( (np.hstack((RBI,eta1.reshape(3,1))),
                np.array([0,0,0,1])  ))   #T_B^I
    # from vehicle-fixed to zero
    T = np.dot(T,T_0_B)
    # manipulator cycle
    for i in range (n):
        TT = Homogeneous_dh(DH[i,:])
        T = np.dot(T,TT)
    return T  
def Directkinematics2(DH,T_0_B):    # pass test
    #usage  T = Directkinematics2(DH,T_0_B)
    #direct kinematics  
    # input:
    #       eta    dim 6x1     vehicle position/orientation
    #       DH     dim nx4     Denavit-Hartenberg table
    #       T_0_B  dim 4x4     Homogeneous transformation               T_0^B
    #                          matrix from vehicle to zero frame
    #output:T      dim 4x4    Homogeneous transformation  matrix from bodyframe to end-effector   
    #  T = T_0_B * T_1_0 * T_ee_1   = T_ee^b
    n = len(DH)     # joint number
    T = T_0_B                  
    # manipulator cycle
    for i in range (n):
        TT = Homogeneous_dh(DH[i,:])
        T = np.dot(T,TT)
    return T  
def Directkinematics3(eta,DH,T_0_B):    # pass test
    #direct kinematics  
    # input:
    #       eta    dim 6x1     vehicle position/orientation
    #       DH     dim nx4     Denavit-Hartenberg table
    #       T_0_B  dim 4x4     Homogeneous transformation               T_0^B
    #                          matrix from vehicle to zero frame
    #output:T      dim 4x4    Homogeneous transformation  matrix from inertial to end-effector   
    # EE FRAME to body fixed NED frame
    n = len(DH)     # joint number
    eta1 = 0*eta[0:3]
    eta2 = eta[3:6]
    RBI = Rpy2Rot(eta2)
    T = np.vstack( (np.hstack((RBI,eta1.reshape(3,1))),
                np.array([0,0,0,1])  ))   #T_B^I
    # from vehicle-fixed to zero
    T = np.dot(T,T_0_B)
    # manipulator cycle
    for i in range (n):
        TT = Homogeneous_dh(DH[i,:])
        T = np.dot(T,TT)
    return T  
def Jacobian(eta,DH,T_0_B):   #pass test   vee jacobian and v3 jacobian are the same
    # Computes the Jacobian from zita to end-effector linear and angular velocities expressed in the inertial frame
    #vee = J * [v1 v2 q_dot]'
    n = len(DH)
    eta1 = eta[0:3]
    eta2 = eta[3:6]
    r_B0_B = T_0_B[0:3,3]
    R_0_B = T_0_B[0:3,0:3]
    J = np.zeros((6,6+n))

    Jman = J_man(DH);   R_B_I = Rpy2Rot(eta2);    R_0_I = np.dot(R_B_I , R_0_B)
    T = Directkinematics(eta,DH,T_0_B); eta_ee1 = T[0:3,3];  r_B0_I = np.dot(R_B_I,r_B0_B);   eta_0ee_I = eta_ee1 - eta1 - r_B0_I

    # Position Jacobian
    J[0:3,0:3]   = R_B_I
    J[0:3,3:6]   = np.dot(-(S(r_B0_I) + S(eta_0ee_I))   , R_B_I)
    J[0:3,6:6+n] = np.dot(R_0_I,  Jman[0:3,:])

    # Orientation Jacobian
    J[3:6,0:3]   = np.zeros((3,3))
    J[3:6,3:6]   = R_B_I
    J[3:6,6:6+n] = np.dot(R_0_I,Jman[3:6,:])
    return J

def J_ko(rpy):    #pass test
    phi   = rpy[0]
    theta = rpy[1]
    psi   = rpy[2]
    cp = cos(psi);		sp = sin(psi)
    ct = cos(theta);	st = sin(theta)
    cf = cos(phi);		sf = sin(phi)
    J_ko = np.array([[ 1    ,0      ,-st ],
                    [ 0     ,cf     ,ct*sf    ],
                    [ 0     ,-sf    ,ct*cf    ]])
    #J_ko   $ v_2 = J_ko * \eta_2_dot $
    return J_ko
def J_ko_inv(rpy):    #pass test
    phi   = rpy[0]
    theta = rpy[1]
    psi   = rpy[2]
    cp = cos(psi);		sp = sin(psi)
    ct = cos(theta);	st = sin(theta)
    cf = cos(phi);		sf = sin(phi)
    J_ko_inv = 1/ct  * np.array([[ ct    ,sf*st      ,cf *st ],
                                 [ 0     ,cf * ct     ,-sf*ct    ],
                                  [ 0     ,  sf    ,  cf   ]])
    #J_ko   $ v_2 = J_ko * \eta_2_dot $
    #J_ko_inv    eta2_dot = J_ko_inv * v2
    return J_ko_inv
def J_e(rpy):   #v = J_e *  eta_dot   pass test 
    # jacobian from velociety in inertia frame to body frame
    phi   = rpy[0]
    theta = rpy[1]
    psi   = rpy[2]
    cp = cos(psi);		sp = sin(psi)
    ct = cos(theta);	st = sin(theta)
    cf = cos(phi);		sf = sin(phi)
    Jko = J_ko(rpy)
    R_B_I = Rpy2Rot(rpy)
    zeros = np.zeros((3,3))
    J_e = np.vstack(( np.hstack((R_B_I.T,zeros))      ,
                        np.hstack((zeros, Jko))      ))#R_B_I.T is inverse of R_B_I
    #v = J_e *  eta_dot
    return J_e
def CheckArray(input):
    out = None
    if type(input) is np.ndarray:
        if (input.shape[0]==1):
            out = input.transpose()
        else:
            pass
        return out
    else:
        print("please use ndarray type")
        return out


def VectorLimit(input,sat):
    # input = CheckArray(input)
    #input should be a one size array
    n = input.shape[0]
    # input = input.shape(n)
    for i in range(n):
        if abs(input[i])>sat[i]:
            input[i] = Sign(input[i])*sat[i]
    return input


def ScalarLimit(input,sat):
    # input = CheckArray(input)
    #input should be a one size array
    if abs(input)>sat:
        input = Sign(input)*sat
    return input


def Sign(input):
    if input>0:
        return 1
    if input<0:
        return -1
    if input ==0:
        print("input is zero")
        return 1


def mypinv(J,w):
    l = 1e-5
    n = J.shape[0]
    r = np.linalg.matrix_rank(J)
    if r == n:
        out = np.dot(np.dot(w,J.transpose()) ,
            np.linalg.inv(np.dot(np.dot(J,w),J.transpose()))  )
    else:
        out = np.dot(np.dot(w,J.transpose()) ,
            np.linalg.inv(np.dot(np.dot(J,w),J.transpose())  + l*np.eye(n)      )  )
    return out


def mypinv2(J,W,A):
    out = mypinv( np.dot(np.dot(J.transpose(),A),J),W )
    out = np.dot(out,J.transpose())
    out = np.dot(out,A)
    out = np.dot(out,A)
    return out


def mysigmoid(low_bound,upper_bound,x):
    if (x>upper_bound):
        out = 1
    elif (x < low_bound):
        out = 0
    else :
        out = (x-low_bound) /(upper_bound-low_bound)
    return out


def msgToNumpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])


def worldToBody(orientation, vector):
    """ 
    Rotates world-frame vector to body-frame

    Rotates vector to body frame

    Parameters:
    orientation (np.array): The current orientation of the robot as a quaternion
    vector (np.array): The 3D vector to rotate

    Returns: 
    np.array: 3 dimensional rotated vector

    """
    vector = np.append(vector, 0)
    orientationInv = quat_conj(orientation)
    newVector = quat_mult(orientationInv, quat_mult(vector, orientation))
    return newVector[:3]


def applyMax(vector, max_vector):
    """ 
    Scales a vector to obey maximums

    Parameters:
    vector (np.array): The unscaled vector
    max_vector (np.array): The maximum values for each element

    Returns: 
    np.array: Vector that obeys the maximums

    """
    scale = 1
    for i in range(len(vector)):
        if abs(vector[i]) > max_vector[i]:
            element_scale = max_vector[i] / abs(vector[i])
            if element_scale < scale:
                scale = element_scale

    return vector * scale


def list2array(list):
    return np.array(list)


def VectorNormalization(vec):
    temp = sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
    return vec/temp


def CheckVector(array_data):
    # covert array to one type
    return array_data.reshape(array_data.size)


def CheckVector2(array_data):
    # covert array to one type
    return array_data.reshape(array_data.size,1)


def Rot2Quat(R):
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w)
    No external libraries required.
    """
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    tr = r00 + r11 + r22

    if tr > 0:
        S = sqrt(tr + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (r21 - r12) / S
        y = (r02 - r20) / S
        z = (r10 - r01) / S

    elif (r00 > r11) and (r00 > r22):
        S = sqrt(1.0 + r00 - r11 - r22) * 2  # S = 4 * x
        w = (r21 - r12) / S
        x = 0.25 * S
        y = (r01 + r10) / S
        z = (r02 + r20) / S

    elif r11 > r22:
        S = sqrt(1.0 + r11 - r00 - r22) * 2  # S = 4 * y
        w = (r02 - r20) / S
        x = (r01 + r10) / S
        y = 0.25 * S
        z = (r12 + r21) / S

    else:
        S = sqrt(1.0 + r22 - r00 - r11) * 2  # S = 4 * z
        w = (r10 - r01) / S
        x = (r02 + r20) / S
        y = (r12 + r21) / S
        z = 0.25 * S
    # Normalize
    norm = sqrt(x*x + y*y + z*z + w*w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([x, y, z, w])


def Rpy2Quat(rpy):
    mat = Rpy2Rot(rpy)
    quat = Rot2Quat(mat)
    return check_quaternion(quat) # x y z w


# input array type quaternion, x y z w sequence
def check_quaternion(q):
    assert isinstance(q, np.ndarray), "Input must be a NumPy array"
    if q[3] < 0:
        return -1.0 * q
    else:
        return q

# quaternions always x y z w
def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
# quaternions always x y z w
def quat_err(q_d, q_now):
    # quaternion error 
    q_e = quat_mult(q_d,quat_conj(q_now))
    q_e = check_quaternion(q_e)    
    return q_e[0:3]

def integrate_quaternion(q, omega, dt):
    norm_omega = np.linalg.norm(omega)
    if norm_omega < 1e-8:
        return q  # no rotation
    theta = norm_omega * dt
    axis = omega / norm_omega
    dq = np.hstack([axis*np.sin(theta/2) ,  np.cos(theta/2)])
    q_next = quat_mult(q, dq)
    q_next /= np.linalg.norm(q_next)
    return q_next

def Quat2Rot(q):
    """
    Convert quaternion [x, y, z, w] to rotation matrix R_b_to_i.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return R
def Quat2Rpy(q):
    """
    Convert quaternion [x, y, z, w] to roll, pitch, yaw (in radians).
    Convention: intrinsic Tait–Bryan angles, XYZ order (roll-pitch-yaw)
    Compatible with ROS and common robotics conventions.
    """
    x, y, z, w = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # clamp to 90°
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def integrate_position_ned(eta1, ori, nu1, dt):
    if(len(ori)==3):
        R_B_I = Rpy2Rot(ori)
    else:
        R_B_I = Quat2Rot(ori)
    return eta1 + R_B_I @ nu1 * dt

def quat_norm(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_conj(q):
    q = np.asarray(q)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def rotvec_to_quat(phi):
    """phi: 3-vector (axis * angle). returns quaternion [w,x,y,z]."""
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = phi / theta
    w = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    return np.concatenate([(s*axis),[w] ])

def quat_to_rotvec(q):
    """Return rotation vector (axis*angle) that rotates current->target given quat q (w,x,y,z)."""
    q = quat_norm(q)
    w = np.clip(q[3], -1.0, 1.0)
    v = q[0:3]
    angle = 2.0 * np.arccos(w)
    if angle < 1e-8:
        # small angle -> approximate rotvec ~ 2*v
        return 2.0 * v
    s = np.sqrt(1 - w*w)
    axis = v / (s + 1e-12)
    # bring angle into [-pi,pi]
    if angle > np.pi:
        angle = angle - 2*np.pi
        axis = -axis
    return axis * angle

def Wrap_PI(array):
    array = np.asarray(array).flatten()
    pi = 3.1415926
    for i in range(len(array)):
        while not (array[i]<pi):
            array[i] = array[i] - 2*pi
        while not (array[i]>-pi):
            array[i] = array[i] + 2*pi
    return array


if __name__ == "__main__":
    rpy = np.array([3.14,0,0])
    qua = Rpy2Quat(rpy)
    print(qua)