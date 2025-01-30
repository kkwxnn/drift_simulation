#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from casadi import *
import numpy as NP
import matplotlib.pyplot as plt
import pandas as pd 

# Vehicle parameters
m = 2.35 # [kg]
L = 0.257 # [m]
g = 9.81
b = 0.14328 # [m]
a = L - b
G_front = m * g * b / L
G_rear = m * g * a / L
C_x = 116
C_alpha = 197
Iz = 0.045 
mu = 1.31
mu_slide = 0.55

dt = 0.05  # 20 Hz

# Declare variables
x = MX.sym('x')
y = MX.sym('y')
yaw = MX.sym('yaw')
vx = MX.sym('vx')
vy = MX.sym('vy')
yawrate = MX.sym('yawrate')

X = vertcat(x,y,yaw,vx,vy,yawrate)

# Reference variables 
vxg = MX.sym('vxg')
rg = MX.sym('rg')
target = vertcat(vxg,rg)

# Control
vx_cmd = MX.sym('vx_cmd')
steer = MX.sym('steer')
ctrl_input = vertcat(vx_cmd,steer)

circle_radius = 1.5
v_max = 2.5 # [m/s]
steer_max = 0.698 # [rad]

# Reference 
def circle_velocity_target(circle_radius, v):
    vx_goal = v
    r_goal = vx_goal / circle_radius  # Yaw rate for circular motion
    return vx_goal, r_goal

def wrap_to_pi(angle):
    return fmod(angle + pi, 2 * pi) - pi

def tire_dyn(vx, vx_cmd, mu, mu_slide, Fz, C_x, C_alpha, alpha):
    # Longitudinal wheel slip
    K = MX(0)  # Default initialization

    # If Ux_cmd == Ux
    K = if_else(is_equal(vx_cmd, vx), 0, K)
    
    # If Ux == 0
    Fx = MX(0)
    Fy = MX(0)
    Fx = if_else(is_equal(vx, 0), sign(vx_cmd) * mu * Fz, Fx)
    Fy = if_else(is_equal(vx, 0), 0, Fy)
    
    # Otherwise, calculate K
    K = if_else(is_equal(K, 0), (vx_cmd - vx) / fmax(fabs(vx), 1e-6), K)

    # Determine the reverse factor if K < 0
    reverse = MX(1)
    reverse = if_else(lt(K, 0), -1, reverse)  # If K < 0, reverse = -1
    K = if_else(lt(K, 0), fabs(K), K)  # Take the absolute value of K if K < 0

    # Alpha > pi/2 adaptation
    alpha = if_else(gt(fabs(alpha), pi / 2),
                       (pi - fabs(alpha)) * sign(alpha),
                       alpha)

    # Calculate gamma
    gamma = sqrt(C_x**2 * (K / (1 + K))**2 + C_alpha**2 * (tan(alpha) / (1 + K))**2)

    # Simulate "less than or equal to" using fmax
    condition = fmax(lt(gamma, 3 * mu * Fz), is_equal(gamma, 3 * mu * Fz))

    # Friction model using the condition for gamma <= 3 * mu * Fz
    F = MX(0)
    F = if_else(condition,
                   gamma - (2 - mu_slide / mu) * gamma**2 / (3 * mu * Fz) + \
                   (1 - (2 / 3) * (mu_slide / mu)) * gamma**3 / (9 * mu**2 * Fz**2),
                   mu_slide * Fz)

    # Finally, compute the forces
    Fx = if_else(is_equal(gamma, 0), 0, C_x / gamma * (K / (1 + K)) * F * reverse)
    Fy = if_else(is_equal(gamma, 0), 0, -C_alpha / gamma * (tan(alpha) / (1 + K)) * F)

    return Fx, Fy

def alpha_cal(vx,vy,steer,yawrate,a,b):
    alpha_F = if_else(
        is_equal(vx, 0),  # Check if Ux == 0 using CasADi's is_equal
        if_else(
            is_equal(vy, 0),  # Check if Uy == 0
            0,  # Stationary: no slip
            pi / 2 * sign(vy) - steer  # Perfect side slip
        ),
        if_else(
            lt(vx, 0),  # Check if Ux < 0 using CasADi's lt (less than)
            arctan2((vy + a * yawrate), fmax(fabs(vx), 1e-6)) + steer,
            arctan2((vy + a * yawrate), fmax(vx, 1e-6)) - steer  # Normal motion
        )
    )
        
    alpha_R = if_else(
        is_equal(vx, 0),  # Check if Ux == 0 using CasADi's is_equal
        if_else(
            is_equal(vy, 0),  # Check if Uy == 0
            0,  # Stationary: no slip
            pi / 2 * sign(vy)  # Perfect side slip
        ),
        if_else(
            lt(vx, 0),  # Check if Ux < 0 using CasADi's lt (less than)
            arctan2((vy - b * yawrate), fmax(fabs(vx), 1e-6)),
            arctan2((vy - b * yawrate), fmax(vx, 1e-6))  # Normal motion
        )
    )

    # Wrap slip angles
    alpha_F = wrap_to_pi(alpha_F)
    alpha_R = wrap_to_pi(alpha_R)

    return alpha_F, alpha_R


# Tire forces 
alpha_F, alpha_R = alpha_cal(vx,vy,steer,yawrate,a,b)
Fxf, Fyf = tire_dyn(vx, vx_cmd, mu, mu_slide, G_front, C_x, C_alpha, alpha_F)
Fxr, Fyr = tire_dyn(vx, vx_cmd, mu, mu_slide, G_rear, C_x, C_alpha, alpha_R)

# Vehicle dynamics
r_dot = (a * Fyf * cos(steer) - b * Fyr) / Iz
vx_dot = (Fxr - Fyf * sin(steer)) / m + yawrate * vy
vy_dot = (Fyf * cos(steer) + Fyr) / m - yawrate * vx

# Compute velocity direction for terrain frame transformation
V = sqrt(vx**2 + vy**2)
beta = if_else(V == 0, 0, arctan2(vy, vx))  # Avoid division by zero
beta = wrap_to_pi(beta)

# Transform velocities to the terrain frame
x_dot = V * cos(beta + yaw)
y_dot = V * sin(beta + yaw)


T = 10 # Time horizon            
N = 20 # number of control intervals       

X_dot = vertcat(x_dot,y_dot,yawrate,vx_dot,vy_dot,r_dot)

Loss = (vx - vxg)**2 + (yawrate - rg)**2

if True:
   # CVODES from the SUNDIALS suite
   dae = {'x':X, 'u':target, 'p':ctrl_input, 'ode':X_dot, 'quad':Loss} #x state, z ref, p control input, x dot, loss
   F = integrator('F', 'cvodes', dae, 0, T/N)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = T/N/M
   f = Function('f', [x, u], [State, L])
   X0 = MX.sym('X0', 2)
   U = MX.sym('U')
   X = X0
   Q = 0
   for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        # X = X + (DT * k1)
        # Q = Q + (DT * k1_q)
   F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

# Fk = F(x0=[1, 1, 0, 1.0e-6, 1.0e-6, 1.0e-6], p=[0, 0], u=[1,1])
# print(Fk['xf'])
# print(Fk['qf'])

# while (1): pass

w = []
w0 = []
lbx = []
ubx = []
J = 0
g = []
lbg = []
ubg = []

# ref_data = pd.read_csv('resampled_vehicle_data.csv')
# vx_ref = ref_data['local_vx'].values
# yawrate_ref = ref_data['yaw_rate'].values
# target_csv = np.vstack((vx_ref, yawrate_ref))

# Formulate the NLP
Xk = MX([0, 0, 0, 1.0, 1.0, 1.0])
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k),2)
    w += [Uk]
    lbx += [ 0.0, -0.698] # bound [vx_cmd, steer]
    ubx += [ 40.0,  0.698]
    w0 += [0,0]

    # Integrate till the end of the interval
    # vxg = target_csv[0, k]  # Longitudinal velocity reference for time step k
    # rg = target_csv[1, k]  # Yaw rate reference for time step k
    
    # vx = Xk[3]

    # if vx >= 0:
    #     v = v_max
    # elif vx < 0:
    #     v = -v_max

    vxg, rg = circle_velocity_target(circle_radius, v_max)
    
    # Create the control input vector (u) for this time step
    u = vertcat(vxg, rg) # target from csv
    u = [vxg, rg]
    Fk = F(x0=Xk, p=Uk, u=u)
    Xk = Fk['xf'] # state
    J = J+Fk['qf'] # state to cost

    # Add inequality constraint
    g += [Xk[0], Xk[1], Xk[2], Xk[3], Xk[4], Xk[5]]
    lbg += [-inf, -inf, -1.57, -40, 1.0e-6, 1.0e-6] # bound [x,y,yaw,vx,vy,yawrate]
    ubg += [inf, inf, 1.57, 40, inf, inf]

# print(g)

# while(1) : pass
# lbx = np.array(lbx).flatten()  # Flatten to ensure correct shape
# ubx = np.array(ubx).flatten() 

# lbg = np.array(lbg).flatten()  
# ubg = np.array(ubg).flatten()

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
w_opt = sol['x']

print(w_opt)
