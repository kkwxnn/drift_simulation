import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Vehicle parameters 
dt = 0.02
m = 2.35
L = 0.257
g = 9.81
b = 0.14328
a = L - b
G_front = m * g * b / L
G_rear = m * g * a / L
C_x = 116  # tire longitudinal stiffness
C_y = 197  # tire lateral stiffness
Iz = 0.045
mu = 1.31  # mu_peak
mu_spin = 0.55  # spinning coefficient

# Circle trajectory parameters
circle_radius = 10.0
circle_center = np.array([0, 0])
v_max = 2.5  # m/s
steer_max = 0.698  # rad

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def tire_dyn(Ux, Ux_cmd, mu, mu_slide, Fz, C_x, C_y, alpha):
    # Longitudinal wheel slip
    if abs(Ux_cmd - Ux) < 1e-6:
        K = 0
    elif abs(Ux) < 1e-6:
        Fx = np.sign(Ux_cmd) * mu * Fz
        Fy = 0
        return Fx, Fy
    else:
        K = (Ux_cmd - Ux) / abs(Ux)

    reverse = 1
    if K < 0:
        reverse = -1
        K = abs(K)

    if abs(alpha) > np.pi / 2:
        alpha = (np.pi - abs(alpha)) * np.sign(alpha)

    gamma = np.sqrt(C_x**2 * (K / max(1 + K, 1e-6))**2 + C_y**2 * (np.tan(alpha) / max(1 + K, 1e-6))**2)

    if gamma <= 3 * mu * Fz:
        F = gamma - (2 - mu_slide / mu) * gamma**2 / (3 * mu * Fz) + \
            (1 - (2 / 3) * (mu_slide / mu)) * gamma**3 / (9 * mu**2 * Fz**2)
    else:
        F = mu_slide * Fz

    if gamma == 0:
        Fx = Fy = 0
    else:
        Fx = C_x / max(gamma, 1e-6) * (K / max(1 + K, 1e-6)) * F * reverse
        Fy = -C_y / max(gamma, 1e-6) * (np.tan(alpha) / max(1 + K, 1e-6)) * F

    return Fx, Fy

def dynamics(x, u):
    pos_x, pos_y, pos_phi, Ux, Uy, r = x
    vx_cmd, delta = u  # Now using vx_cmd and delta as control inputs
    
    # Clip steering angle to physical limits
    delta = np.clip(delta, -steer_max, steer_max)
    
    # Calculate slip angles
    if Ux == 0 and Uy == 0:
        alpha_F = alpha_R = 0
    elif Ux == 0:
        alpha_F = np.pi / 2 * np.sign(Uy) - delta
        alpha_R = np.pi / 2 * np.sign(Uy)
    elif Ux < 0:
        alpha_F = np.arctan((Uy + a * r) / abs(Ux)) + delta
        alpha_R = np.arctan((Uy - b * r) / abs(Ux))
    else:
        alpha_F = np.arctan((Uy + a * r) / abs(Ux)) - delta
        alpha_R = np.arctan((Uy - b * r) / abs(Ux))

    alpha_F, alpha_R = wrap_to_pi(alpha_F), wrap_to_pi(alpha_R)

    # Calculate tire forces - now using vx_cmd as the commanded longitudinal velocity
    Fxf, Fyf = tire_dyn(Ux, Ux, mu, mu_spin, G_front, C_x, C_y, alpha_F)
    Fxr, Fyr = tire_dyn(Ux, vx_cmd, mu, mu_spin, G_rear, C_x, C_y, alpha_R)

    # Vehicle dynamics equations
    r_dot = (a * Fyf * np.cos(delta) - b * Fyr) / Iz
    Ux_dot = (Fxr - Fyf * np.sin(delta)) / m + r * Uy
    Uy_dot = (Fyf * np.cos(delta) + Fyr) / m - r * Ux
    
    # Calculate position changes
    beta = np.arctan2(Uy, Ux) if (Ux != 0 or Uy != 0) else 0
    U = np.sqrt(Ux**2 + Uy**2)
    
    pos_x_dot = U * np.cos(pos_phi + beta)
    pos_y_dot = U * np.sin(pos_phi + beta)
    
    return np.array([pos_x_dot, pos_y_dot, r, Ux_dot, Uy_dot, r_dot])

def dynamics_finite(x, u, dt):
    k1 = dynamics(x, u)
    return x + dt * k1

def circle_target(t, radius, center):
    angle = t * 0.1  # Angular velocity (radians per second)
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    dx = -radius * np.sin(angle)
    dy = radius * np.cos(angle)
    yaw = np.arctan2(dy, dx)
    return np.array([x, y, yaw])

def circle_velocity_target(radius, v):
    angular_velocity = v / radius 
    vx_goal = v
    r_goal = angular_velocity
    return vx_goal, r_goal

def mpc_cost(U, *args):
    N, state, target = args
    cost = 0.0
    vx_goal, r_goal = circle_velocity_target(circle_radius, v_max)

    alpha_vx = 1.0
    alpha_r = 1.0
    alpha_pos = 10.0

    current_state = state.copy()
    
    for i in range(N):
        vx_cmd, delta = U[2*i], U[2*i+1]
        current_state = dynamics_finite(current_state, np.array([vx_cmd, delta]), dt)
        
        x, y, yaw, vx, vy, r = current_state
        
        # Position error
        pos_error = (x - target[0])**2 + (y - target[1])**2
        
        # Velocity and yaw rate tracking
        vel_cost = alpha_vx*(vx - vx_goal)**2 + alpha_r*(r - r_goal)**2
        
        # Heading error
        diff_yaw = yaw - target[2]
        heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2
        
        cost += vel_cost + alpha_pos*pos_error + heading_error
        
    return cost

# MPC parameters
N = 3  # Prediction horizon
state = np.array([0.0, 0, np.pi/2, 0, 0, 0])  # Initial state [x, y, yaw, vx, vy, r]

# Initial guess for controls - now using [vx_cmd, delta]
vx_cmd_initial = 0.0
delta_initial = 0.0
U0 = [vx_cmd_initial, delta_initial] * N

# Constraints for controls
bounds = [(0, v_max), (-steer_max, steer_max)] * N  # vx_cmd between 0 and v_max, delta between -steer_max and steer_max

# Run MPC
trajectory = [state]
controls = []
time = [0]
targets = []

for t in range(650):
    target = circle_target(t * dt, circle_radius, circle_center)
    targets.append(target)

    result = minimize(
        mpc_cost, U0, args=(N, state, target),
        bounds=bounds, method='SLSQP'
    )
    
    if not result.success:
        print("Optimization failed!")
        break

    U_opt = result.x
    control = U_opt[:2]
    
    # Apply control
    state = dynamics_finite(state, control, dt)
    
    # Store results
    trajectory.append(state)
    controls.append(control)
    time.append((t + 1) * dt)

# Convert to numpy arrays
trajectory = np.array(trajectory)
controls = np.array(controls)
targets = np.array(targets)