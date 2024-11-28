import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca
from scipy.optimize import minimize

# Initialize Joystick
pygame.init()
# pygame.joystick.init()
# joystick = pygame.joystick.Joystick(0)
# joystick.init()

# Constants
dt = 0.02
N = 20  # Prediction horizon
m = 2.35  # mass (kg)
L = 0.257  # wheelbase (m)
g = 9.81
b = 0.14328  # CoG to rear axle
a = L - b  # CoG to front axle
G_front = m * g * b / L
G_rear = m * g * a / L
C_x = 116  # longitudinal stiffness
C_alpha = 197  # lateral stiffness
Iz = 0.045  # rotational inertia
mu = 1.31
mu_spin = 0.55

# Constants for MPC
N = 10  # Prediction horizon
R = np.eye(2) * 0.01  # Control input weight
Q = np.diag([10, 10, 1])  # State error weight

# Target for steady-state circular drift
radius = 5.0
omega = 0.5  # Desired angular velocity
Ux_target = radius * omega
target = np.array([0, radius, 0])  # Center of circle (x, y)

# Vehicle state: [x, y, phi, Ux, Uy, r]
x = np.zeros(6)
u = np.array([0.0, 0.0])  # Initial control inputs: [Ux_cmd, delta]

# Initialize visualization
fig, ax = plt.subplots()
ax.axis([-2, 2, -2, 2])
ax.set_aspect('equal')
line, = ax.plot([], [], 'b-')
traj_cog, = ax.plot([], [], 'g-')
traj_r, = ax.plot([], [], 'r-')
heading_arrow, = ax.plot([], [], 'k-', lw=2)  

# Geometry
P = np.array([[-0.15, -0.15, 0.15, 0.15, -0.15],
              [-0.08, 0.08, 0.08, -0.08, -0.08],
              [1, 1, 1, 1, 1]])
W = np.array([[-0.03, -0.03, 0.03, 0.03, -0.03],
              [-0.015, 0.015, 0.015, -0.015, -0.015],
              [1, 1, 1, 1, 1]])
CoG = np.array([[0], [0], [1]])
r_axle = np.array([[-0.15], [0], [1]])

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def tire_dyn(Ux, Ux_cmd, mu, mu_slide, Fz, C_x, C_alpha, alpha):
    # longitude wheel slip
    if Ux_cmd == Ux:
        K = 0
    elif Ux == 0:
        Fx = np.sign(Ux_cmd) * mu * Fz
        Fy = 0
        return Fx, Fy
    else:
        K = (Ux_cmd - Ux) / abs(Ux)

    # instead of avoiding -1, now look for positive equivalent
    reverse = 1
    if K < 0:
        reverse = -1
        K = abs(K)

    # alpha > pi/2 cannot be adapted to this formula
    # because of the use of tan(). Use the equivalent angle instead.
    if abs(alpha) > np.pi / 2:
        alpha = (np.pi - abs(alpha)) * np.sign(alpha)

    gamma = np.sqrt(C_x**2 * (K / (1 + K))**2 + C_alpha**2 * (np.tan(alpha) / (1 + K))**2)
    if gamma <= 3 * mu * Fz:
        F = gamma - (2 - mu_slide / mu) * gamma**2 / (3 * mu * Fz) + \
            (1 - (2 / 3) * (mu_slide / mu)) * gamma**3 / (9 * mu**2 * Fz**2)
    else:
        # more accurate modeling with peak friction value
        F = mu_slide * Fz

    if gamma == 0:
        Fx = Fy = 0
    else:
        Fx = C_x / gamma * (K / (1 + K)) * F * reverse
        Fy = -C_alpha / gamma * (np.tan(alpha) / (1 + K)) * F
    return Fx, Fy

def dynamics(x, u):
    pos_x, pos_y, pos_phi = x[:3]
    Ux, Uy, r = x[3:]
    Ux_cmd, delta = u

    # Tire Dyanmics
    # lateral slip angle alpha
    if Ux == 0 and Uy == 0: # vehicle is still no slip
        alpha_F = alpha_R = 0
    elif Ux == 0: # perfect side slip
        alpha_F = np.pi / 2 * np.sign(Uy) - delta
        alpha_R = np.pi / 2 * np.sign(Uy)
    elif Ux < 0: # rare ken block situations
        alpha_F = np.arctan((Uy + a * r) / abs(Ux)) + delta
        alpha_R = np.arctan((Uy - b * r) / abs(Ux))
    else: # normal situation
        alpha_F = np.arctan((Uy + a * r) / abs(Ux)) - delta
        alpha_R = np.arctan((Uy - b * r) / abs(Ux))

    # safety that keep alpha in valid range
    alpha_F, alpha_R = wrap_to_pi(alpha_F), wrap_to_pi(alpha_R)

    Fxf, Fyf = tire_dyn(Ux, Ux, mu, mu_spin, G_front, C_x, C_alpha, alpha_F)
    Fxr, Fyr = tire_dyn(Ux, Ux_cmd, mu, mu_spin, G_rear, C_x, C_alpha, alpha_R)

    # Vehicle Dynamics
    r_dot = (a * Fyf * np.cos(delta) - b * Fyr) / Iz
    Ux_dot = (Fxr - Fyf * np.sin(delta)) / m + r * Uy
    Uy_dot = (Fyf * np.cos(delta) + Fyr) / m - r * Ux

    U = np.sqrt(Ux**2 + Uy**2)
    if Ux == 0 and Uy == 0:
        beta = 0
    elif Ux == 0:
        beta = np.pi / 2 * np.sign(Uy)
    elif Ux < 0 and Uy == 0:
        beta = np.pi
    elif Ux < 0:
        beta = np.sign(Uy) * np.pi - np.arctan(Uy / abs(Ux))
    else:
        beta = np.arctan(Uy / abs(Ux))
    beta = wrap_to_pi(beta)

    Ux_terrain = U * np.cos(beta + pos_phi)
    Uy_terrain = U * np.sin(beta + pos_phi)
    return np.array([Ux_terrain, Uy_terrain, r, Ux_dot, Uy_dot, r_dot])

def dynamics_finite(x, u, dt):
    k1 = dynamics(x, u)
    return x + dt * k1

def circle_velocity_target(radius, v):
    angular_velocity = v / radius 

    vx_goal = v
    r_goal = angular_velocity

    return vx_goal, r_goal

def mpc_cost(u_flat, x, vx_goal, r_goal, dt, N, weights):
    w1, w2 = weights  # Weights for velocity and yaw rate error
    u = u_flat.reshape(N, -1)  # Reshape control inputs into matrix form
    
    cost = 0.0
    x_k = np.copy(x)  # Current state
    
    for k in range(N):
        u_k = u[k, :]
        x_k = dynamics_finite(x_k, u_k, dt)  # State propagation
        
        vx_k, _, _, _, _, r_k = x_k  # Extract velocity and yaw rate
        w_ss = w1 * (vx_k - vx_goal)**2 + w2 * (r_k - r_goal)**2  # Steady-state cost
        cost += w_ss

    return cost

def constraints(u_flat, x0):
    u = u_flat.reshape(N, 2)
    x = x0.copy()
    x_pred = []
    for i in range(N):
        x = dynamics_finite(x, u[i], dt)
        x_pred.append(x)
    return np.array(x_pred).flatten()

# Bounds for control inputs
bounds = [(-2.5, 2.5)] * N + [(-0.698, 0.698)] * N

# Parameters for optimization
N = 5  # Horizon length
dt = 0.02  # Time step
weights = [1.0, 1.0]  # Weighting factors for cost components
u0 = np.zeros(N * 2)  # Initial guess for control inputs (throttle, steering)

def update_plot_mpc(frame):
    global x, u0  # Use global state and initial guess for control inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Solve optimization problem
    result = minimize(
        mpc_cost,
        u0,
        args=(x, Ux_target, omega, dt, N, weights),
        bounds=bounds,
        method='SLSQP',
    )

    # Extract optimized control inputs
    if result.success:
        u_opt = result.x.reshape(N, 2)
        throttle, steer = u_opt[0]  # Apply the first control input
        u0 = result.x  # Use optimized inputs as the next guess
    else:
        throttle, steer = 0, 0  # Default to zero if optimization fails

    # Update vehicle state
    u = np.array([throttle, steer])
    x[:] = dynamics_finite(x, u, dt)

    # Visualization
    pos_x, pos_y, pos_phi = x[:3]
    A = np.array([[np.cos(pos_phi), -np.sin(pos_phi), pos_x],
                  [np.sin(pos_phi), np.cos(pos_phi), pos_y],
                  [0, 0, 1]])
    pos = A @ P
    CoG_n = A @ CoG
    rear_n = A @ r_axle

    # Update car position and trajectory
    line.set_data(pos[0, :], pos[1, :])
    traj_cog.set_data(np.append(traj_cog.get_xdata(), CoG_n[0, 0]),
                      np.append(traj_cog.get_ydata(), CoG_n[1, 0]))
    traj_r.set_data(np.append(traj_r.get_xdata(), rear_n[0, 0]),
                    np.append(traj_r.get_ydata(), rear_n[1, 0]))

    # Update heading arrow
    arrow_length = 0.3
    arrow_x = [pos_x, pos_x + arrow_length * np.cos(pos_phi)]
    arrow_y = [pos_y, pos_y + arrow_length * np.sin(pos_phi)]
    heading_arrow.set_data(arrow_x, arrow_y)

# Ensure initial state is not zero for meaningful visualization
x = np.array([0.0, 0.0, 0.0, Ux_target, 0.0, omega])

# Ensure initial visualization is correct
line.set_data([], [])
traj_cog.set_data([], [])
traj_r.set_data([], [])
heading_arrow.set_data([], [])

# Start animation
ani = FuncAnimation(fig, update_plot_mpc, interval=dt * 1000)
plt.show()
pygame.quit()
