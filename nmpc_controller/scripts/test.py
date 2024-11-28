import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pygame
import matplotlib.animation as animation

# Constants for both models
dt = 0.1  # time step

circle_radius = 10.0  # Radius of the circle
circle_center = np.array([0, 0])  # Center of the circle

# Define constants for the vehicle parameters
m = 2.35  # mass (kg)
L = 0.257  # wheelbase (m)
g = 9.81
b = 0.14328  # CoG to rear axle
a = L - b  # CoG to front axle
G_front = m * g * b / L  # calculated load
G_rear = m * g * a / L  # calculated load
C_x = 116  # longitudinal stiffness
C_alpha = 197  # lateral stiffness
Iz = 0.045  # rotational inertia
mu = 1.31  # friction coefficient
mu_spin = 0.55  # spin friction coefficient

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def tire_dyn(Ux, Ux_cmd, mu, mu_slide, Fz, C_x, C_alpha, alpha):
    # Tire dynamics (same as before)
    if Ux_cmd == Ux:
        K = 0
    elif Ux == 0:
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

    # Calculate gamma
    gamma = np.sqrt(C_x**2 * (K / (1 + K))**2 + C_alpha**2 * (np.tan(alpha) / (1 + K))**2)
    
    # Add a small epsilon to avoid divide by zero
    epsilon = 1e-6
    gamma = max(gamma, epsilon)

    if gamma <= 3 * mu * Fz:
        F = gamma - (2 - mu_slide / mu) * gamma**2 / (3 * mu * Fz) + \
            (1 - (2 / 3) * (mu_slide / mu)) * gamma**3 / (9 * mu**2 * Fz**2)
    else:
        F = mu_slide * Fz

    Fx = C_x / gamma * (K / (1 + K)) * F * reverse
    Fy = -C_alpha / gamma * (np.tan(alpha) / (1 + K)) * F
    return Fx, Fy

def drift_model(state, control, dt):
    x, y, yaw, vx, vy, r = state
    vx_cmd, delta = control

    # lateral slip angle alpha
    if vx == 0 and vy == 0: # vehicle is still no slip
        alpha_f = alpha_r = 0
    elif vx == 0: # perfect side slip
        alpha_f = np.pi / 2 * np.sign(vy) - delta
        alpha_r = np.pi / 2 * np.sign(vy)
    elif vx < 0: # rare ken block situations
        alpha_f = np.arctan((vy + a * r) / abs(vx)) + delta
        alpha_r = np.arctan((vy - b * r) / abs(vx))
    else: # normal situation
        alpha_f = np.arctan((vy + a * r) / abs(vx)) - delta
        alpha_r = np.arctan((vy - b * r) / abs(vx))

    # Calculate slip angles
    alpha_f, alpha_r = wrap_to_pi(alpha_f), wrap_to_pi(alpha_r)

    # Calculate tire forces
    Fyf, Fyr = tire_dyn(vx, vx, mu, mu_spin, G_front, C_x, C_alpha, alpha_f)
    Fxr, Fyr = tire_dyn(vx, vx_cmd, mu, mu_spin, G_rear, C_x, C_alpha, alpha_r)
    
    U = np.sqrt(vx**2 + vy**2)
    if vx == 0 and vy == 0:
        beta = 0
    elif vx == 0:
        beta = np.pi / 2 * np.sign(vy)
    elif vx < 0 and vy == 0:
        beta = np.pi
    elif vx < 0:
        beta = np.sign(vy) * np.pi - np.arctan(vy / abs(vx))
    else:
        beta = np.arctan(vy / abs(vx))
    beta = wrap_to_pi(beta)

    x_dot = U * np.cos(beta + yaw)
    y_dot = U * np.sin(beta + yaw)
    yaw_dot = r
    vx_dot = (Fxr - Fyf * np.sin(delta)) / m + r * vy
    vy_dot = (Fyf * np.cos(delta) + Fyr) / m - r * vx
    r_dot = (a * Fyf * np.cos(delta) - b * Fyr) / Iz

    x += x_dot * dt
    y += y_dot * dt
    yaw += yaw_dot * dt  
    vx += vx_dot * dt
    vy += vy_dot * dt
    r += r_dot * dt

    return np.array([x, y, yaw, vx, vy, r])


def mpc_cost(U, *args):
    N, state, target = args
    cost = 0.0

    vx_goal, r_goal = circle_velocity_target(circle_radius, 2.5) # 2.5 is v_max

    alpha_vx = 1.0
    alpha_r = 1.0

    for i in range(N):
        state = drift_model(state, U[2*i:2*i+2], dt)
        
        x, y, yaw, vx, vy, r = state

        w_ss = alpha_vx*(vx - vx_goal)**2 + alpha_r*(r - r_goal)**2
        cost += w_ss

    return cost

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

# MPC parameters
N = 5  # Prediction horizon
state = np.array([10, 0, np.pi/2, 0, 0, 0])  # Initial state (x, y, yaw, vx, vy, r)

# Initial guess for controls
U0 = np.zeros(2 * N)

# Constraints for controls
bounds = [(-2.5, 2.5),                  # Ux_cmd bounds
          (-0.698, 0.698)] * N          # steer bounds

# Run MPC
trajectory = [state]
controls = []  # Store control inputs (Ux_cmd, delta)
time = [0]  # Time stamps
targets = []  # Store dynamic targets

for t in range(500):  # Simulate for 300 time steps
    target = circle_target(t * dt, circle_radius, circle_center)
    targets.append(target)

    result = minimize(
        mpc_cost, U0, args=(N, state, target),
        bounds=bounds, method='SLSQP'
    )
    if not result.success:
        print("Optimization failed!")
        break

    # Apply the first control input
    U_opt = result.x
    control = U_opt[:2]
    state = drift_model(state, control, dt)
    trajectory.append(state)
    controls.append(control)
    time.append((t + 1) * dt)

    # Shift the predicted controls
    U0 = np.hstack([U_opt[2:], np.zeros(2)])

# Extract trajectory and control inputs
trajectory = np.array(trajectory)
controls = np.array(controls)
targets = np.array(targets)

import matplotlib.animation as animation

# Setup for animation
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize plot elements
trajectory_line, = ax.plot([], [], 'b-', lw=2, label='Robot Trajectory')
target_circle_line, = ax.plot([], [], 'g--', label='Target Circle')  
car_marker, = ax.plot([], [], 'ro', label='Car Position')
yaw_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw')

# Add additional text elements for display
vx_text = ax.text(0.05, 0.95, 'vx: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
vy_text = ax.text(0.05, 0.90, 'vy: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

ax.set_title("MPC - Circular Trajectory Animation")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid()
ax.axis('equal')

# Update function for animation
def update(frame):
    global yaw_arrow

    # Clear the previous arrow to avoid duplication
    if yaw_arrow:
        yaw_arrow.remove()

    # Update trajectory
    trajectory_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])

    # Update car position
    car_marker.set_data(trajectory[frame, 0], trajectory[frame, 1])

    # Update yaw arrow
    x, y, yaw = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    yaw_arrow = ax.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

    # Update target circle
    target = targets[frame]
    target_circle_line.set_data(targets[:, 0], targets[:, 1])  # Keep target circle visible

    # Update target position marker at the current frame
    target_position_marker.set_data(target[0], target[1])

    # Get the relevant values for display
    vx = trajectory[frame, 3]
    vy = trajectory[frame, 4]

    # Update the text elements
    vx_text.set_text(f'vx: {vx:.2f} m/s')
    vy_text.set_text(f'vy: {vy:.2f} m/s')


    # Dynamically adjust plot limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    buffer = 1.0  # Space around the car for visibility

    if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
        ax.set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
        ax.set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)

    return trajectory_line, car_marker, yaw_arrow, vx_text, vy_text, target_circle_line, target_position_marker

# Add target position marker to show the current target
target_position_marker, = ax.plot([], [], 'go', label='Target Position')

# Adjust legend position to avoid overlap with the text
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory), interval=50, blit=False
)

plt.show()
