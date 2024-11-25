import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
m = 1.98  # Mass of the vehicle (kg)
Iz = 0.24  # Moment of inertia around z-axis (kg*m^2)
Lf = 0.125  # Distance from the center of gravity to the front axle (m)
Lr = 0.125  # Distance from the center of gravity to the rear axle (m)
# Bf, Cf, Df = 7.4, 1.2, -2.27
# Br, Cr, Dr = 7.4, 1.2, -2.27
Bf, Cf, Df = 2.0, 1.2, -1.0
Br, Cr, Dr = 2.0, 1.2, -1.0

# Blending speed thresholds
v_blend_min = 0.1
# v_blend_max = 2.5
v_blend_max = 5.0

# Define parameters
dt = 0.2 #0.1  # time step

circle_radius = 10.0  # Radius of the circle
circle_center = np.array([0, 0])  # Center of the circle

def calculate_lambda(vx, vy):
    phi = v_blend_min + 0.5 * (v_blend_max - v_blend_min)
    w = (2 * np.pi) / (v_blend_max - v_blend_min)
    lambda_val = 0.5 * (np.tanh(w * ((vx**2 + vy**2)**0.5 - phi)) + 1)
    return lambda_val

def calculate_slip_angles(vx, vy, r, delta):
    epsilon = 1e-5
    vx_safe = max(vx, epsilon)  # Ensure vx is not zero to avoid undefined slip angle
    alpha_f = -np.arctan((Lf * r + vy) / vx_safe) + delta
    alpha_r = np.arctan((Lr * r - vy) / vx_safe)

    if vx < epsilon:
        alpha_f = alpha_r = 0  # Both slip angles are zero if vx is very low
    return alpha_f, alpha_r

def calculate_tire_forces(alpha_f, alpha_r):
    Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
    Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
    return Fyf, Fyr

# State equations of the bicycle model
def drift_model(state, control, dt):
    x, y, yaw, vx, vy, r, delta = state
    Fx, Delta_delta = control

    lam = calculate_lambda(vx, vy)
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)
    # Fyf = 0.0
    # Fyr = 0.0

    B = 0.5
    # Dynamic model equations
    x_dot_dyn = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_dyn = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_dyn = r
    vx_dot_dyn = (1 / m) * (-B*vx + Fx - Fyf * np.sin(delta) + m * vy * r)
    vy_dot_dyn = (1 / m) * (B*vy + Fyr + Fyf * np.cos(delta) - m * vx * r)
    r_dot_dyn = (1 / Iz) * (-B*r + Fyf * Lf * np.cos(delta) - Fyr * Lr)
    delta_dot_dyn = Delta_delta

    # Kinematic model equations
    x_dot_kin = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_kin = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_kin = r
    vx_dot_kin = (-B*vx + Fx) / m
    vy_dot_kin = (Delta_delta * vx) * (Lr / (Lr + Lf))
    r_dot_kin = (Delta_delta * vx) * (1 / (Lr + Lf))
    delta_dot_kin = Delta_delta

    # lam = 0 # Kinematics Model Only
    # lam = 1 # Dynamics Model Only
    # Fused Kinematic-Dynamic Bicycle Model
    x_dot = lam * x_dot_dyn + (1 - lam) * x_dot_kin
    y_dot = lam * y_dot_dyn + (1 - lam) * y_dot_kin
    yaw_dot = lam * yaw_dot_dyn + (1 - lam) * yaw_dot_kin
    vx_dot = lam * vx_dot_dyn + (1 - lam) * vx_dot_kin
    vy_dot = lam * vy_dot_dyn + (1 - lam) * vy_dot_kin
    r_dot = lam * r_dot_dyn + (1 - lam) * r_dot_kin
    delta_dot = lam * delta_dot_dyn + (1 - lam) * delta_dot_kin

    # Update states
    x += x_dot * dt
    y += y_dot * dt
    yaw += yaw_dot * dt  
    vx += vx_dot * dt
    vy += vy_dot * dt
    r += r_dot * dt
    delta += delta_dot * dt

    return np.array([x, y, yaw, vx, vy, r, delta])

# Cost function for MPC
def mpc_cost(U, *args):
    N, state, target = args
    cost = 0.0

    # Prediction loop for the horizon N
    for i in range(N):
        # Predict the next state using the drift model
        state = drift_model(state, U[2 * i:2 * i + 2], dt)
        
        # Extract current state variables
        x, y, yaw, vx, vy, r, delta = state
        
        # Extract target state variables
        x_target, y_target, yaw_target = target
        
        # Calculate position error (distance squared)
        position_error = (x - x_target)**2 + (y - y_target)**2
        
        # Calculate heading error (yaw difference squared)
        diff_yaw = yaw - yaw_target
        heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2
        
        # Add regularization terms for control effort (penalize large inputs)
        control_effort = 0.1 * (U[2 * i]**2 + U[2 * i + 1]**2)  # Weight for Fx and Delta delta

        # For the final step (terminal state), include the velocity and heading penalties
        if i == N - 1:
            # Add velocity and yaw rate penalties at the last step
            cost = position_error + heading_error + (vx**2) + (vy**2) + (r**2) + control_effort
            # cost += position_error + heading_error
        # else:
        #     # Otherwise, just penalize position and heading errors plus control effort
        #     cost += position_error + heading_error + control_effort
        #     # cost += position_error + heading_error

    return cost



# Generate circle trajectory
def circle_target(t, radius, center):
    """Generate a target point on a circle."""
    angle = t * 0.1  # Angular velocity (radians per second)
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)

    dx = -radius * np.sin(angle)
    dy = radius * np.cos(angle)
    yaw = np.arctan2(dy, dx)

    return np.array([x, y, yaw])

# MPC parameters
N = 5  # Prediction horizon
state = np.array([10.0, 0, np.pi/2, 0, 0, 0, 0])  # Initial state 

# Initial guess for controls
# U0 = np.zeros(2 * N)
Fx_initial = 0.0  # Constant guess for F_x
Delta_delta_initial = 0.0  # Constant guess for Delta delta

U0 = [Fx_initial, Delta_delta_initial] * N

# Constraints for controls
bounds = [(-5.0, 5.0),                  # Fx bounds
          (-0.698, 0.698)] * N          # Delta delta bounds


# Run MPC
trajectory = [state]
controls = []  # Store control inputs (Fx, Delta delta)
time = [0]  # Time stamps
targets = []  # Store dynamic targets


for t in range(1000):  # 1000
    # Get current target on the circle
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
    # U0 = np.hstack([U_opt[2:], np.zeros(2)])

# Extract trajectory and control inputs
trajectory = np.array(trajectory)
controls = np.array(controls)
targets = np.array(targets)

########################### Visualize in graph ##########################################

# # Plot trajectory, target circle, yaw, and velocity direction
# plt.figure(figsize=(10, 8))
# plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory', linewidth=2)
# plt.plot(targets[:, 0], targets[:, 1], '--', label='Target Circle', linewidth=1.5)

# # Plot yaw direction
# for i in range(0, len(trajectory), 10):  # Plot every 10th step to avoid clutter
#     x, y, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]
#     vx, vy = trajectory[i, 3], trajectory[i, 4]

#     # Yaw direction
#     yaw_dx = 0.5 * np.cos(yaw)  # Scale arrows for better visualization
#     yaw_dy = 0.5 * np.sin(yaw)
#     plt.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw' if i == 0 else "")

    
# plt.xlabel('X position')
# plt.ylabel('Y position')
# plt.legend()
# plt.title('MPC - Circular Trajectory with Yaw Visualization')
# plt.grid()
# plt.axis('equal')
# plt.show()


# # Plot control inputs (delta and acceleration)
# plt.figure(figsize=(10, 5))
# plt.plot(time[:-1], controls[:, 0], label='Fx')
# plt.plot(time[:-1], controls[:, 1], label='Delta delta')
# plt.xlabel('Time [s]')
# plt.ylabel('Control Input')
# plt.legend()
# plt.title('Control Inputs vs Time')
# plt.grid()
# plt.show()

########################### Visualize in animation ##########################################

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
Fx_text = ax.text(0.05, 0.85, 'Fx: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
Fyf_text = ax.text(0.05, 0.80, 'Fyf: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
Fyr_text = ax.text(0.05, 0.75, 'Fyr: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
alpha_f_text = ax.text(0.05, 0.70, 'Alpha_f: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
alpha_r_text = ax.text(0.05, 0.65, 'Alpha_r: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
cost_text = ax.text(0.05, 0.60, 'Cost: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

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
    Fx = controls[frame, 0]
    Delta_delta = controls[frame, 1]
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, trajectory[frame, 5], trajectory[frame, 6])
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)
    cost = mpc_cost(U_opt, N, trajectory[frame], target)

    # Update the text elements
    vx_text.set_text(f'vx: {vx:.2f} m/s')
    vy_text.set_text(f'vy: {vy:.2f} m/s')
    Fx_text.set_text(f'Fx: {Fx:.2f} N')
    Fyf_text.set_text(f'Fyf: {Fyf :.2f} N')
    Fyr_text.set_text(f'Fyr: {Fyr :.2f} N')
    alpha_f_text.set_text(f'Alpha_f: {alpha_f:.2f} rad')
    alpha_r_text.set_text(f'Alpha_r: {alpha_r:.2f} rad')
    cost_text.set_text(f'Cost: {cost:.2f}')

    # Dynamically adjust plot limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    buffer = 1.0  # Space around the car for visibility

    if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
        ax.set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
        ax.set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)

    return trajectory_line, car_marker, yaw_arrow, vx_text, vy_text, Fx_text, Fyf_text, Fyr_text, alpha_f_text, alpha_r_text, cost_text, target_circle_line, target_position_marker

# Add target position marker to show the current target
target_position_marker, = ax.plot([], [], 'go', label='Target Position')

# Adjust legend position to avoid overlap with the text
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory), interval=50, blit=False
)

plt.show()

# import matplotlib.animation as animation

# # Setup for animation
# fig, ax = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})

# # Top subplot: Trajectory visualization
# ax[0].set_title("MPC - Circular Trajectory Animation")
# trajectory_line, = ax[0].plot([], [], 'b-', lw=2, label='Robot Trajectory')
# target_circle, = ax[0].plot(targets[:, 0], targets[:, 1], 'g--', label='Target Circle')
# car_marker, = ax[0].plot([], [], 'ro', label='Car Position')
# yaw_arrow = ax[0].arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw')

# ax[0].set_xlabel("X Position")
# ax[0].set_ylabel("Y Position")
# ax[0].legend()
# ax[0].grid()
# ax[0].axis('equal')

# # Bottom subplot: Heading error
# ax[1].set_title("Heading Error Over Time")
# heading_error_line, = ax[1].plot([], [], 'r-', lw=2, label='Heading Error')
# ax[1].set_xlabel("Time [s]")
# ax[1].set_ylabel("Heading Error [rad]")
# ax[1].grid()
# ax[1].legend()

# # Initialize heading error data
# time_data = []
# heading_error_data = []

# # Update function for animation
# def update(frame):
#     global yaw_arrow

#     # Clear the arrow to avoid duplication
#     if yaw_arrow:
#         yaw_arrow.remove()

#     # Update trajectory
#     trajectory_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])

#     # Update car position
#     car_marker.set_data(trajectory[frame, 0], trajectory[frame, 1])

#     # Update yaw arrow
#     x, y, yaw = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2]
#     yaw_dx = 0.5 * np.cos(yaw)
#     yaw_dy = 0.5 * np.sin(yaw)
#     yaw_arrow = ax[0].arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

#     # Calculate heading error
#     yaw_target = targets[frame, 2]
#     diff_yaw = yaw - yaw_target
#     heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2

#     # Update heading error data
#     time_data.append(time[frame])
#     heading_error_data.append(heading_error)

#     # Update heading error plot
#     heading_error_line.set_data(time_data, heading_error_data)

#     # Dynamically adjust heading error plot limits
#     ax[1].set_xlim(0, time[frame] + 1)
#     ax[1].set_ylim(
#         min(heading_error_data) - 0.1, 
#         max(heading_error_data) + 0.1
#     )

#     # Dynamically adjust trajectory plot limits
#     x_min, x_max = ax[0].get_xlim()
#     y_min, y_max = ax[0].get_ylim()
#     buffer = 1.0  # Space around the car for visibility

#     if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
#         ax[0].set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
#         ax[0].set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)

#     return trajectory_line, car_marker, heading_error_line

# # Create animation
# ani = animation.FuncAnimation(
#     fig, update, frames=len(trajectory),
#     interval=100, blit=False
# )

# plt.tight_layout()
# plt.show()
