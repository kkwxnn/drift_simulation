import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
m = 1.98  # Mass of the vehicle (kg)
Iz = 0.24  # Moment of inertia around z-axis (kg*m^2)
Lf = 0.125  # Distance from the center of gravity to the front axle (m)
Lr = 0.125  # Distance from the center of gravity to the rear axle (m)
Bf, Cf, Df = 7.4, 1.2, -2.27
Br, Cr, Dr = 7.4, 1.2, -2.27

# Blending speed thresholds
v_blend_min = 0.1
v_blend_max = 2.5

# Define parameters
dt = 0.1  # time step

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

    # Dynamic model equations
    x_dot_dyn = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_dyn = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_dyn = r
    vx_dot_dyn = (1 / m) * (Fx - Fyf * np.sin(delta) + m * vy * r)
    vy_dot_dyn = (1 / m) * (Fyr + Fyf * np.cos(delta) - m * vx * r)
    r_dot_dyn = (1 / Iz) * (Fyf * Lf * np.cos(delta) - Fyr * Lr)
    delta_dot_dyn = Delta_delta

    # Kinematic model equations
    x_dot_kin = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_kin = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_kin = r
    vx_dot_kin = Fx / m
    vy_dot_kin = (Delta_delta * vx) * (Lr / (Lr + Lf))
    r_dot_kin = (Delta_delta * vx) * (1 / (Lr + Lf))
    delta_dot_kin = Delta_delta

    lam = 0 # Kinematics Model Only
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
        heading_error = (yaw - yaw_target)**2
        
        # Add regularization terms for control effort (penalize large inputs)
        control_effort = 0.1 * (U[2 * i]**2 + U[2 * i + 1]**2)  # Weight for Fx and Delta delta

        # For the final step (terminal state), include the velocity and heading penalties
        if i == N - 1:
            # Add velocity and yaw rate penalties at the last step
            cost += position_error + heading_error + (vx**2) + (vy**2) + (r**2) + control_effort
        else:
            # Otherwise, just penalize position and heading errors plus control effort
            cost += position_error + heading_error + control_effort

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
state = np.array([0, 0, 0, 0, 0, 0, 0])  # Initial state 

# Initial guess for controls
U0 = np.zeros(2 * N)

# Constraints for controls
# bounds = [(-5.0, 5.0),                  # Fx bounds
#           (-0.698, 0.698)] * N          # Delta delta bounds
bounds = [(-5.0, 5.0), (-1.5, 1.5)] * N


# Run MPC
trajectory = [state]
controls = []  # Store control inputs (Fx, Delta delta)
time = [0]  # Time stamps
targets = []  # Store dynamic targets

for t in range(300):  # 1000
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
    U0 = np.hstack([U_opt[2:], np.zeros(2)])

# Extract trajectory and control inputs
trajectory = np.array(trajectory)
controls = np.array(controls)
targets = np.array(targets)

# # Plot trajectory and target circle
# plt.figure(figsize=(8, 6))
# plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory')
# plt.plot(targets[:, 0], targets[:, 1], '--', label='Target Circle')
# plt.xlabel('X position')
# plt.ylabel('Y position')
# plt.legend()
# plt.title('MPC - Circular Trajectory')
# plt.grid()
# plt.axis('equal')
# plt.show()

# Plot trajectory, target circle, yaw, and velocity direction
plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory', linewidth=2)
plt.plot(targets[:, 0], targets[:, 1], '--', label='Target Circle', linewidth=1.5)

# Plot yaw direction
for i in range(0, len(trajectory), 10):  # Plot every 10th step to avoid clutter
    x, y, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]
    vx, vy = trajectory[i, 3], trajectory[i, 4]

    # Yaw direction
    yaw_dx = 0.5 * np.cos(yaw)  # Scale arrows for better visualization
    yaw_dy = 0.5 * np.sin(yaw)
    plt.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw' if i == 0 else "")

    
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('MPC - Circular Trajectory with Yaw Visualization')
plt.grid()
plt.axis('equal')
plt.show()


# Plot control inputs (delta and acceleration)
plt.figure(figsize=(10, 5))
plt.plot(time[:-1], controls[:, 0], label='Fx')
plt.plot(time[:-1], controls[:, 1], label='Delta delta')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Inputs vs Time')
plt.grid()
plt.show()
