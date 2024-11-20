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

# Circular trajectory parameters
R = 5.0  # Radius of the circular trajectory (m)
x_c, y_c = 0.0, 0.0  # Center of the circle
angular_velocity = 0.5  # Rad/s (angular velocity of the vehicle around the circle)

# Time and trajectory
dt = 0.1  # Time step (s)
t_max = 50  # Total time steps for the trajectory
time_steps = np.linspace(0, t_max, int(t_max / dt))

# Initial state variables
x, y = 0.0, 0.0  # Initial position
yaw = 0.0  # Initial heading angle
vx, vy, r, delta = 0.0, 0.0, 0.0, 0.0  # Initialize velocities and steering angle

# Generate the circular trajectory (Goal)
target_x = x_c + R * np.cos(angular_velocity * time_steps)
target_y = y_c + R * np.sin(angular_velocity * time_steps)
target_yaw = np.arctan2(np.gradient(target_y), np.gradient(target_x))

# Actual vehicle path (initially empty)
vehicle_path_x = [x]
vehicle_path_y = [y]

# Combine the target positions with the yaw angles
Goal = np.vstack((target_x, target_y, target_yaw)).T

# Cost function for NMPC
def cost_function(u, x, y, yaw, delta, Goal, t):
    # Control input
    Fx = u[0]  # Force input
    delta_control = u[1]  # steering angle change (Delta_delta)

    # Update state based on current control inputs
    x_next, y_next, yaw_next, vx_next, vy_next, r_next, delta_next = update_state(Fx, delta_control, dt, x, y, yaw, vx, vy, r, delta)

    # Circular path cost (penalizes deviation from the path)
    target_x, target_y, _ = Goal[t]
    distance_to_path = np.sqrt((target_x - x_next)**2 + (target_y - y_next)**2)

    # Steering effort cost (penalizes large steering inputs)
    steering_cost = np.abs(delta_next)
    
    # Combine path following cost, steering cost
    total_cost = distance_to_path + 0.1 * steering_cost
    return total_cost

# Drift model update function
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

def update_state(Fx, Delta_delta, dt, x, y, yaw, vx, vy, r, delta):
    delta += Delta_delta
    lam = calculate_lambda(vx, vy)
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)

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

    return x, y, yaw, vx, vy, r, delta

# MPC Control Loop (simplified for this example)
def mpc_control(x, y, yaw, vx, vy, r, delta, Goal, time_steps, dt):
    # Placeholder for the MPC controller loop
    vehicle_path_x, vehicle_path_y = [x], [y]

    for t in range(1, len(time_steps)):
        # Define the control inputs as variables to optimize
        u0 = np.array([10.0, 0.05])  # Initial guesses 

        # Use optimization to minimize the cost function
        result = minimize(cost_function, u0, args=(x, y, yaw, delta, Goal, t), bounds=[(-np.pi / 4, np.pi / 4), (0, 15)])

        # Extract the optimal control inputs
        delta_control_opt, Fx_opt = result.x
        
        # Update the state using the optimal control inputs
        x, y, yaw, vx, vy, r, delta = update_state(Fx_opt, delta_control_opt, dt, x, y, yaw, vx, vy, r, delta)

        # Log the vehicle path
        vehicle_path_x.append(x)
        vehicle_path_y.append(y)

    return vehicle_path_x, vehicle_path_y

# Run the MPC and plot the results
vehicle_path_x, vehicle_path_y = mpc_control(x, y, yaw, vx, vy, r, delta, Goal, time_steps, dt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(vehicle_path_x, vehicle_path_y, label="MPC Path", color="blue")
plt.plot(target_x, target_y, label="Target Path (Circle)", color="red", linestyle="--")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("MPC Tracking on Circular Path")
plt.legend()
plt.grid()
plt.show()
