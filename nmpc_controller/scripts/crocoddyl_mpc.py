import crocoddyl
import numpy as np

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
R = 5.0  # Radius of the circular trajectory (meters)
x0, y0 = 0.0, 0.0  # Center of the circle
omega = 0.1  # Angular velocity to control the speed along the trajectory (radians per second)

# Define the circular trajectory over time
def circular_trajectory(t, R, x0, y0, omega):
    # Calculate the angle based on time
    theta = omega * t  # Theta increases with time to make the robot follow the circular path
    # Calculate the position (x, y) and the yaw
    x_goal = x0 + R * np.cos(theta)
    y_goal = y0 + R * np.sin(theta)
    yaw_goal = theta  # Yaw is equal to the angle for a circular trajectory
    return x_goal, y_goal, yaw_goal

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

class DriftModel:
    def __init__(self):
        # Define state dimension (7: x, y, yaw, vx, vy, r, delta)
        self.state_dim = 7
        # Define control dimension (2: Fx, Delta_delta)
        self.control_dim = 2
        
        # Create the state and control bounds
        self.lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self.upper_bound = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    
    def state_update(self, state, control, dt):
        x, y, yaw, vx, vy, r, delta = state
        Fx, Delta_delta = control

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

        # Update state
        next_state = state + np.array([x_dot, y_dot, yaw_dot, vx_dot, vy_dot, r_dot, delta_dot]) * dt
        return next_state

# Create the drift model
drift_model = DriftModel()

# Define the state, control, and cost function in Crocoddyl
x_init = np.zeros(7)
u_init = np.zeros(2)
dt = 0.1  # Time step

# Initialize the state and control vectors
state = np.zeros(7)
control = np.zeros(2)

# Set up the time and the goal trajectory
t = 0
Goal = [circular_trajectory(t, R, x0, y0, omega)]

# Define the cost function for the controller
def cost_function(u, state, Goal, t):
    Fx = u[0]  # Force input
    Delta_delta = u[1]  # Steering angle change (Delta_delta)

    # Update state based on current control inputs
    state_next = drift_model.state_update(state, u, dt)

    # Get goal values
    x_goal, y_goal, yaw_goal = Goal[t]

    x_next, y_next, yaw_next, vx_next, vy_next, r_next = state_next[:6]

    vg = np.sqrt((x_next - x_goal)**2 + (y_next - y_goal)**2)
    rg = np.arctan2((y_goal - y_next), (x_goal - x_next))

    alpha_vx = 1.0
    alpha_r = 1.0

    w = alpha_vx * (vx_next - vg)**2 + alpha_r * (r_next - rg)**2
    distance_error = ((x_goal - x_next)**2 + (y_goal - y_next)**2)
    yaw_error = (yaw_goal - yaw_next)**2

    total_cost = w + distance_error + yaw_error + (vx_next**2) + (vy_next**2) + (r_next**2)
    
    return total_cost

