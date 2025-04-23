import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.animation as animation

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
circle_radius = 1.5
circle_center = np.array([0, 0])
v_max = 2.5  # m/s
steer_max = 0.698  # rad

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def tire_dyn(Ux, Ux_cmd, mu, mu_slide, Fz, C_x, C_y, alpha):
    # Longitudinal wheel slip
    eps = 1e-3
    if abs(Ux_cmd - Ux) < eps:
        K = 0
    elif abs(Ux) < eps:
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

def calculate_slip_angles(vx, vy, r, delta):
    if vx == 0 and vy == 0:
        alpha_F = alpha_R = 0
    elif vx == 0:
        alpha_F = np.pi / 2 * np.sign(vy) - delta
        alpha_R = np.pi / 2 * np.sign(vy)
    elif vx < 0:
        alpha_F = np.arctan((vy + a * r) / abs(vx)) + delta
        alpha_R = np.arctan((vy - b * r) / abs(vx))
    else:
        alpha_F = np.arctan((vy + a * r) / abs(vx)) - delta
        alpha_R = np.arctan((vy - b * r) / abs(vx))
    return alpha_F, alpha_R

def calculate_tire_forces(alpha_f, alpha_r):
    Fxf, Fyf = tire_dyn(0, 0, mu, mu_spin, G_front, C_x, C_y, alpha_f)  # Using dummy vx values since we'll recalculate
    Fxr, Fyr = tire_dyn(0, 0, mu, mu_spin, G_rear, C_x, C_y, alpha_r)
    return Fyf, Fyr

    
def dynamics(x, u):
    pos_x, pos_y, pos_phi = x[:3]
    Ux, Uy, r = x[3:]
    Ux_cmd, delta = u

    # Clip steering angle to physical limits
    delta = np.clip(delta, -steer_max, steer_max)
    
    # Tire Dynamics - lateral slip angle alpha (original calculation preserved)
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

    # Keep alpha in valid range (original approach preserved)
    alpha_F, alpha_R = wrap_to_pi(alpha_F), wrap_to_pi(alpha_R)

    # Tire forces (modified to use Ux_cmd for front wheels too)
    Fxf, Fyf = tire_dyn(Ux, Ux_cmd, mu, mu_spin, G_front, C_x, C_y, alpha_F)
    Fxr, Fyr = tire_dyn(Ux, Ux_cmd, mu, mu_spin, G_rear, C_x, C_y, alpha_R)

    # Vehicle Dynamics (original calculation preserved with sign fix)
    r_dot = (a * Fyf * np.cos(delta) - b * Fyr) / Iz
    Ux_dot = (Fxr + Fyf * np.sin(delta)) / m + r * Uy  # Fixed sign here
    Uy_dot = (Fyf * np.cos(delta) + Fyr) / m - r * Ux

    # Position update (original beta calculation preserved)
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

    # Modified position update to properly handle orientation
    pos_x_dot = Ux * np.cos(pos_phi) - Uy * np.sin(pos_phi)
    pos_y_dot = Ux * np.sin(pos_phi) + Uy * np.cos(pos_phi)
    
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
        
        # # Position error
        # pos_error = (x - target[0])**2 + (y - target[1])**2
        
        # # Velocity and yaw rate tracking
        # vel_cost = alpha_vx*(vx - vx_goal)**2 + alpha_r*(r - r_goal)**2
        
        # # Heading error
        # diff_yaw = yaw - target[2]
        # heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2
        
        # cost += vel_cost + alpha_pos*pos_error + heading_error
        # print(cost)

        ## แก้ cost

    return cost

# MPC parameters
N = 3 #10  # Prediction horizon
state = np.array([0.0, 0, np.pi/2, 0, 0, 0])  # Initial state [x, y, yaw, vx, vy, r]

# Initial guess for controls
vx_cmd_initial = 1.0
delta_initial = 0.0
U0 = [vx_cmd_initial, delta_initial] * N

# Constraints for controls
bounds = [(0, v_max), (-steer_max, steer_max)] * N

# Run MPC
trajectory = [state]
controls = []
time = [0]
targets = []
costs = []

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

    # Store the optimal cost
    costs.append(result.fun)

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

########################### Visualization ##########################################

# Static plot
plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory', linewidth=2)
plt.plot(targets[:, 0], targets[:, 1], '--', label='Target Circle', linewidth=1.5)

# Plot yaw direction
for i in range(0, len(trajectory), 10):
    x, y, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    plt.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw' if i == 0 else "")

plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('MPC - Circular Trajectory with Yaw Visualization')
plt.grid()
plt.axis('equal')
plt.savefig("mpc_trajectory_plot.png")
plt.show()

# Animation
########################### Visualize in animation ##########################################

import matplotlib.animation as animation

# Setup for animation
fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure for better text display

# Initialize plot elements
trajectory_line, = ax.plot([], [], 'b-', lw=2, label='Robot Trajectory')
target_circle_line, = ax.plot([], [], 'g--', label='Target Circle')  
car_marker, = ax.plot([], [], 'ro', label='Car Position')
yaw_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw')

# Add text elements for displaying variables
text_x = 0.02  # X position of text (relative to axes)
text_y_start = 0.95  # Starting Y position of text (relative to axes)
text_spacing = 0.05  # Vertical spacing between text elements

# Create text elements for all the variables we want to display
vx_text = ax.text(text_x, text_y_start, 'vx: 0', transform=ax.transAxes, fontsize=10, fontweight='bold')
vy_text = ax.text(text_x, text_y_start - text_spacing, 'vy: 0', transform=ax.transAxes, fontsize=10, fontweight='bold')
r_text = ax.text(text_x, text_y_start - 2*text_spacing, 'r: 0', transform=ax.transAxes, fontsize=10)
delta_text = ax.text(text_x, text_y_start - 3*text_spacing, 'delta: 0', transform=ax.transAxes, fontsize=10)
Fx_text = ax.text(text_x, text_y_start - 4*text_spacing, 'Fx: 0', transform=ax.transAxes, fontsize=10)
Delta_delta_text = ax.text(text_x, text_y_start - 5*text_spacing, 'Δdelta: 0', transform=ax.transAxes, fontsize=10)
alpha_f_text = ax.text(text_x, text_y_start - 6*text_spacing, 'α_f: 0', transform=ax.transAxes, fontsize=10)
alpha_r_text = ax.text(text_x, text_y_start - 7*text_spacing, 'α_r: 0', transform=ax.transAxes, fontsize=10)
Fyf_text = ax.text(text_x, text_y_start - 8*text_spacing, 'Fyf: 0', transform=ax.transAxes, fontsize=10)
Fyr_text = ax.text(text_x, text_y_start - 9*text_spacing, 'Fyr: 0', transform=ax.transAxes, fontsize=10)
cost_text = ax.text(text_x, text_y_start - 10*text_spacing, 'Cost: 0', transform=ax.transAxes, fontsize=10, fontweight='bold')

ax.set_title("MPC - Circular Trajectory Animation")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid()
ax.axis('equal')

# Update function for animation
def update(frame):
    global yaw_arrow

    # Clear the previous arrow if it exists
    if 'yaw_arrow' in globals() and yaw_arrow:
        yaw_arrow.remove()

    # Update trajectory
    trajectory_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])

    # Update car position
    car_marker.set_data(trajectory[frame, 0], trajectory[frame, 1])

    # Update yaw arrow
    x, y, yaw = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    yaw_arrow = ax.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

    # Update target circle
    target_circle_line.set_data(targets[:, 0], targets[:, 1])

    # Get current state and control values
    current_state = trajectory[frame]
    vx, vy, r = current_state[3], current_state[4], current_state[5]
    
    if frame < len(controls):
        current_control = controls[frame]
        delta = current_control[1]  # delta is the second element in control pair
    else:
        current_control = [0, 0]
        delta = 0
    
    # Calculate slip angles and tire forces
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)
    
    # Get the stored cost
    current_cost = costs[frame] if frame < len(costs) else 0

    # Update all text elements
    vx_text.set_text(f'vx: {vx:.2f} m/s')
    vy_text.set_text(f'vy: {vy:.2f} m/s')
    r_text.set_text(f'r: {r:.2f} rad/s')
    delta_text.set_text(f'delta: {delta:.2f} rad')
    Fx_text.set_text(f'Fx cmd: {current_control[0]:.2f} m/s')
    Delta_delta_text.set_text(f'delta cmd: {delta:.2f} rad')
    alpha_f_text.set_text(f'α_f: {alpha_f:.2f} rad')
    alpha_r_text.set_text(f'α_r: {alpha_r:.2f} rad')
    Fyf_text.set_text(f'Fyf: {Fyf:.2f} N')
    Fyr_text.set_text(f'Fyr: {Fyr:.2f} N')
    cost_text.set_text(f'Cost: {current_cost:.6f}')

    # Dynamically adjust plot limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    buffer = 2.0  # Space around the car for visibility

    if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
        ax.set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
        ax.set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)

    return (trajectory_line, car_marker, yaw_arrow, vx_text, vy_text, r_text, delta_text,
            Fx_text, Delta_delta_text, alpha_f_text, alpha_r_text, Fyf_text, Fyr_text,
            cost_text, target_circle_line)

# Adjust legend position to avoid overlap with the text
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory), interval=50, blit=False
)

plt.tight_layout()
plt.show()
