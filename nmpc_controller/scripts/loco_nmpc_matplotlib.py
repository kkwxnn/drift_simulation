#################### offline mpc #####################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.optimize as opt
import time

# Vehicle parameters 
# Constants
dt = 0.02
m = 2.35
L = 0.257
g = 9.81
b = 0.14328
a = L - b
G_front = m * g * b / L
G_rear = m * g * a / L
C_x = 116 # tire longitudinal stiffness
C_y = 197 # tire lateral stiffness
Iz = 0.045
mu = 1.31 # mu_peak
mu_spin = 0.55 # spinning coefficient

circle_radius = 5.0
v_max = 2.5 # m/s
steer_max = 0.698 # rad

# Initialize state (x, y, theta, Ux, Uy, r)
x = np.array([0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

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
    pos_x, pos_y, pos_phi = x[:3]
    Ux, Uy, r = x[3:]
    Ux_cmd, delta = u

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

    Fxf, Fyf = tire_dyn(Ux, Ux, mu, mu_spin, G_front, C_x, C_y, alpha_F)
    Fxr, Fyr = tire_dyn(Ux, Ux_cmd, mu, mu_spin, G_rear, C_x, C_y, alpha_R)

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

def circle_velocity_target(circle_radius, v):
    vx_goal = v
    r_goal = vx_goal / circle_radius
    return vx_goal, r_goal

def cost_function(u, x0, N, dt):
    vx = x0[3]
    if vx >= 0:
        v = v_max
    else:
        v = -v_max
    
    vx_goal, r_goal = circle_velocity_target(circle_radius, v)
    
    # Calculate desired position based on current state
    theta_desired = np.arctan2(x0[1], x0[0]) + r_goal * dt * np.arange(1, N+1)
    x_desired = circle_radius * np.cos(theta_desired)
    y_desired = circle_radius * np.sin(theta_desired)
    
    # Weighting factors
    alpha_vx = 1.0
    alpha_r = 1.0
    alpha_pos = 10.0  # New weight for position error
    
    cost = 0
    state = np.copy(x0)
    
    for i in range(N):
        throttle, steer = u[2*i], u[2*i+1]
        state = dynamics_finite(state, np.array([throttle, steer]), dt)
        x, y, yaw, vx, vy, r = state
        
        # Include position error in cost
        pos_error = (x - x_desired[i])**2 + (y - y_desired[i])**2
        cost += (alpha_vx * (vx - vx_goal)**2 + 
                alpha_r * (r - r_goal)**2 + 
                alpha_pos * pos_error)
    return cost

# MPC setup
N = 3 # prediction horizon
throttle_bound = (-v_max, v_max)  
steer_bound = (-steer_max, steer_max)

# Initialize control sequence
throttle_initial = np.random.uniform(throttle_bound[0], throttle_bound[1], N)
steer_initial = np.random.uniform(steer_bound[0], steer_bound[1], N)
u_initial = np.zeros(2 * N)
u_initial[::2] = throttle_initial
u_initial[1::2] = steer_initial

def mpc_control(x0):
    bounds = [throttle_bound, steer_bound] * N
    result = opt.minimize(cost_function, u_initial, args=(x0, N, dt), method='SLSQP', bounds=bounds)
    if not result.success:
        print("Optimization failed:", result.message)
    return result.x[:2]

# ==============================================
# ==============================================
# Offline Optimization Phase
# ==============================================
print("Running offline optimization...")
start_time = time.time()

# Storage for results
states = [x.copy()]
controls = []
targets = []

# Calculate time to complete one full circle
circumference = 2 * np.pi * circle_radius
time_per_circle = circumference / v_max

for t in range(1000):  # Simulate 500 steps
    # Calculate angle based on time and desired velocity
    theta = (2 * np.pi * t * dt) / time_per_circle
    target = np.array([circle_radius * np.cos(theta), circle_radius * np.sin(theta)])
    targets.append(target)
    
    # Get control from MPC
    u = mpc_control(x)
    controls.append(u)
    
    # Apply dynamics
    x = dynamics_finite(x, u, dt)
    states.append(x.copy())

print(f"Optimization completed in {time.time() - start_time:.2f} seconds")

# Convert to arrays
states = np.array(states)
controls = np.array(controls)
targets = np.array(targets)

# ==============================================
# Visualization Phase
# ==============================================
print("Preparing visualization...")

# First, create a static plot of the trajectory
plt.figure(figsize=(10, 8))
plt.plot(states[:, 0], states[:, 1], 'b-', label='Robot Trajectory', linewidth=2)
plt.plot(targets[:, 0], targets[:, 1], 'g--', label='Target Circle', linewidth=1.5)

# Plot yaw direction arrows every 20 steps
for i in range(0, len(states), 20):
    x, y, yaw = states[i, 0], states[i, 1], states[i, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    plt.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw' if i == 0 else "")

plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.title('MPC - Circular Trajectory with Yaw Visualization')
plt.grid()
plt.axis('equal')
plt.savefig("mpc_trajectory_plot.png")
plt.show()

# Now create the animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("MPC Circular Trajectory Tracking")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Initialize plot elements
trajectory_line, = ax.plot([], [], 'b-', lw=2, label='Robot Trajectory')
target_circle, = ax.plot([], [], 'g--', lw=1.5, label='Target Circle')
car_marker, = ax.plot([], [], 'ro', markersize=8, label='Car Position')
yaw_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='r', ec='r')

# Add text elements for vehicle state
vx_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
vy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
yaw_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
r_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
control_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)

# Add legend
ax.legend(loc='upper right')

def init():
    trajectory_line.set_data([], [])
    target_circle.set_data([], [])
    car_marker.set_data([], [])
    yaw_arrow.set_data(x=0, y=0, dx=0, dy=0)
    vx_text.set_text('')
    vy_text.set_text('')
    yaw_text.set_text('')
    r_text.set_text('')
    control_text.set_text('')
    return trajectory_line, target_circle, car_marker, yaw_arrow, vx_text, vy_text, yaw_text, r_text, control_text

def update(frame):
    # Update trajectory (up to current frame)
    trajectory_line.set_data(states[:frame, 0], states[:frame, 1])
    
    # Update target circle (full circle)
    theta = np.linspace(0, 2*np.pi, 100)
    target_x = circle_radius * np.cos(theta)
    target_y = circle_radius * np.sin(theta)
    target_circle.set_data(target_x, target_y)
    
    # Update car position
    x, y = states[frame, 0], states[frame, 1]
    car_marker.set_data([x], [y])
    
    # Update yaw arrow
    yaw = states[frame, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    
    # Remove old arrow and create new one
    global yaw_arrow
    yaw_arrow.remove()
    yaw_arrow = ax.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r')
    
    # Update text displays
    vx = states[frame, 3]
    vy = states[frame, 4]
    r = states[frame, 5]
    throttle, steer = controls[frame, 0], controls[frame, 1]
    
    vx_text.set_text(f'vx: {vx:.2f} m/s')
    vy_text.set_text(f'vy: {vy:.2f} m/s')
    yaw_text.set_text(f'yaw: {np.degrees(yaw):.1f}°')
    r_text.set_text(f'r: {r:.2f} rad/s')
    control_text.set_text(f'Throttle: {throttle:.2f}\nSteer: {np.degrees(steer):.1f}°')
    
    # Adjust view if needed
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    buffer = 1.0
    
    if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
        ax.set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
        ax.set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)
    
    return trajectory_line, target_circle, car_marker, yaw_arrow, vx_text, vy_text, yaw_text, r_text, control_text

# Create animation
ani = FuncAnimation(
    fig, update, frames=len(states),
    init_func=init, blit=False, interval=50
)

plt.tight_layout()
plt.show()

# Save animation (optional)
# ani.save('mpc_trajectory.mp4', writer='ffmpeg', fps=30)