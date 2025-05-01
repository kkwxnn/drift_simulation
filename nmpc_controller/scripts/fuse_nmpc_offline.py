import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error
import time 

# Constants
m = 2.35 # 4.78  # Mass of the vehicle (kg)
Iz = 0.045 # 0.0665  # Moment of inertia around z-axis (kg*m^2)
Lf = 0.11372 # 0.18 # 0.125  # Distance from the center of gravity to the front axle (m)
Lr = 0.14328 # 0.18 # 0.125  # Distance from the center of gravity to the rear axle (m)
# Bf, Cf, Df = 7.4, 1.2, -2.27
# Br, Cr, Dr = 7.4, 1.2, -2.27
Bf, Cf, Df = 2.0, 1.2, -1.0
Br, Cr, Dr = 2.0, 1.2, -1.0

# Blending speed thresholds
v_blend_min = 0.1
v_blend_max = 2.5
# v_blend_max = 5.0
v_max = 2.5 # m/s
steer_max = 0.698 # rad

# Define parameters
dt = 0.1 # 0.02  # time step

circle_radius = 1.5 # 10.0  # Radius of the circle
circle_center = np.array([0, 0])  # Center of the circle

# Generate circle trajectory
def circle_target(t, radius, center):
    angle = t * 0.1  # This will make the angle grow over time
    angle = angle % (2 * np.pi)  # Ensure it loops over a full circle
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

def drift_model(state, control, dt):
    x, y, yaw, vx, vy, r, delta = state
    Fx, Delta_delta = control

    lam = calculate_lambda(vx, vy)
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)
    # Fyf = 0.0
    # Fyr = 0.0

    B = 0.1
    # Dynamic model equations
    x_dot_dyn = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_dyn = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_dyn = r
    vx_dot_dyn = (1 / m) * (-B*vx + Fx - Fyf * np.sin(delta) + m * vy * r)
    vy_dot_dyn = (1 / m) * (-B*vy + Fyr + Fyf * np.cos(delta) - m * vx * r)
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

    lam = 0.05 # Kinematics Model Only
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

    x, y, yaw, vx, vy, r, delta = state

    if vx >= 0:
        v = v_max
    elif vx < 0:
        v = -v_max
    
    vx_goal, r_goal = circle_velocity_target(circle_radius, v)

    alpha_vx = 1.0
    alpha_r = 1.0

    # Prediction loop for the horizon N
    for i in range(N):
        # Predict the next state using the drift model
        state = drift_model(state, U[2 * i:2 * i + 2], dt) # subscribe vehicle state from gazebo !!
        
        x, y, yaw, vx, vy, r, delta = state

        w_ss = alpha_vx*(vx - vx_goal)**2 + alpha_r*(r - r_goal)**2 # mpc circular cost function
        cost += w_ss
        
    # Transient Drift Parking
    x_target, y_target, yaw_target = target
    position_error = (x - x_target)**2 + (y - y_target)**2
    diff_yaw = yaw - yaw_target
    heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2

    # Cost for Parking
    J_park = position_error + heading_error + (vx**2) + (vy**2) + (r**2)
    # cost += J_park

    print(cost)
    return cost


# MPC parameters
N = 3 # 10 # 3  # Prediction horizon

state = np.array([circle_radius, 0, np.pi/2, 0, 0, 0, 0]) 

# Initial guess for controls

# Random Initial guess
# Fx_bound = (-v_max, v_max)  
# Delta_delta_bound = (-steer_max, steer_max)

# Fx_initial = np.random.uniform(Fx_bound[0], Fx_bound[1], N)
# Delta_delta_initial = np.random.uniform(Delta_delta_bound[0], Delta_delta_bound[1], N)

# U0 = np.zeros(2 * N) 
# U0[::2] = Fx_initial 
# U0[1::2] = Delta_delta_initial  

# Zeros Initial guess
Fx_initial = 0.0  # Constant guess for Fx
Delta_delta_initial = 0.0  # Constant guess for Delta delta

U0 = [Fx_initial, Delta_delta_initial] * N

# Constraints for controls
bounds = [(-5.0, 5.0),                  # Fx bounds
          (-0.698, 0.698)] * N          # Delta delta bounds

# Run MPC
trajectory = [state]
controls = []  # Store control inputs (Fx, Delta delta)
time_list = [0]  # Time stamps
targets = []  # Store dynamic targets
costs = []
alpha_f_list = []
alpha_r_list = []
runtime_list = []

for t in range(200):  # 650
    # Get current target on the circle
    target = circle_target(t * dt, circle_radius, circle_center)
    targets.append(target)

    start_time = time.time()  # Start timing

    result = minimize(
        mpc_cost, U0, args=(N, state, target),
        bounds=bounds, method='SLSQP'
    )

    runtime_list.append(time.time() - start_time)  # Store iteration runtime

    if not result.success:
        print("Optimization failed!")
        break
    
    # Store the optimal cost
    costs.append(result.fun)

    # Apply the first control input
    U_opt = result.x
    control = U_opt[:2]
    
    state = drift_model(state, control, dt)
    trajectory.append(state)
    controls.append(control)
    time_list.append((t + 1) * dt)

    vx, vy, r, delta = state[3], state[4], state[5], state[6]
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)
    alpha_f_list.append(alpha_f)
    alpha_r_list.append(alpha_r)

    # Shift the predicted controls
    # U0 = np.hstack([U_opt[2:], np.zeros(2)])

# Extract trajectory and control inputs
trajectory = np.array(trajectory)
controls = np.array(controls)
targets = np.array(targets)

vx_goal_list, r_goal_list = zip(*[circle_velocity_target(circle_radius, 2.5) for _ in trajectory[:, 3]])
vx_goal_list = np.array(vx_goal_list)
r_goal_list = np.array(r_goal_list)

########################### Visualize Trajectory in graph ##########################################

# Plot trajectory, target circle, yaw, and velocity direction
plt.figure(figsize=(10, 8))

# Plot robot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Robot Trajectory', linewidth=2)

# Plot target circle (green dashed line)
theta = np.linspace(0, 2 * np.pi, 100)
target_x = circle_center[0] + circle_radius * np.cos(theta)  # Use scalar value of target center
target_y = circle_center[1] + circle_radius * np.sin(theta)  # Use scalar value of target center
plt.plot(target_x, target_y, 'g--', label='Target Circle', linewidth=1.5)

# Plot yaw direction (velocity direction as arrows)
for i in range(0, len(trajectory), 10):  # Plot every 10th step to avoid clutter
    x, y, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]
    vx, vy = trajectory[i, 3], trajectory[i, 4]

    # Yaw direction (scaled for better visualization)
    yaw_dx = 0.5 * np.cos(yaw)  # Scale arrows for better visualization
    yaw_dy = 0.5 * np.sin(yaw)

    # Plot the yaw direction as arrows
    plt.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r', label='Yaw' if i == 0 else "")

# Compute RMSE for xy trajectory vs target circle (position)
rmse_xy = np.sqrt(mean_squared_error(targets[:, 0:2], trajectory[:len(targets), 0:2]))

# Compute RMSE for r and rg (yaw rate)
rmse_r = np.sqrt(mean_squared_error(r_goal_list, trajectory[:, 5]))

# Compute RMSE for vx and vxg (longitudinal velocity)
rmse_vx = np.sqrt(mean_squared_error(vx_goal_list, trajectory[:, 3]))

# Add RMSE text on the plot (in axes coordinates)
plt.text(
    0.80, 0.10,  # Adjusted y position to avoid overlap with previous text
    f'RMSE Position = {rmse_xy:.4f} m\nRMSE r (yaw rate) = {rmse_r:.4f} rad/s\nRMSE vx = {rmse_vx:.4f} m/s',
    transform=plt.gca().transAxes,  # Use current Axes for coordinate transform
    color='red',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('MPC - Circular Trajectory with Yaw Visualization')
plt.grid()
plt.axis('equal')
plt.savefig(f"Model1_r_{circle_radius}_N_{N}_dt_{dt}_trajectory.png")
print("Save Trajectory!")
plt.show()

########################### Visualize in animation ##########################################

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

    # Clear the previous arrow to avoid duplication
    if yaw_arrow:
        yaw_arrow.remove()

    # Update trajectory
    trajectory_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])

    # Update car position
    car_marker.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])

    # Update yaw arrow
    x, y, yaw = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2]
    yaw_dx = 0.5 * np.cos(yaw)
    yaw_dy = 0.5 * np.sin(yaw)
    yaw_arrow = ax.arrow(x, y, yaw_dx, yaw_dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

    # Update target circle
    theta = np.linspace(0, 2 * np.pi, 100)
    target_x = circle_center[0] + circle_radius * np.cos(theta)
    target_y = circle_center[1] + circle_radius * np.sin(theta)
    target_circle_line.set_data(target_x, target_y)

    # Update text elements with current state information
    vx = trajectory[frame, 3]
    vy = trajectory[frame, 4]
    r = trajectory[frame, 5]
    delta = trajectory[frame, 6] 
    
    # Get control inputs safely
    if frame < len(controls):
        current_control = controls[frame]
        Fx = current_control[0]  # Longitudinal force (Fx)
        Delta_delta = current_control[1]  # Steering control input (Delta_delta)
    else:
        current_control = [0, 0]  # Default control values when out of bounds
        Fx = 0  # Default longitudinal force when out of bounds
        Delta_delta = 0  # Default steering control input when out of bounds

    # Calculate slip angles (optional for debugging)
    alpha_f, alpha_r = calculate_slip_angles(vx, vy, r, delta)

    # Calculate tire forces (optional for debugging)
    Fyf, Fyr = calculate_tire_forces(alpha_f, alpha_r)

    # Calculate cost (optional for debugging)
    cost = costs[frame] if frame < len(costs) else 0  # Ensure cost is within bounds

    # Update all text elements
    vx_text.set_text(f'vx: {vx:.2f} m/s')
    vy_text.set_text(f'vy: {vy:.2f} m/s')
    r_text.set_text(f'r: {r:.2f} rad/s')
    delta_text.set_text(f'delta: {delta:.2f} rad')
    Fx_text.set_text(f'Fx: {Fx:.2f} N')
    Delta_delta_text.set_text(f'Control Input (Δdelta): {Delta_delta:.2f} rad')
    alpha_f_text.set_text(f'α_f: {alpha_f:.2f} rad')
    alpha_r_text.set_text(f'α_r: {alpha_r:.2f} rad')
    Fyf_text.set_text(f'Fyf: {Fyf:.2f} N')
    Fyr_text.set_text(f'Fyr: {Fyr:.2f} N')
    cost_text.set_text(f'Cost: {cost:.2f}')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    buffer = 2.0

    if x < x_min + buffer or x > x_max - buffer or y < y_min + buffer or y > y_max - buffer:
        ax.set_xlim(min(x_min, x) - buffer, max(x_max, x) + buffer)
        ax.set_ylim(min(y_min, y) - buffer, max(y_max, y) + buffer)


    # Return updated plot elements for animation
    return trajectory_line, target_circle_line, car_marker, yaw_arrow, vx_text, vy_text, r_text, delta_text, Fx_text, Delta_delta_text, alpha_f_text, alpha_r_text, Fyf_text, Fyr_text, cost_text



# Adjust legend position to avoid overlap with the text
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory), interval=50, blit=False
)

# ani.save(f"Model1_r_{circle_radius}_N_{N}_dt_{dt}_animation..gif", writer='pillow', fps=20, dpi=150)
print("Save GIF!")

plt.tight_layout()
plt.show()


# ========================== Additional Metrics Plot ==========================

sim_time = np.linspace(0, len(trajectory) * 0.1, len(trajectory))  # Time vector for plotting

# Prepare data
yaw_list = trajectory[:, 2]
vx_list = trajectory[:, 3]
vy_list = trajectory[:, 4]
r_list = trajectory[:, 5]
vx_goal_list, r_goal_list = zip(*[circle_velocity_target(circle_radius, 2.5) for _ in vx_list])
vx_goal_list = np.array(vx_goal_list)
r_goal_list = np.array(r_goal_list)

# Create subplots
fig, axs = plt.subplots(4, 2, figsize=(14, 14))
fig.suptitle('MPC Performance Metrics Visualization', fontsize=16)

# Plot vx vs vx_goal
axs[0, 0].plot(sim_time, vx_list, color='blue', label='vx')
axs[0, 0].plot(sim_time, vx_goal_list, 'r--', label='vx_goal')
axs[0, 0].set_title('vx vs vx_goal')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Velocity [m/s]')
axs[0, 0].legend()
axs[0, 0].grid()

# Plot r vs r_goal
axs[0, 1].plot(sim_time, r_list, color='blue', label='r')
axs[0, 1].plot(sim_time, r_goal_list, 'r--', label='r_goal')
axs[0, 1].set_title('r vs r_goal')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Yaw Rate [rad/s]')
axs[0, 1].legend()
axs[0, 1].grid()

# Plot heading (yaw)
axs[1, 0].plot(sim_time, yaw_list, color='blue', label='Yaw')
axs[1, 0].set_title('Heading (Yaw) Over Time')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Yaw (heading) [rad]')
axs[1, 0].legend()
axs[1, 0].grid()

# Plot vy overtime
axs[1, 1].plot(sim_time, vy_list, color='blue', label='vy')
axs[1, 1].set_title('vy over Time')
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel('vy [m/s]')
axs[1, 1].legend()
axs[1, 1].grid()

# Plot Control Input
axs[2, 0].plot(sim_time[:-1], controls[:, 0], label='Fx', color='blue')
axs[2, 0].plot(sim_time[:-1], controls[:, 1], label='Delta delta', color='orange')
axs[2, 0].set_title('Control Inputs Over Time')
axs[2, 0].set_xlabel('Time [s]')
axs[2, 0].set_ylabel('Control Value')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Plot Front and Rear Slip Angles
axs[2, 1].plot(sim_time[:-1], alpha_f_list, label='Front Slip Angle', color='blue')
axs[2, 1].plot(sim_time[:-1], alpha_r_list, label='Rear Slip Angle', color='orange')
axs[2, 1].set_title('Front and Rear Slip Angles Over Time')
axs[2, 1].set_xlabel('Time [s]')
axs[2, 1].set_ylabel('Slip Angle [rad]')
axs[2, 1].legend()
axs[2, 1].grid(True)

# Plot cost over time
axs[3, 0].plot(sim_time[:-1], costs, color='blue', label='Cost')
axs[3, 0].set_title('MPC Cost over Time')
axs[3, 0].set_xlabel('Time [s]')
axs[3, 0].set_ylabel('Cost')
axs[3, 0].legend()
axs[3, 0].grid()

# Plot runtime per iteration
axs[3, 1].plot(sim_time[:-1], runtime_list, color='blue', label='Runtime per Iteration')
axs[3, 1].set_title('Runtime per Iteration')
axs[3, 1].set_xlabel('Time [s]')
axs[3, 1].set_ylabel('Runtime [s]')
axs[3, 1].legend(loc='upper right')
axs[3, 1].grid()

# Display total runtime text at bottom-right in axes coordinates
total_runtime = np.sum(runtime_list)
axs[3, 1].text(
    0.95, 0.05,
    f'Total Runtime: {total_runtime:.3f} s',
    transform=axs[2, 1].transAxes,
    color='red',
    fontsize=10,
    horizontalalignment='right',
    verticalalignment='bottom',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"Model1_r_{circle_radius}_N_{N}_dt_{dt}_subplots.png")
print("Save Subplot!")
plt.show()


