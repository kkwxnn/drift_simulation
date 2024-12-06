import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.optimize as opt

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
C_x = 116
C_alpha = 197
Iz = 0.045
mu = 1.31 # Typical dry road friction
mu_spin = 0.55 # Friction when tires are spinning

circle_radius = 1.5
v_max = 2.5 # m/s
steer_max = 0.698 # rad

# Initialize plot
fig, ax = plt.subplots()
ax.axis([-2, 2, -2, 2])
ax.set_aspect('equal')
line, = ax.plot([], [], 'b-')
traj_cog, = ax.plot([], [], 'g-')
traj_r, = ax.plot([], [], 'r-')

# Geometry
P = np.array([[-0.15, -0.15, 0.15, 0.15, -0.15],
              [-0.08, 0.08, 0.08, -0.08, -0.08],
              [1, 1, 1, 1, 1]])
W = np.array([[-0.03, -0.03, 0.03, 0.03, -0.03],
              [-0.015, 0.015, 0.015, -0.015, -0.015],
              [1, 1, 1, 1, 1]])
CoG = np.array([[0], [0], [1]])
r_axle = np.array([[-0.15], [0], [1]])

# Initialize state (x, y, theta, Ux, Uy, r)
x = [0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0]

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

def circle_velocity_target(circle_radius, v_max):
    vx_goal = v_max
    r_goal = vx_goal / circle_radius  # Yaw rate for circular motion
    return vx_goal, r_goal

# Define the cost function for MPC
def cost_function(u, x0, N, dt):
    # x0: initial state
    # N: prediction horizon
    # u: control inputs over the horizon (throttle, steering)
    vx_goal, r_goal = circle_velocity_target(circle_radius, v_max)
    
    # Weighting factors for the velocity and yaw rate errors
    alpha_vx = 1.0
    alpha_r = 1.0

    cost = 0
    state = np.copy(x0)
    
    # Predict the states and calculate the cost
    for i in range(N):
       
        throttle, steer = u[2*i], u[2*i+1]
        
        state = dynamics_finite(state, np.array([throttle, steer]), dt)
        
        x, y, yaw, vx, vy, r = state
        
        # cost += np.sum(state[:3]**2) + np.sum(u[2*i:2*i+2]**2)
        cost += alpha_vx * (vx - vx_goal)**2 + alpha_r * (r - r_goal)**2 # w_ss
        # print(f"Control inputs: {u}, Cost: {cost}")

    return cost

# MPC setup
N = 3  # prediction horizon

# u_initial = np.random.uniform(-2.0, 2.0, 2 * N)  # Randomize within bounds
# u_initial = [-2.043563,    0.54835626,  0.12012904,  0.19187852, -1.50917811, -0.61073667] # Backward
u_initial = [0.84183408,  0.45151292,  0.7560102,   0.35675445, -2.09682026,  0.33119607] # Forward

throttle_bound = (-v_max, v_max)  
steer_bound = (-steer_max, steer_max)

# throttle_initial = np.random.uniform(throttle_bound[0], throttle_bound[1], N)
# steer_initial = np.random.uniform(steer_bound[0], steer_bound[1], N)

# u_initial = np.zeros(2 * N) 
# u_initial[::2] = throttle_initial 
# u_initial[1::2] = steer_initial   

print(u_initial) 


def mpc_control(x0):
    
    throttle_bound = (-v_max, v_max)  
    steer_bound = (-steer_max, steer_max)  
    
    bounds = [throttle_bound, steer_bound] * N
    
    result = opt.minimize(cost_function, u_initial, args=(x0, N, dt), method='SLSQP', bounds=bounds)
    # print(f"Optimized control inputs: {result.x}")

    if not result.success:
        print("Optimization failed:", result.message)

    return result.x[:2]  # Return first control input pair (throttle, steer)

# Define the trajectories as lists to store the data points
traj_cog_x = []
traj_cog_y = []
traj_r_x = []
traj_r_y = []

# Heading arrow initialization
heading_arrow = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=3, color="black", label="Heading")

# Legend for trajectories
ax.legend([traj_cog, traj_r], ["CoG Trajectory (Green)", "Rear Trajectory (Red)"], loc="upper right")

# Add data lists to store values for subplots
vx_data = []
vy_data = []
yaw_data = []
r_data = []
slip_angle_data = []
steering_angle_data = []
vx_cmd_data = []

velocity_texts = [
    ax.text(0.05, 0.95, "v_x: 0.00", transform=ax.transAxes, color="black", fontsize=10),
    ax.text(0.05, 0.90, "v_y: 0.00", transform=ax.transAxes, color="black", fontsize=10)
]

# Modify the update_plot function to collect data for subplots
def update_plot(frame):
    global x
    u = mpc_control(x)  # Get control inputs from MPC
    x = dynamics_finite(x, u, dt)  # Apply the dynamics model

    pos_x, pos_y, pos_phi = x[:3]
    v_x, v_y = x[3], x[4]  # Assuming v_x and v_y are at index 3 and 4 of the state vector
    r = x[5]  # Yaw rate

    # Calculate slip angle for front and rear
    alpha_F = np.arctan((v_y + a * r) / abs(v_x))  # Front slip angle
    alpha_R = np.arctan((v_y - b * r) / abs(v_x))  # Rear slip angle
    
    # Get the current input
    vx_cmd = u[0]
    steering_angle = u[1]

    # Append the data for the subplots
    vx_data.append(v_x)
    vy_data.append(v_y)
    yaw_data.append(pos_phi)
    r_data.append(r)
    slip_angle_data.append(alpha_F)  # Assuming you want the front slip angle
    steering_angle_data.append(steering_angle)
    vx_cmd_data.append(vx_cmd)

    # Update the trajectory and heading arrow as before
    A = np.array([[np.cos(pos_phi), -np.sin(pos_phi), pos_x],
                  [np.sin(pos_phi), np.cos(pos_phi), pos_y],
                  [0, 0, 1]])
    pos = A @ P
    CoG_n = A @ CoG
    rear_n = A @ r_axle

    line.set_data(pos[0, :], pos[1, :])

    # Append new points to the trajectory lists
    traj_cog_x.append(CoG_n[0, 0])
    traj_cog_y.append(CoG_n[1, 0])
    traj_r_x.append(rear_n[0, 0])
    traj_r_y.append(rear_n[1, 0])

    # Set the data for the trajectory plots
    traj_cog.set_data(traj_cog_x, traj_cog_y)
    traj_r.set_data(traj_r_x, traj_r_y)

    # Update the heading arrow
    heading_arrow.set_offsets([pos_x, pos_y])
    heading_arrow.set_UVC(np.cos(pos_phi), np.sin(pos_phi))

    # Update the static velocity annotations (in top-left corner)
    velocity_texts[0].set_text(f"v_x: {v_x:.2f}")
    velocity_texts[1].set_text(f"v_y: {v_y:.2f}")
    
    # Stop after N frames (e.g., 500 frames)
    if frame == 200:
        plt.close()  # Close the plot window
        plot_simulation_results()  # Call the function to plot after the animation

# Create the plotting function for subplots
def plot_simulation_results():
    # Create subplots for the simulation data
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Vehicle Dynamics during Simulation", fontsize=16)

    # Plot v_x and v_y
    axs[0, 0].plot(vx_data, label="v_x (m/s)")
    axs[0, 0].axhline(y=v_max, color='r', linestyle='--', label="v_x_goal")
    axs[0, 0].set_ylabel("v_x (m/s)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(vy_data, label="v_y (m/s)")
    axs[0, 1].set_ylabel("v_y (m/s)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot yaw angle and yaw rate (r)
    axs[1, 0].plot(yaw_data, label="Yaw (rad)")
    axs[1, 0].set_ylabel("Yaw (rad)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(r_data, label="Yaw rate (rad/s)")
    axs[1, 1].axhline(y=v_max/circle_radius, color='r', linestyle='--', label="r_goal")
    axs[1, 1].set_ylabel("Yaw rate (rad/s)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    # Plot slip angle
    axs[2, 0].plot(slip_angle_data, label="Slip angle (rad)")
    axs[2, 0].set_ylabel("Slip angle (rad)")
    axs[2, 0].legend()
    axs[2, 0].grid()

    # # Plot steering angle
    # axs[2, 1].plot(steering_angle_data, label="Steering angle (rad)")
    # axs[2, 1].set_ylabel("Steering angle (rad)")
    # axs[2, 1].legend()
    # axs[2, 1].grid()

    # Plot Control Input
    axs[2, 1].plot(steering_angle_data, label="Steering angle (rad)")
    axs[2, 1].plot(vx_cmd_data, label="v_x command (m/s)")
    axs[2, 1].set_ylabel("Control Input")
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Start the animation
ani = FuncAnimation(fig, update_plot, frames=500, interval=dt * 1000)
plt.show()



