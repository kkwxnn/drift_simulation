import numpy as np
import pygame
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
C_x = 116 # tire longitudinal stiffness
C_y = 197 # tire lateral stiffness
Iz = 0.045
mu = 1.31 # mu_peak
mu_spin = 0.55 # spinning coefficient

circle_radius = 1.5
v_max = 2.5 # m/s
steer_max = 0.698 # rad

# Initialize Pygame
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Vehicle Trajectory Visualization")
clock = pygame.time.Clock()

# Scaling factor to convert meters to pixels
scale = 200  # 1 meter = 200 pixels
origin = (width // 2, height // 2)  # Center of the screen

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

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

def tire_dyn(Ux, Ux_cmd, mu, mu_slide, Fz, C_x, C_y, alpha):
    # Longitudinal wheel slip
    if abs(Ux_cmd - Ux) < 1e-6:  # Ux_cmd approximately equal to Ux
        K = 0
    elif abs(Ux) < 1e-6:  # Handle Ux = 0
        Fx = np.sign(Ux_cmd) * mu * Fz
        Fy = 0
        return Fx, Fy
    else:
        K = (Ux_cmd - Ux) / abs(Ux)

    # Handle K < 0 case
    reverse = 1
    if K < 0:
        reverse = -1
        K = abs(K)

    # Alpha > pi/2 adaptation
    if abs(alpha) > np.pi / 2:
        alpha = (np.pi - abs(alpha)) * np.sign(alpha)

    # Calculate gamma with safeguard for K = -1
    gamma = np.sqrt(C_x**2 * (K / max(1 + K, 1e-6))**2 + C_y**2 * (np.tan(alpha) / max(1 + K, 1e-6))**2)

    # Friction model for gamma <= 3 * mu * Fz
    if gamma <= 3 * mu * Fz:
        F = gamma - (2 - mu_slide / mu) * gamma**2 / (3 * mu * Fz) + \
            (1 - (2 / 3) * (mu_slide / mu)) * gamma**3 / (9 * mu**2 * Fz**2)
    else:
        F = mu_slide * Fz

    # Compute forces with safeguard for gamma = 0
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

    Fxf, Fyf = tire_dyn(Ux, Ux, mu, mu_spin, G_front, C_x, C_y, alpha_F)
    Fxr, Fyr = tire_dyn(Ux, Ux_cmd, mu, mu_spin, G_rear, C_x, C_y, alpha_R)

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

def circle_velocity_target(circle_radius, v):
    vx_goal = v
    r_goal = vx_goal / circle_radius  # Yaw rate for circular motion
    return vx_goal, r_goal

# Define the cost function for MPC
def cost_function(u, x0, N, dt):
    # x0: initial state
    # N: prediction horizon
    # u: control inputs over the horizon (throttle, steering)
    vx = x0[3]

    if vx >= 0:
        v = v_max
    elif vx < 0:
        v = -v_max
    
    vx_goal, r_goal = circle_velocity_target(circle_radius, v)
    
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
N = 10 #3 # prediction horizon

throttle_bound = (-v_max, v_max)  
steer_bound = (-steer_max, steer_max)

throttle_initial = np.random.uniform(throttle_bound[0], throttle_bound[1], N)
steer_initial = np.random.uniform(steer_bound[0], steer_bound[1], N)

u_initial = np.zeros(2 * N) 
u_initial[::2] = throttle_initial 
u_initial[1::2] = steer_initial   

print(u_initial) 

# u_initial = [-1.47259428,  0.4394246,   2.11706717,  0.48196754, -1.48412146,  0.53425404,
#             -1.13362362,  0.60127473,  1.51094541, -0.01842053, -1.43568519, -0.5998862,
#             0.87441805, -0.41889592, -1.97393051,  0.52452572, -0.75560225,  0.35606266,
#             -1.32023036,  0.67893892] # N = 10

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

def update_frame():
    global x
    u = mpc_control(x)  # Get control inputs from MPC
    x = dynamics_finite(x, u, dt)  # Apply the dynamics model

    pos_x, pos_y, pos_phi = x[:3]
    v_x, v_y = x[3], x[4]  # Assuming v_x and v_y are at index 3 and 4 of the state vector

    A = np.array([[np.cos(pos_phi), -np.sin(pos_phi), pos_x],
                  [np.sin(pos_phi), np.cos(pos_phi), pos_y],
                  [0, 0, 1]])
    pos = A @ P
    CoG_n = A @ CoG
    rear_n = A @ r_axle

    # Append new points to the trajectory lists
    traj_cog_x.append(CoG_n[0, 0])
    traj_cog_y.append(CoG_n[1, 0])
    traj_r_x.append(rear_n[0, 0])
    traj_r_y.append(rear_n[1, 0])

    # Clear the screen
    screen.fill(WHITE)

    # Draw the target circle
    pygame.draw.circle(screen, BLUE, origin, int(circle_radius * scale), 1)

    # Draw the vehicle
    pygame.draw.polygon(screen, BLACK, [(int((pos[0, i] * scale) + origin[0]), int((pos[1, i] * scale) + origin[1])) for i in range(4)])

    # Draw the trajectories
    if len(traj_cog_x) > 1:
        pygame.draw.lines(screen, GREEN, False, [(int(x * scale) + origin[0], int(y * scale) + origin[1]) for x, y in zip(traj_cog_x, traj_cog_y)], 2)
        pygame.draw.lines(screen, RED, False, [(int(x * scale) + origin[0], int(y * scale) + origin[1]) for x, y in zip(traj_r_x, traj_r_y)], 2)

    # Draw the heading arrow
    dx = 0.3 * np.cos(pos_phi)
    dy = 0.3 * np.sin(pos_phi)
    pygame.draw.line(screen, BLACK, (int(pos_x * scale) + origin[0], int(pos_y * scale) + origin[1]), 
                     (int((pos_x + dx) * scale) + origin[0], int((pos_y + dy) * scale) + origin[1]), 2)

    # Display the velocities
    font = pygame.font.SysFont(None, 24)
    vx_text = font.render(f"v_x: {v_x:.2f}", True, BLACK)
    vy_text = font.render(f"v_y: {v_y:.2f}", True, BLACK)
    screen.blit(vx_text, (10, 10))
    screen.blit(vy_text, (10, 30))

    # Update the display
    pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    update_frame()
    clock.tick(int(1 / dt))

pygame.quit()