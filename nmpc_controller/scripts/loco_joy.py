import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize Joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Constants
dt = 0.02 

# Model parameters
m = 2.35 # mass (kg)
L = 0.257 # wheelbase (m)
g = 9.81
b = 0.14328 # CoG to rear axle
a = L - b # CoG to front axle
G_front = m * g * b / L # calculated load or specify front rear load directly
G_rear = m * g * a / L
C_x = 116 # longitude stiffness
C_alpha = 197 # laternal stiffness
Iz = 0.045 # rotation inertia 
mu = 1.31
mu_spin = 0.55

# Initialize visualization
fig, ax = plt.subplots()
ax.axis([-2, 2, -2, 2])
ax.set_aspect('equal')
line, = ax.plot([], [], 'b-')
traj_cog, = ax.plot([], [], 'g-')
traj_r, = ax.plot([], [], 'r-')
heading_arrow, = ax.plot([], [], 'k-', lw=2)  


# Geometry
P = np.array([[-0.15, -0.15, 0.15, 0.15, -0.15],
              [-0.08, 0.08, 0.08, -0.08, -0.08],
              [1, 1, 1, 1, 1]])
W = np.array([[-0.03, -0.03, 0.03, 0.03, -0.03],
              [-0.015, 0.015, 0.015, -0.015, -0.015],
              [1, 1, 1, 1, 1]])
CoG = np.array([[0], [0], [1]])
r_axle = np.array([[-0.15], [0], [1]])

# Initialize state
x = np.zeros(6)

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


def update_plot(frame):
    global x
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    throttle = max(-2.5, -2.5 * joystick.get_axis(1))  # Adjust throttle scaling 
    steer = -0.698 * joystick.get_axis(0)  # Adjust steering scaling 
    u = np.array([throttle, steer])
    x = dynamics_finite(x, u, dt)

    pos_x, pos_y, pos_phi = x[:3]
    A = np.array([[np.cos(pos_phi), -np.sin(pos_phi), pos_x],
                  [np.sin(pos_phi), np.cos(pos_phi), pos_y],
                  [0, 0, 1]])
    pos = A @ P
    CoG_n = A @ CoG
    rear_n = A @ r_axle

    # Update car body position
    line.set_data(pos[0, :], pos[1, :])

    # Update trajectories
    traj_cog.set_data(np.append(traj_cog.get_xdata(), CoG_n[0, 0]),
                      np.append(traj_cog.get_ydata(), CoG_n[1, 0]))
    traj_r.set_data(np.append(traj_r.get_xdata(), rear_n[0, 0]),
                    np.append(traj_r.get_ydata(), rear_n[1, 0]))

    # Update heading arrow
    arrow_length = 0.3  # Adjust length as needed
    arrow_x = [pos_x, pos_x + arrow_length * np.cos(pos_phi)]
    arrow_y = [pos_y, pos_y + arrow_length * np.sin(pos_phi)]
    heading_arrow.set_data(arrow_x, arrow_y)


ani = FuncAnimation(fig, update_plot, interval=dt * 1000)
plt.show()
pygame.quit()
