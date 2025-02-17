import numpy as np
import pygame
import scipy.optimize as opt

# Pygame initialization
pygame.init()
WIDTH, HEIGHT = 600, 600
SCALE = 100  # Scale factor for visualization
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Vehicle parameters
dt = 0.02  # Matching Matplotlib dt (50 Hz)
m = 2.35
L = 0.257
g = 9.81
b = 0.14328
a = L - b
G_front = m * g * b / L
G_rear = m * g * a / L
C_x = 116
C_y = 197
Iz = 0.045
mu = 1.31
mu_spin = 0.55
circle_radius = 1.5
v_max = 2.5
steer_max = 0.698

# Initialize state
x = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
traj_cog = []

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def dynamics(x, u):
    pos_x, pos_y, pos_phi = x[:3]
    Ux, Uy, r = x[3:]
    Ux_cmd, delta = u
    
    pos_x += Ux * np.cos(pos_phi) * dt
    pos_y += Ux * np.sin(pos_phi) * dt
    pos_phi += r * dt
    
    return np.array([pos_x, pos_y, pos_phi, Ux_cmd, Uy, r])


def dynamics_finite(x, u, dt):
    return x + dt * dynamics(x, u)


def cost_function(u, x0, N, dt):
    vx_goal = v_max
    r_goal = vx_goal / circle_radius
    cost = 0
    state = np.copy(x0)
    for i in range(N):
        throttle, steer = u[2 * i], u[2 * i + 1]
        state = dynamics_finite(state, np.array([throttle, steer]), dt)
        cost += (state[3] - vx_goal) ** 2 + (state[5] - r_goal) ** 2
    return cost

N = 3
throttle_bound = (-v_max, v_max)
steer_bound = (-steer_max, steer_max)
u_initial = np.zeros(2 * N)

def mpc_control(x0):
    bounds = [(-v_max, v_max), (-steer_max, steer_max)] * N
    result = opt.minimize(cost_function, u_initial, args=(x0, N, dt), method='SLSQP', bounds=bounds)
    return result.x[:2] if result.success else np.zeros(2)


def draw_vehicle(pos_x, pos_y, pos_phi):
    vehicle_length = 0.3 * SCALE
    vehicle_width = 0.15 * SCALE
    rect = pygame.Rect(0, 0, vehicle_length, vehicle_width)
    rect.center = (WIDTH / 2 + pos_x * SCALE, HEIGHT / 2 - pos_y * SCALE)
    rotated_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    rotated_surface.fill(RED)
    rotated_rect = pygame.transform.rotate(rotated_surface, -np.degrees(pos_phi))
    screen.blit(rotated_rect, rotated_rect.get_rect(center=rect.center))

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill(WHITE)
    u = mpc_control(x)
    x = dynamics_finite(x, u, dt)
    traj_cog.append((x[0], x[1]))
    
    draw_vehicle(x[0], x[1], x[2])
    
    for i in range(len(traj_cog) - 1):
        pygame.draw.line(screen, GREEN, (WIDTH / 2 + traj_cog[i][0] * SCALE, HEIGHT / 2 - traj_cog[i][1] * SCALE),
                         (WIDTH / 2 + traj_cog[i + 1][0] * SCALE, HEIGHT / 2 - traj_cog[i + 1][1] * SCALE), 2)
    
    pygame.display.flip()
    clock.tick(50)  # Matching Matplotlib animation speed (50 Hz)

pygame.quit()
