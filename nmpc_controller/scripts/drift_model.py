import numpy as np
import pygame
import math

# Constants
m = 1.98  # Mass of the vehicle (kg)
Iz = 0.24  # Moment of inertia around z-axis (kg*m^2)
Lf = 0.125  # Distance from the center of gravity to the front axle (m)
Lr = 0.125  # Distance from the center of gravity to the rear axle (m)
Bf, Cf, Df = 7.4, 1.2, 2.27
Br, Cr, Dr = 7.4, 1.2, 2.27
v_blend_min, v_blend_max = 0.1, 2.5  # Blending speed thresholds

# Initial state variables
x, y = 0.0, 0.0  # Initial position
yaw = 0.0  # Initial heading angle
vx, vy, r, delta = 0.0, 0.0, 0.0, 0.0  # Initialize velocities, and steering angle

# Simulation parameters
dt = 0.01  # Time step
screen_size = (800, 600)  # Screen size for visualization
scale = 50  # Scale factor for visualizing position (pixels per meter)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Car Simulation")
clock = pygame.time.Clock()

def calculate_lambda(vx, vy):
    phi = v_blend_min + 0.5 * (v_blend_max - v_blend_min)
    w = (2 * np.pi) / (v_blend_max - v_blend_min)
    lambda_val = 0.5 * (np.tanh(w * ((vx**2 + vy**2)**0.5 - phi)) + 1)
    return lambda_val

def calculate_slip_angles(vx, vy, r, delta):
    epsilon = 1e-5
    vx_safe = max(vx, epsilon)

    alpha_f = -np.arctan((Lf * r + vy) / vx_safe) + delta
    alpha_r = np.arctan((Lr * r - vy) / vx_safe)

    if vx < epsilon:
        alpha_f = alpha_r = 0

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

    x_dot_dyn = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_dyn = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_dyn = r
    vx_dot_dyn = (1 / m) * (Fx - Fyf * np.sin(delta) + m * vy * r)
    vy_dot_dyn = (1 / m) * (Fyr + Fyf * np.cos(delta) - m * vx * r)
    r_dot_dyn = (1 / Iz) * (Fyf * Lf * np.cos(delta) - Fyr * Lr)
    delta_dot_dyn = Delta_delta

    x_dot_kin = vx * np.cos(yaw) - vy * np.sin(yaw)
    y_dot_kin = vx * np.sin(yaw) + vy * np.cos(yaw)
    yaw_dot_kin = r
    vx_dot_kin = Fx / m
    vy_dot_kin = (Delta_delta * vx) * (Lr / (Lr + Lf))
    r_dot_kin = (Delta_delta * vx) * (1 / (Lr + Lf))
    delta_dot_kin = Delta_delta

    x_dot = lam * x_dot_dyn + (1 - lam) * x_dot_kin
    y_dot = lam * y_dot_dyn + (1 - lam) * y_dot_kin
    yaw_dot = lam * yaw_dot_dyn + (1 - lam) * yaw_dot_kin
    vx_dot = lam * vx_dot_dyn + (1 - lam) * vx_dot_kin
    vy_dot = lam * vy_dot_dyn + (1 - lam) * vy_dot_kin
    r_dot = lam * r_dot_dyn + (1 - lam) * r_dot_kin
    delta_dot = lam * delta_dot_dyn + (1 - lam) * delta_dot_kin

    x += x_dot * dt
    y += y_dot * dt
    yaw += yaw_dot * dt
    vx += vx_dot * dt
    vy += vy_dot * dt
    r += r_dot * dt
    delta += delta_dot * dt

    # Clamp vx to be within [-2.5, 2.5] m/s
    vx = max(min(vx, 2.5), -2.5)

    return x, y, yaw, vx, vy, r, delta

# Car visualization function
def draw_car(x, y, yaw):
    car_length = 0.32 * scale
    car_width = 0.12 * scale

    # Define car corners
    car_corners = np.array([
        [car_length / 2, car_width / 2],
        [car_length / 2, -car_width / 2],
        [-car_length / 2, -car_width / 2],
        [-car_length / 2, car_width / 2]
    ])

    # Rotation matrix for yaw
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # Rotate and translate corners
    rotated_corners = np.dot(car_corners, rotation_matrix.T)
    translated_corners = rotated_corners + np.array([x * scale + screen_size[0] // 2, y * scale + screen_size[1] // 2])
    pygame.draw.polygon(screen, (0, 255, 0), translated_corners)

    # Draw heading arrow
    arrow_length = car_length  # Length of the arrow
    arrow_start = np.array([x * scale + screen_size[0] // 2, y * scale + screen_size[1] // 2])
    arrow_end = arrow_start + arrow_length * np.array([np.cos(yaw), np.sin(yaw)])
    pygame.draw.line(screen, (255, 0, 0), arrow_start, arrow_end, 3)  # Red arrow

# Function to render text
def render_text(surface, text, position, color=(255, 255, 255), font_size=24):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

# Main simulation loop
Fx = 0.0
Delta_delta = 0.0

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        Fx += 0.1
    if keys[pygame.K_s]:
        Fx -= 0.1
    if keys[pygame.K_a]:
        Delta_delta = -0.1
    if keys[pygame.K_d]:
        Delta_delta = 0.1

    x, y, yaw, vx, vy, r, delta = update_state(Fx, Delta_delta, dt, x, y, yaw, vx, vy, r, delta)

    draw_car(x, y, yaw)  # Draw the car

    # Display velocity and steering angle
    render_text(screen, f"Velocity (vx): {vx:.2f} m/s", (10, 10))
    render_text(screen, f"Steering Angle (delta): {delta:.2f} rad", (10, 40))

    pygame.display.flip()
    clock.tick(60)  # 60 FPS

    Delta_delta = 0.0  # Reset steering input

pygame.quit()

