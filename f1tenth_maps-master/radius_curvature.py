import os
import cv2
import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares

# =============================
# 1. Load map and metadata
# =============================
def load_map(png_path, yaml_path):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    with open(yaml_path, 'r') as f:
        map_metadata = yaml.safe_load(f)
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    return img, resolution, origin

# =============================
# 2. Preprocess: binarize + skeletonize
# =============================
def get_skeleton(img):
    # Keep only white lane (pure white = 255)
    lane_mask = cv2.inRange(img, 250, 255)  # Only pixels in [250, 255] range
    lane_mask = lane_mask // 255  # Normalize to 0 or 1

    # Skeletonize only the white lane
    skeleton = skeletonize(lane_mask).astype(np.uint8)
    return skeleton


# =============================
# 3. Extract centerline points
# =============================
def extract_points(skeleton):
    points = np.column_stack(np.where(skeleton > 0))
    points = np.flip(points, axis=1)  # (x, y)
    return points

# =============================
# 4. Sort points roughly along the path
# =============================
def sort_points(points):
    sorted_points = [points[0]]
    points = list(points[1:])
    while points:
        last = sorted_points[-1]
        dists = np.linalg.norm(points - last, axis=1)
        idx = np.argmin(dists)
        sorted_points.append(points[idx])
        points.pop(idx)
    return np.array(sorted_points)

# =============================
# 5. Compute curvature
# =============================
def compute_curvature(x, y):
    # Smooth first (important to reduce noise)
    x_smooth = savgol_filter(x, window_length=31, polyorder=3)
    y_smooth = savgol_filter(y, window_length=31, polyorder=3)

    # First derivatives
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

# =============================
# 6. Circle fitting (least squares)
# =============================
def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    def calc_R(c):
        xc, yc = c
        return np.sqrt((x - xc)**2 + (y - yc)**2)
    def cost(c):
        Ri = calc_R(c)
        return Ri - Ri.mean()
    res = least_squares(cost, [x_m, y_m])
    xc, yc = res.x
    Ri = calc_R([xc, yc])
    radius = Ri.mean()
    return xc, yc, radius

# =============================
# 7. Main Program
# =============================
def main():
    # Set the base directory
    base_dir = r"C:\Users\kwan2\Documents\GitHub\drift_simulation\f1tenth_maps-master\maps" # Window
    # base_dir = r"/home/kkwxnn/drift_simulation/src/f1tenth_maps-master/maps" # Ubuntu
    png_path = os.path.join(base_dir, "blackbox2022_1.png")
    yaml_path = os.path.join(base_dir, "blackbox2022_1.yaml")

    # Load the map
    img, resolution, origin = load_map(png_path, yaml_path)
    skeleton = get_skeleton(img)
    points = extract_points(skeleton)
    points = sort_points(points)

    # Pixel to meters conversion
    x_pix = points[:, 0]
    y_pix = points[:, 1]
    x_meters = x_pix * resolution + origin[0]
    y_meters = (img.shape[0] - y_pix) * resolution + origin[1]  # flip y-axis

    # Compute curvature
    curvature = compute_curvature(x_meters, y_meters)

    # Detect corners: peaks in curvature
    peaks, _ = find_peaks(curvature, height=np.percentile(curvature, 90), distance=30)
    print(f"Detected {len(peaks)} potential corners.")

    # Fit circle and filter only large corners (>90 deg)
    raw_corners = []
    window_size = 40  # number of points to fit around each peak

    for peak in peaks:
        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(x_meters))
        x_corner = x_meters[start:end]
        y_corner = y_meters[start:end]

        if len(x_corner) < 10:
            continue

        # Fit circle
        xc, yc, radius = fit_circle(x_corner, y_corner)

        # Compute turn angle
        vec1 = np.array([x_corner[0] - xc, y_corner[0] - yc])
        vec2 = np.array([x_corner[-1] - xc, y_corner[-1] - yc])
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-5 or norm2 < 1e-5:
            continue

        angle_rad = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # if angle_deg > 90:
        if angle_deg > 45:
            raw_corners.append((radius, angle_deg, xc, yc))

    # Sort corners by X position (you can change to sort by Y or other criteria if needed)
    raw_corners.sort(key=lambda c: c[2])  # sort by center X
    corner_infos = [(i + 1, *c) for i, c in enumerate(raw_corners)]  # add corner IDs

    for corner_id, radius, angle_deg, xc, yc in corner_infos:
        print(f"Corner {corner_id}: Radius = {radius:.2f} m | Angle = {angle_deg:.1f}° | Center = ({xc:.2f}, {yc:.2f})")
        
    # Save results to CSV
    map_name = os.path.splitext(os.path.basename(png_path))[0]
    output_dir = r"C:\Users\kwan2\Documents\GitHub\drift_simulation\f1tenth_maps-master"
    output_csv = os.path.join(output_dir, f"{map_name}_corners.csv")
    output_plot = os.path.join(output_dir, f"{map_name}_corners.png")

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corner ID", "Radius (meters)", "Turn Angle (degrees)", "Center X (meters)", "Center Y (meters)"])
        for corner in corner_infos:
            writer.writerow(corner)

    print(f"\nCorner data saved to: {output_csv}")

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray',
               extent=[origin[0], origin[0] + img.shape[1] * resolution,
                       origin[1], origin[1] + img.shape[0] * resolution])
    plt.plot(x_meters, y_meters, 'r-', linewidth=1)

    for corner_id, radius, angle_deg, xc, yc in corner_infos:
        plt.plot(xc, yc, 'bo')
        # Corner ID (bold and red, to the right of the point)
        plt.text(xc + 1.5, yc, f"{corner_id}",
                color='red', fontsize=10, fontweight='bold', ha='left', va='center')

        # Radius and angle (blue, below the point)
        plt.text(xc, yc + 1.5, f"{radius:.1f}m\n{angle_deg:.0f}°",
                color='blue', fontsize=8, ha='center', va='top')

    plt.gca().invert_yaxis()
    plt.title('Centerline and Detected Large Corners (>45°)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid()

    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
