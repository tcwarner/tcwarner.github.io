import cv2
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import csv
import sys
import io

# Load the image and apply contrast adjustment
def load_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        # Set all values below 50 to 0
        image[image < 50] = 0
        # Scale values from 50 to 170 to 0 to 255
        mask = (image >= 50) & (image <= 170)
        image[mask] = np.clip(((image[mask] - 50) / 100.0) * 255, 0, 255)
        # Set all values above 150 to 255
        image[image > 170] = 255
    return image

# Pre-process the image with erosion to help isolate the boundaries
def preprocess_image(image):
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)  # Erode to remove small regions
    return eroded_image

def on_click(event, x, y, flags, param):
    global clicked_point, image_display, image, scale_percent, cell_counter
    if not hasattr(on_click, "visited_cells"):
        on_click.visited_cells = []

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

        x = int(x / (scale_percent / 100))  # Adjust for resized image
        y = int(y / (scale_percent / 100))

        # Check if this point is near an already clicked region
        for cell in on_click.visited_cells:
            if np.hypot(cell[0] - x, cell[1] - y) < 20:  # within 20 pixels
                #print("Cell already analyzed. Click ignored.")
                return

        on_click.visited_cells.append((x, y))

        processed_image = preprocess_image(image)
        boundary_points = trace_cell_boundary(processed_image, x, y)

        if boundary_points:
            ordered_boundary = order_boundary_points(boundary_points)
            smoothed_boundary = fit_smooth_curve(ordered_boundary)

            # Mark the cell with a bright red outline on both image and image_display
            for i in range(len(smoothed_boundary) - 1):
                pt1 = (int(smoothed_boundary[i][0]), int(smoothed_boundary[i][1]))
                pt2 = (int(smoothed_boundary[i + 1][0]), int(smoothed_boundary[i + 1][1]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)
                pt1_disp = (int(pt1[0] * scale_percent / 100), int(pt1[1] * scale_percent / 100))
                pt2_disp = (int(pt2[0] * scale_percent / 100), int(pt2[1] * scale_percent / 100))
                cv2.line(image_display, pt1_disp, pt2_disp, (0, 0, 255), 2)
            text_center=(pt1_disp[0]-40,pt1_disp[1]+10)
            cv2.putText(image_display, str(cell_counter), (text_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # Generate and adjust perpendicular vectors
            vectors = generate_perpendicular_vectors(smoothed_boundary, vector_length=50)
            adjusted_vectors = []
            vector_lengths = []
            vector_data = []

            for vector in vectors:
                start_point, end_point = vector
                num_steps = int(np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]))
                if num_steps == 0:
                    continue  # Avoid division by zero

                raw_intensities = []
                for i in range(num_steps + 1):
                    px = int(start_point[0] + i * (end_point[0] - start_point[0]) / num_steps)
                    py = int(start_point[1] + i * (end_point[1] - start_point[1]) / num_steps)
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        raw_intensities.append(image[py, px])

                if not raw_intensities:
                    continue  # Skip if no valid intensities were gathered

                # Apply 3-point smoothing
                smoothed_intensities = []
                for i in range(len(raw_intensities)):
                    if i == 0 or i == len(raw_intensities) - 1:
                        smoothed_intensities.append(raw_intensities[i])
                    else:
                        smoothed_intensities.append(sum(raw_intensities[i-1:i+2]) / 3)

                # Find first local minimum below 60
                min_distance = None
                for i in range(1, len(smoothed_intensities) - 1):
                    if smoothed_intensities[i] < 60 and smoothed_intensities[i] < smoothed_intensities[i - 1] and smoothed_intensities[i] < smoothed_intensities[i + 1]:
                        min_distance = i
                        break

                if min_distance is not None:
                    adjusted_end = (
                        int(start_point[0] + min_distance * (end_point[0] - start_point[0]) / num_steps),
                        int(start_point[1] + min_distance * (end_point[1] - start_point[1]) / num_steps)
                    )
                    adjusted_vectors.append((start_point, adjusted_end))
                    vector_lengths.append(min_distance)
                    vector_data.append(((start_point, adjusted_end), min_distance))

            # Remove outliers (1 standard deviation) with one iteration
            def remove_outliers_once(data):
                if not data:
                    return data
                lengths = [length for _, length in data]
                avg = np.mean(lengths)
                std = np.std(lengths)
                threshold = 1 * std
                return [(vec, length) for vec, length in data if abs(length - avg) <= threshold]

            filtered_data = remove_outliers_once(vector_data)

            # Print filtered vector lengths and statistics
            filtered_lengths = [length for _, length in filtered_data]
            print("Cell "+str(cell_counter)+",", filtered_lengths)
            cell_counter+=1
            if filtered_lengths:
                avg_length = np.mean(filtered_lengths)
                std_length = np.std(filtered_lengths)
                #print(f"Filtered Average Length, {avg_length:.2f}")
                #print(f"Filtered Standard Deviation, {std_length:.2f}")

            # Draw filtered vectors in white with thickness 2
            for vector, _ in filtered_data:
                start_point, end_point = vector
                #cv2.arrowedLine(image_display, start_point, end_point, (255, 255, 255), 2)

            region_size = 400
            x_min = max(x - region_size // 2, 0)
            y_min = max(y - region_size // 2, 0)
            x_max = min(x + region_size // 2, image.shape[1])
            y_max = min(y + region_size // 2, image.shape[0])
            roi = image[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi, (region_size, region_size), interpolation=cv2.INTER_AREA)

            # Draw the smooth curve on the cropped region (resized)
            for i in range(len(smoothed_boundary) - 1):
                bx1_rescaled = int((smoothed_boundary[i][0] - x_min) * region_size / (x_max - x_min))
                by1_rescaled = int((smoothed_boundary[i][1] - y_min) * region_size / (y_max - y_min))
                bx2_rescaled = int((smoothed_boundary[i + 1][0] - x_min) * region_size / (x_max - x_min))
                by2_rescaled = int((smoothed_boundary[i + 1][1] - y_min) * region_size / (y_max - y_min))
                cv2.line(roi_resized, (bx1_rescaled, by1_rescaled), (bx2_rescaled, by2_rescaled), (0, 0, 255), 2)

            # Draw filtered vectors on the region image
            for vector, _ in filtered_data:
                start_point, end_point = vector
                start_point_rescaled = (int((start_point[0] - x_min) * region_size / (x_max - x_min)),
                                       int((start_point[1] - y_min) * region_size / (y_max - y_min)))
                end_point_rescaled = (int((end_point[0] - x_min) * region_size / (x_max - x_min)),
                                     int((end_point[1] - y_min) * region_size / (y_max - y_min)))
                cv2.arrowedLine(roi_resized, start_point_rescaled, end_point_rescaled, (255, 255, 255), 2)

            # Display the region with the smooth boundary and vectors
            cv2.imshow("Region with Smoothed Boundary", roi_resized)

            # Update the main image with the smoothed boundary and vectors
            cv2.imshow("Electron Microscopy Image", image_display)


def trace_cell_boundary(image, x, y):
    boundary_points = []
    visited = set()
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))

        if image[cy, cx] >= 200:
            boundary_points.append((cx, cy))
            continue

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and (nx, ny) not in visited:
                stack.append((nx, ny))

    filtered_boundary_points = filter_largest_continuous_curve(boundary_points)
    return filtered_boundary_points

def filter_largest_continuous_curve(points, distance_threshold=3):
    dist_matrix = distance.cdist(points, points)
    adjacency_list = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if dist_matrix[i, j] <= distance_threshold:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    def dfs_iterative(node, visited, component, adjacency_list):
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                component.append(current_node)
                for neighbor in adjacency_list[current_node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    visited = set()
    components = []
    for i in range(len(points)):
        if i not in visited:
            component = []
            dfs_iterative(i, visited, component, adjacency_list)
            components.append(component)

    largest_component = max(components, key=len)
    return [points[i] for i in largest_component]

def order_boundary_points(points):
    # Use Convex Hull for reliable ordering
    points = np.array(points)
    hull = cv2.convexHull(points.astype(np.float32))
    ordered_points = hull[:, 0, :]
    return ordered_points.tolist()

def fit_smooth_curve(points, smoothing_factor=1.0):
    # Apply Gaussian filter to smooth the points before fitting
    points = np.array(points)
    smoothed_x = gaussian_filter1d(points[:, 0], sigma=1)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma=1)
    smoothed_points = np.column_stack((smoothed_x, smoothed_y))

    if not np.array_equal(smoothed_points[0], smoothed_points[-1]):  # Ensure the boundary is closed
        smoothed_points = np.vstack([smoothed_points, smoothed_points[0]])

    tck, u = splprep([smoothed_points[:, 0], smoothed_points[:, 1]], s=smoothing_factor, per=True)
    u_fine = np.linspace(0, 1, len(smoothed_points) * 5)
    smooth_x, smooth_y = splev(u_fine, tck)
    return list(zip(smooth_x, smooth_y))

def generate_perpendicular_vectors(smoothed_boundary, vector_length=50):
    vectors = []
    num_vectors = 25
    step = len(smoothed_boundary) // num_vectors

    for i in range(0, len(smoothed_boundary), step):
        pt1 = smoothed_boundary[i]
        pt2 = smoothed_boundary[(i + 1) % len(smoothed_boundary)]  # Wrap around for the last point

        # Compute the tangent vector
        tangent_x, tangent_y = pt2[0] - pt1[0], pt2[1] - pt1[1]

        # Normalize the tangent vector
        tangent_length = np.sqrt(tangent_x**2 + tangent_y**2)
        tangent_x /= tangent_length
        tangent_y /= tangent_length

        # Perpendicular vector (normal)
        normal_x = -tangent_y
        normal_y = tangent_x

        # Normalize the normal vector and scale it by the desired length
        normal_length = np.sqrt(normal_x**2 + normal_y**2)
        normal_x *= vector_length / normal_length
        normal_y *= vector_length / normal_length

        # Create the perpendicular vector start and end points
        start_point = (int(pt1[0]), int(pt1[1]))
        end_point = (int(pt1[0] + normal_x), int(pt1[1] + normal_y))

        vectors.append((start_point, end_point))

    return vectors

# Preserve printing to console and capture for CSV
class DualOutput:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

def main():
    global image, image_display, scale_percent
    filepath = "9677008.tif"
    image = load_image(filepath)

    global cell_counter
    cell_counter = 1

    if image is None:
        print("Error: Could not load image.")
        return

    screen_height = 800
    scale_percent = min(100, (screen_height / image.shape[0]) * 100)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image_display = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    cv2.imshow("Electron Microscopy Image", image_display)
    cv2.setMouseCallback("Electron Microscopy Image", on_click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save captured log to CSV file after GUI closes
    log_filename = filepath.rsplit('.', 1)[0] + "-analysis.csv"
    with open(log_filename, "w", newline="") as file:
        writer = csv.writer(file)
        log_buffer.seek(0)
        for line in log_buffer:
            clean_line = line.replace("[", "").replace("]", "")
            parts = [part.strip() for part in clean_line.strip().split(",") if part.strip()]
            if parts:
                writer.writerow(parts)

if __name__ == "__main__":
    log_buffer = io.StringIO()
    sys.stdout = DualOutput(sys.__stdout__, log_buffer)
    main()
    sys.stdout = sys.__stdout__