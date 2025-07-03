import cv2
import numpy as np
import math
import itertools

# --- Configuration ---
SAVE_FRAME_NO = 10
NUM_BALLS = 3
VIDEO_FILENAME = 'juggling2.mp4'
PIXELS_PER_METER = 980.0
FPS = 30.0
TIME_STEP_SIZE = 1.0 / FPS
EULER_STEPS = 18

# HSV color bounds for ball detection
HSV_COLOR_BOUNDS = {
    'darkGreen': (np.array([35, 50, 20], np.uint8), np.array([80, 255, 120], np.uint8)),
    'green': (np.array([30, 0, 0], np.uint8), np.array([100, 255, 255], np.uint8)),
    'white': (np.array([0, 0, 80], np.uint8), np.array([255, 50, 120], np.uint8)),
    'yellow': (np.array([15, 204, 204], np.uint8), np.array([20, 255, 255], np.uint8)),
    'red': (np.array([0, 153, 127], np.uint8), np.array([4, 230, 179], np.uint8)),
    'orange': (np.array([15, 204, 204], np.uint8), np.array([20, 255, 255], np.uint8)),
    'darkYellow': (np.array([20, 115, 140], np.uint8), np.array([25, 205, 230], np.uint8)),
    'darkYellowAttempt2(isolating)': (np.array([20, 90, 117], np.uint8), np.array([32, 222, 222], np.uint8)),
    'orange2': (np.array([2, 150, 140], np.uint8), np.array([19, 255, 204], np.uint8)),
}

# Ball marker colors (BGR)
BALL_POSITION_MARKER_COLORS = [(200, 0, 0), (255, 200, 200), (0, 200, 0), (0, 0, 200)]
BALL_TRAJECTORY_MARKER_COLORS = [(200, 55, 55), (255, 200, 200), (55, 255, 55), (55, 55, 255)]

# --- Tracking parameters ---
WEIGHTED_FILTER = True
POSITION_PREDICTION_WEIGHT = 0.2
POSITION_OBSERVATION_WEIGHT = 0.8
VELOCITY_PREDICTION_WEIGHT = 0.2
VELOCITY_OBSERVATION_WEIGHT = 0.8
AVERAGED_OBSERVED_VELOCITY = False
BACKGROUND_SUBTRACTION = True
OUTLIER_REJECTION = True

# --- Physics ---
G_SECONDS = 9.81 * PIXELS_PER_METER
G_TIMESTEPS = G_SECONDS * (TIME_STEP_SIZE ** 2)

# --- Utility Functions ---
def get_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None

def blur(image):
    return cv2.medianBlur(image, 5)

def euler_extrapolate(position, velocity, acceleration, time_delta):
    position = [position[0] + velocity[0] * time_delta, position[1] + velocity[1] * time_delta]
    velocity = [velocity[0] + acceleration[0] * time_delta, velocity[1] + acceleration[1] * time_delta]
    return position, velocity

def get_trajectory(initial_position, initial_velocity, acceleration, num_traj_points):
    positions = []
    position = list(initial_position)
    velocity = list(initial_velocity)
    for _ in range(num_traj_points):
        position, velocity = euler_extrapolate(position, velocity, acceleration, 1)
        positions.append(position[:])
    return positions

def reject_outlier_points(points, m=2):
    if len(points) == 0:
        return np.array([])
    xs, ys = points[:, 0], points[:, 1]
    mean_x, std_x = np.mean(xs), np.std(xs)
    mean_y, std_y = np.mean(ys), np.std(ys)
    mask = (np.abs(xs - mean_x) < std_x * m) & (np.abs(ys - mean_y) < std_y * m)
    return points[mask]

def estimate_velocity(pos0, pos1, normalized=False):
    dx, dy = pos1[0] - pos0[0], pos1[1] - pos0[1]
    if normalized:
        mag = np.hypot(dx, dy)
        return [dx / mag, dy / mag] if mag != 0 else [0, 0]
    return [dx, dy]

def process_for_thresholding(frame, frame_no=0):
    blurred = blur(frame)
    if BACKGROUND_SUBTRACTION:
        fgbg = cv2.createBackgroundSubtractorMOG2(500, 30, True)
        fgmask = fgbg.apply(frame, None, 0.01)
        frame = cv2.bitwise_and(frame, frame, mask=fgmask)
        if frame_no == SAVE_FRAME_NO:
            cv2.imwrite('backgroundSub.jpg', frame)
            print('Wrote bg subtracted image')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def smooth_noise(frame):
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame, kernel)
    return frame

def initialize_ball_states(num_balls):
    return [[0, 0] for _ in range(num_balls)], [[0, 0] for _ in range(num_balls)]

def distance4d(p, q, r, s):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (r[0] - s[0]) ** 2 + (r[1] - s[1]) ** 2)

def find_balls_in_image(image, ball_centers, ball_velocities, frame_no=0):
    num_balls_to_find = len(ball_centers)
    points = np.column_stack(np.where(image > 0)).astype(np.float32)
    if OUTLIER_REJECTION:
        points = reject_outlier_points(points)
        if frame_no == SAVE_FRAME_NO and len(points) > 0:
            h, w = image.shape
            blank = np.zeros((h, w, 3), np.uint8)
            for x, y in points:
                cv2.circle(blank, (int(y), int(x)), 6, (255, 255, 255), thickness=6)
            cv2.imwrite('outlierRejected.jpg', blank)
            print('Wrote outlier image')
    if len(points) == 0 or len(points) < num_balls_to_find:
        return []
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(points, num_balls_to_find, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.tolist()
    centers = [[c[1], c[0]] for c in centers]  # (x, y) order
    pairings = []
    for (ball_center, ball_velocity), center in itertools.product(zip(ball_centers, ball_velocities), centers):
        theoretical_velocity = [center[0] - ball_center[0], center[1] - ball_center[1]]
        dist = distance4d(ball_center, center, theoretical_velocity, ball_velocity)
        pairings.append([(ball_center, ball_velocity), (center, theoretical_velocity), dist])
    sorted_pairings = sorted(pairings, key=lambda item: item[2])
    min_matches = []
    used_centers = set()
    for pairing in sorted_pairings:
        c = tuple(pairing[1][0])
        if c not in used_centers:
            min_matches.append(pairing)
            used_centers.add(c)
        if len(min_matches) == num_balls_to_find:
            break
    return min_matches

def draw_balls_and_trajectory(frame, matches, ball_centers, ball_velocities, ball_indices, centers_to_pair, velocities_to_pair, frame_no=0):
    if not matches:
        return frame
    matched_indices = set()
    for match in matches:
        for i, (center, velocity) in enumerate(zip(centers_to_pair, velocities_to_pair)):
            if match[0] == (center, velocity) and i not in matched_indices:
                global_idx = ball_indices[i]
                prev_pos = ball_centers[global_idx]
                prev_vel = ball_velocities[global_idx]
                obs_pos = match[1][0]
                obs_vel = [obs_pos[0] - prev_pos[0], obs_pos[1] - prev_pos[1]]
                if AVERAGED_OBSERVED_VELOCITY:
                    obs_vel = [(obs_vel[0] + prev_vel[0]) / 2.0, (obs_vel[1] + prev_vel[1]) / 2.0]
                if WEIGHTED_FILTER:
                    pred_pos = [prev_pos[0] + prev_vel[0], prev_pos[1] + prev_vel[1]]
                    pred_vel = [prev_vel[0], prev_vel[1] + G_TIMESTEPS]
                    ball_centers[global_idx] = [pred_pos[0] * POSITION_PREDICTION_WEIGHT + obs_pos[0] * POSITION_OBSERVATION_WEIGHT,
                                                pred_pos[1] * POSITION_PREDICTION_WEIGHT + obs_pos[1] * POSITION_OBSERVATION_WEIGHT]
                    ball_velocities[global_idx] = [pred_vel[0] * VELOCITY_PREDICTION_WEIGHT + obs_vel[0] * VELOCITY_OBSERVATION_WEIGHT,
                                                   pred_vel[1] * VELOCITY_PREDICTION_WEIGHT + obs_vel[1] * VELOCITY_OBSERVATION_WEIGHT]
                else:
                    ball_centers[global_idx] = obs_pos
                    ball_velocities[global_idx] = obs_vel
                matched_indices.add(i)
                break
    # Draw position markers and predicted trajectory
    for i in ball_indices:
        cx, cy = int(ball_centers[i][0]), int(ball_centers[i][1])
        cv2.circle(frame, (cx, cy), 6, BALL_POSITION_MARKER_COLORS[i], thickness=6)
        positions = get_trajectory((cx, cy), ball_velocities[i], (0, G_TIMESTEPS), EULER_STEPS)
        for pos in positions:
            px, py = int(pos[0]), int(pos[1])
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                cv2.circle(frame, (px, py), 2, BALL_TRAJECTORY_MARKER_COLORS[i], thickness=2)
    if frame_no == SAVE_FRAME_NO:
        cv2.imwrite('predicted.jpg', frame)
        print('Wrote predicted image')
    return frame

def main():
    ball_centers, ball_velocities = initialize_ball_states(NUM_BALLS)
    cap = cv2.VideoCapture(VIDEO_FILENAME)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    frame_no = 0
    while cap.isOpened():
        frame = get_frame(cap)
        if frame is None:
            break
        frame_copy = frame.copy()
        hsv_frame = process_for_thresholding(frame, frame_no)
        for color, ball_indices in zip(['orange2', 'darkYellowAttempt2(isolating)'], ([0, 1], [2])):
            color_bounds = HSV_COLOR_BOUNDS[color]
            thresh_img = cv2.inRange(hsv_frame, color_bounds[0], color_bounds[1])
            if frame_no == SAVE_FRAME_NO:
                cv2.imwrite('thresholded.jpg', thresh_img)
                print('Wrote thresholded image')
            thresh_img = smooth_noise(thresh_img)
            if frame_no == SAVE_FRAME_NO:
                cv2.imwrite('denoised.jpg', thresh_img)
                print('Wrote denoised image')
            centers_to_pair = [ball_centers[idx] for idx in ball_indices]
            velocities_to_pair = [ball_velocities[idx] for idx in ball_indices]
            matches = find_balls_in_image(thresh_img, centers_to_pair, velocities_to_pair, frame_no)
            frame_copy = draw_balls_and_trajectory(frame_copy, matches, ball_centers, ball_velocities, ball_indices, centers_to_pair, velocities_to_pair, frame_no)
        cv2.imshow('Image with Estimated Ball Center', frame_copy)
        out.write(frame_copy)
        k = cv2.waitKey(int(1000.0 / FPS)) & 0xFF
        if k == 27:
            break
        frame_no += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
