import cv2
import numpy as np
import time
import csv

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for drawing paths
color = np.random.randint(0, 255, (100, 3))

cap = cv2.VideoCapture('./newdata/new.mp4') 


ret, first_frame = cap.read()
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize variables
points_prev = cv2.goodFeaturesToTrack(gray_first, maxCorners=100, qualityLevel=0.3, minDistance=7)
mask = np.zeros_like(first_frame)

# Variables for recording motion coordinates
motion_coordinates = []

# CSV file setup
csv_filename = 'motion_patterns.csv'
csv_header = ['Time', 'Motion Pattern']
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_header)

# Variables for capturing coordinates every 3 seconds
capture_interval = 3  # seconds
capture_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    points_next, status, _ = cv2.calcOpticalFlowPyrLK(gray_first, gray_frame, points_prev, None, **lk_params)

    # Select good points
    good_new = points_next[status == 1]
    good_old = points_prev[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Record motion coordinates
        motion_coordinates.append((int(a), int(b)))

    img = cv2.add(frame, mask)

    # Check if it's time to capture motion coordinates
    elapsed_time = time.time() - capture_start_time
    if elapsed_time >= capture_interval:
        # Save motion pattern to CSV
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        motion_pattern = ','.join([f'({x},{y})' for x, y in motion_coordinates])
        csv_writer.writerow([current_time, motion_pattern])

        # Reset variables for the next interval
        motion_coordinates = []
        capture_start_time = time.time()

    cv2.imshow('Optical Flow', img)

    # Update previous points and frames
    points_prev = good_new.reshape(-1, 1, 2)
    gray_first = gray_frame.copy()

    # Breaking the loop when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
