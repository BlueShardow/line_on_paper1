# David Tyler Dewenter

import cv2
import numpy as np
import math


def line_length(line):
    """
    Calculates the Euclidean length of a line.
    """
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(line):
    """
    Calculates the angle of a line in degrees.
    """
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    while angle < 0:
        angle += 180
    while angle > 180:
        angle -= 180
    return angle

def merge_close_lines_recursive(lines, min_distance, merge_angle_tolerance, vertical_leeway=1.5, horizontal_leeway=1.5):
    """
    Recursively merges lines that are close and have similar angles.
    Handles horizontal and vertical lines with distinct merging criteria.

    vertical_leeway: Multiplier for vertical tolerance in vertical lines.
    horizontal_leeway: Multiplier for horizontal tolerance in horizontal lines.
    """

    def weighted_average(p1, w1, p2, w2):
        """Computes a weighted average of two points."""
        return (p1 * w1 + p2 * w2) / (w1 + w2)

    def merge_once(lines):
        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = line1
            angle1 = calculate_angle(line1)
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = line2
                    angle2 = calculate_angle(line2)

                    # Check parallelism
                    if is_parallel(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
                        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
                        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

                        # Apply separate logic for merging based on orientation
                        if is_horizontal1 and is_horizontal2:
                            # Horizontal lines: More leeway in horizontal, less in vertical
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
                            if vertical_distance > min_distance * horizontal_leeway or horizontal_distance > min_distance:
                                continue
                        elif not is_horizontal1 and not is_horizontal2:
                            # Vertical lines: More leeway in vertical, less in horizontal
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
                            if vertical_distance > min_distance or horizontal_distance > min_distance * vertical_leeway:
                                continue

                        # Merge lines using weighted averages
                        l2_len = line_length(line2)
                        new_x1 = weighted_average(new_x1, line_weight, x3, l2_len)
                        new_y1 = weighted_average(new_y1, line_weight, y3, l2_len)
                        new_x2 = weighted_average(new_x2, line_weight, x4, l2_len)
                        new_y2 = weighted_average(new_y2, line_weight, y4, l2_len)
                        line_weight += l2_len
                        used[j] = True

            merged_lines.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            used[i] = True

        return merged_lines

    # Perform iterative merging until lines stabilize
    prev_lines = []
    while prev_lines != lines:
        prev_lines = lines
        lines = merge_once(lines)

    return lines


def visualize_angles(frame, lines, color=(255, 255, 255)):
    """
    Draws angle values near each line on the frame for debugging.
    """
    for line in lines:
        x1, y1, x2, y2 = line
        angle = calculate_angle(line)
        # Position the text near the midpoint of the line
        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # Put angle text on the frame
        cv2.putText(frame, f"{angle:.1f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def line_intersection_at_y(line, y):
    """
    Given a line (x1, y1, x2, y2) and a y-value, find the x-coordinate on the line at that y.
    If the line is nearly vertical, handle separately.
    """
    x1, y1, x2, y2 = line
    # Check for vertical line
    if abs(x2 - x1) < 1e-6:
        # Vertical line: x is constant
        return x1
    # Otherwise, interpolate
    # slope m = (y2 - y1) / (x2 - x1)
    m = (y2 - y1) / (x2 - x1)
    # y - y1 = m (x - x1)
    # x = x1 + (y - y1)/m
    x = x1 + (y - y1) / m
    return x


def draw_center_line_for_parallel_pairs(frame, parallel_lines):
    """
    Draw a center line for parallel pairs of lines.
    Ensures the center line is only drawn within the overlapping range.
    """
    used = [False] * len(parallel_lines)
    pairs = []

    # Pair parallel lines
    for i in range(len(parallel_lines)):
        if used[i]:
            continue
        for j in range(i + 1, len(parallel_lines)):
            if not used[j]:
                pairs.append((parallel_lines[i], parallel_lines[j]))
                used[i] = True
                used[j] = True
                break

    # Draw center lines for each pair
    for lineA, lineB in pairs:
        x1A, y1A, x2A, y2A = lineA
        x1B, y1B, x2B, y2B = lineB

        # Handle horizontal lines
        if abs(y1A - y2A) < abs(x1A - x2A) and abs(y1B - y2B) < abs(x1B - x2B):
            center_y = int((y1A + y2A + y1B + y2B) / 4)
            min_x = max(min(x1A, x2A), min(x1B, x2B))
            max_x = min(max(x1A, x2A), max(x1B, x2B))
            if max_x > min_x:
                cv2.line(frame, (min_x, center_y), (max_x, center_y), (255, 0, 0), 3)

        # Handle vertical lines
        elif abs(x1A - x2A) < abs(y1A - y2A) and abs(x1B - x2B) < abs(y1B - y2B):
            center_x = int((x1A + x2A + x1B + x2B) / 4)
            min_y = max(min(y1A, y2A), min(y1B, y2B))
            max_y = min(max(y1A, y2A), max(y1B, y2B))
            if max_y > min_y:
                cv2.line(frame, (center_x, min_y), (center_x, max_y), (255, 0, 0), 3)

        # Handle diagonal or non-aligned lines
        else:
            common_min_y = max(min(y1A, y2A), min(y1B, y2B))
            common_max_y = min(max(y1A, y2A), max(y1B, y2B))
            if common_max_y <= common_min_y:
                continue

            top_y = int(common_min_y)
            bottom_y = int(common_max_y)
            top_xA = line_intersection_at_y(lineA, top_y)
            top_xB = line_intersection_at_y(lineB, top_y)
            bottom_xA = line_intersection_at_y(lineA, bottom_y)
            bottom_xB = line_intersection_at_y(lineB, bottom_y)

            if top_xA > top_xB:
                top_xA, top_xB = top_xB, top_xA
            if bottom_xA > bottom_xB:
                bottom_xA, bottom_xB = bottom_xB, bottom_xA

            min_top_x = max(top_xA, top_xB)
            max_bottom_x = min(bottom_xA, bottom_xB)

            if min_top_x < max_bottom_x:
                cv2.line(frame, (int(min_top_x), top_y), (int(max_bottom_x), bottom_y), (255, 0, 0), 3)


def is_parallel(line1, line2, toward_tolerance, away_tolerance, distance_threshold):
    """
    Checks if two lines are parallel, with improved handling for horizontal and vertical lines.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    angle1 = calculate_angle(line1)
    angle2 = calculate_angle(line2)

    # Allow a small angle tolerance for nearly parallel lines
    angle_diff = abs(angle1 - angle2)
    if angle_diff > toward_tolerance and (180 - angle_diff) > away_tolerance:
        return False

    # Horizontal or vertical check
    is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
    is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

    # Check alignment and proximity
    if is_horizontal1 and is_horizontal2:
        vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
        return vertical_distance < distance_threshold
    elif not is_horizontal1 and not is_horizontal2:
        horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
        return horizontal_distance < distance_threshold

    return False


def detect_and_classify_lines(frame, max_line_gap=85, toward_tolerance=75, away_tolerance=75, merge_angle_tolerance=65,
                              distance_threshold=999999999, min_distance=250, min_line_length=150,
                              min_overlap_ratio=0.8,
                              proximity_threshold=20):
    """
    Detects and classifies lines based on their color, parallelism, proximity, and length.
    After identifying parallel blue lines, attempts to draw a center line.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue color range in HSV
    lower_blue = np.array([0, 0, 0])  # Widened lower bound for blue
    upper_blue = np.array([355, 255, 255])  # Upper bound for blue

    # Create a mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dilate the mask to account for proximity
    kernel = np.ones((proximity_threshold, proximity_threshold), np.uint8)
    blue_mask_dilated = cv2.dilate(blue_mask, kernel, iterations=1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    edges = cv2.Canny(enhanced_gray, 50, 150)

    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=max_line_gap
    )

    # Track parallel and non-parallel blue lines
    parallel_blue_lines = []
    non_parallel_blue_lines = []
    non_blue_lines = []

    if lines is not None:
        # Filter lines by length early
        filtered_lines = [line[0] for line in lines if line_length(line[0]) >= min_line_length]

        blue_lines = []
        for line in filtered_lines:
            x1, y1, x2, y2 = line

            # Create a mask for the current line
            mask_line = np.zeros_like(blue_mask)
            cv2.line(mask_line, (x1, y1), (x2, y2), 255, 2)

            # Check overlap with blue mask
            overlap = cv2.bitwise_and(blue_mask, mask_line)
            overlap_ratio = np.sum(overlap > 0) / np.sum(mask_line > 0) if np.sum(mask_line > 0) > 0 else 0

            # Check proximity to blue regions using the dilated mask
            proximity_overlap = cv2.bitwise_and(blue_mask_dilated, mask_line)
            proximity_ratio = np.sum(proximity_overlap > 0) / np.sum(mask_line > 0) if np.sum(mask_line > 0) > 0 else 0

            # Classify the line as blue if it satisfies either condition
            if overlap_ratio >= min_overlap_ratio or proximity_ratio >= min_overlap_ratio:
                blue_lines.append((x1, y1, x2, y2))
            else:
                non_blue_lines.append((x1, y1, x2, y2))

        # Merge close blue lines
        blue_lines = merge_close_lines_recursive(blue_lines, min_distance, merge_angle_tolerance)

        # Classify blue lines as parallel or not
        # We will store parallel sets
        used_in_parallel = set()
        for i in range(len(blue_lines)):
            found_parallel = False
            for j in range(i + 1, len(blue_lines)):
                if is_parallel(blue_lines[i], blue_lines[j], toward_tolerance, away_tolerance, distance_threshold):
                    parallel_blue_lines.append(blue_lines[i])
                    parallel_blue_lines.append(blue_lines[j])
                    found_parallel = True
                    used_in_parallel.add(i)
                    used_in_parallel.add(j)
            if not found_parallel and i not in used_in_parallel:
                non_parallel_blue_lines.append(blue_lines[i])

    # Visualize angles for all lines
    visualize_angles(frame, parallel_blue_lines, color=(0, 255, 0))  # Green for parallel lines
    visualize_angles(frame, non_parallel_blue_lines, color=(0, 255, 255))  # Yellow for non-parallel lines

    # Draw parallel blue lines in green
    for line in parallel_blue_lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green

    # Now draw the center line for parallel pairs
    draw_center_line_for_parallel_pairs(frame, parallel_blue_lines)

    return frame, blue_mask, blue_mask_dilated, edges, enhanced_gray, parallel_blue_lines


def calculate_distance_from_center(line, center_x, center_y):
    """
    Calculates the perpendicular distance of a line from the image center.
    """
    x1, y1, x2, y2 = line
    # Midpoint of the line
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    # Euclidean distance to the center
    return math.sqrt((mid_x - center_x) ** 2 + (mid_y - center_y) ** 2)


def get_bounding_box_between_lines(line1, line2):
    """
    Calculates the bounding box between two lines.
    """
    x1_min = min(line1[0], line1[2])
    x1_max = max(line1[0], line1[2])
    y1_min = min(line1[1], line1[3])
    y1_max = max(line1[1], line1[3])

    x2_min = min(line2[0], line2[2])
    x2_max = max(line2[0], line2[2])
    y2_min = min(line2[1], line2[3])
    y2_max = max(line2[1], line2[3])

    # Combine the bounding box
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def crop_to_center_lines(frame, parallel_lines):
    """
    Crops the image to focus on the two center-most parallel lines.
    """
    if len(parallel_lines) < 2:
        return None  # Not enough lines to crop

    # Calculate the image center
    height, width = frame.shape[:2]
    center_x, center_y = width / 2, height / 2

    # Find the two lines closest to the center
    distances = [
        (line, calculate_distance_from_center(line, center_x, center_y))
        for line in parallel_lines
    ]
    distances.sort(key=lambda x: x[1])  # Sort by distance from center

    # Select the two closest lines
    closest_lines = [distances[0][0], distances[1][0]]

    # Get the bounding box
    x_min, y_min, x_max, y_max = get_bounding_box_between_lines(*closest_lines)

    # Crop the image
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    return cropped_frame


def draw_closest_center_parallel_lines(frame, parallel_lines):
    """
    Draws the two closest parallel lines to the center and their center line.
    """
    if len(parallel_lines) < 2:
        return  # Not enough lines to draw

    # Calculate the image center
    height, width = frame.shape[:2]
    center_x, center_y = width / 2, height / 2

    # Find the two lines closest to the center
    distances = [
        (line, calculate_distance_from_center(line, center_x, center_y))
        for line in parallel_lines
    ]
    distances.sort(key=lambda x: x[1])  # Sort by distance from center

    # Select the two closest lines
    closest_lines = [distances[0][0], distances[1][0]]

    # Draw the two lines in green
    for line in closest_lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green

    # Draw the center line in blue
    draw_center_line_for_parallel_pairs(frame, closest_lines)

    return closest_lines


def detect_and_classify_lines_with_overlay(frame, **kwargs):
    """
    Wrapper function for detecting, classifying, and overlaying lines based on the closest parallel lines.
    """
    # Perform the usual detection and classification
    detected_frame, blue_mask, blue_mask_dilated, edges, enhanced_gray, parallel_blue_lines = detect_and_classify_lines(
        frame, **kwargs)

    # Highlight only the closest parallel lines
    draw_closest_center_parallel_lines(detected_frame, parallel_blue_lines)

    return detected_frame, blue_mask, blue_mask_dilated, edges, enhanced_gray, parallel_blue_lines


""""Above is the line detection logic, below is the websocket video stream stuff _________________________________________________________________________________________________________"""
"""DEBUGGING BELOW_________________________________________"""
# Accessing the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read frame.")
        break

    frame = frame[100: 700, 100: 700]
    
    # Perform detection with the updated overlay logic
    detected_frame, blue_mask, blue_mask_dilated, edges, enhanced_gray, parallel_blue_lines = detect_and_classify_lines_with_overlay(
        frame.copy())



    # Display the results
    cv2.imshow('Line Classification', detected_frame)

    """
    if cropped_frame is not None:
        cv2.imshow('Cropped ROI', cropped_frame)
    """

    cv2.imshow('Lines', edges)
    cv2.imshow('Contrast', enhanced_gray)

    key = cv2.waitKey(1) & 0xFF
    if chr(key).lower() == 'q':
        break

cap.release()
cv2.destroyAllWindows()
