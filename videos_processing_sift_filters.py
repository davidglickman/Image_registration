import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import psutil  # For CPU & memory usage tracking

# Constants
FOV = 50  # Camera Field of View in degrees
CAMERA_HEIGHT = 100  # Camera height in meters
IMAGE_SIZE = (640, 480)

def track_resources():
    """Returns current CPU and memory usage."""
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def simulate_drone_view(image, fov=FOV, height=CAMERA_HEIGHT):
    """Simulate the drone's camera view by cropping based on FOV and height."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape

    crop_ratio = np.tan(np.radians(fov / 2)) * height * 0.01  # Scale to prevent over-cropping
    crop_width = max(int(w * (1 - crop_ratio)), 1)
    crop_height = max(int(h * (1 - crop_ratio)), 1)

    x_start = max((w - crop_width) // 2, 0)
    y_start = max((h - crop_height) // 2, 0)
    cropped = image_cv[y_start:y_start + crop_height, x_start:x_start + crop_width]

    resized = cv2.resize(cropped, IMAGE_SIZE)
    return resized

def rotate_image(image, angle):
    """Rotate the image by a given angle in degrees."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def extract_sift_features(image):
    """Extract SIFT features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_sift_features(des1, des2):
    """Match SIFT features using FLANN matcher."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

def estimate_transform_ransac(kp1, kp2, matches):
    """Estimate rotation & translation using RANSAC, returns inliers and homography."""
    if len(matches) < 4:
        print("[ERROR] Not enough matches for RANSAC.")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = [m for i, m in enumerate(matches) if mask[i]]
    return H, inliers

# Performance tracking
start_time = time.time()
cpu_start, mem_start = track_resources()

# Open video file
video_path = r"D:\Documents\immune navigation\movies\DJI_0002.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {frame_count}")

# Process frames
frame_jump = 20
num_frames_to_process = min(10 * frame_jump, frame_count - frame_jump)

for i in range(0, num_frames_to_process, frame_jump):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, i + frame_jump)
    ret2, frame2 = cap.read()

    if not ret1 or not ret2:
        print(f"[WARNING] Could not read frames {i} or {i + frame_jump}.")
        break

    image1_cv = cv2.resize(frame1, IMAGE_SIZE)
    image2_cv = cv2.resize(frame2, IMAGE_SIZE)

    kp1, des1 = extract_sift_features(image1_cv)
    kp2, des2 = extract_sift_features(image2_cv)

    matches = match_sift_features(des1, des2)
    H, inliers = estimate_transform_ransac(kp1, kp2, matches)

    if H is not None:
        dx = H[0, 2]
        dy = H[1, 2]
        rotation_angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        print(f"Frame {i} to {i+frame_jump}: Estimated Translation: Δx = {dx:.2f}, Δy = {dy:.2f}, Rotation: {rotation_angle:.2f}°")

        match_img = cv2.drawMatches(image1_cv, kp1, image2_cv, kp2, inliers[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f"SIFT Matching Frames {i} to {i+frame_jump}", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Frame {i} to {i+frame_jump}: No homography found.")

cap.release()

# Final resource tracking
cpu_end, mem_end = track_resources()
total_time = time.time() - start_time
print(f"\n[INFO] Total execution time: {total_time:.2f} sec")
print(f"[INFO] CPU usage: {cpu_end - cpu_start:.2f}%")
print(f"[INFO] Memory usage: {mem_end - mem_start:.2f}%")