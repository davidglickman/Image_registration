import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import time
import psutil
import os
import sys

# Constants
IMAGE_SIZE = (640, 480)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add SuperGluePretrainedNetwork to sys.path
superglue_path = r"D:\Documents\immune navigation\SuperGluePretrainedNetwork"  # change to your path
sys.path.append(superglue_path)

from models.matching import Matching
from models.utils import frame2tensor

# SuperGlue setup (use same config as demo_superglue.py)
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
    },
    'superglue': {
        'weights': 'indoor',  # Change if needed
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(DEVICE)
keys = ['keypoints', 'scores', 'descriptors']

# Image preprocessing
def process_frame(frame):
    """Processes a frame for SuperGlue."""
    frame = cv2.resize(frame, IMAGE_SIZE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame_tensor = frame2tensor(frame_gray, DEVICE)
    return frame_tensor

# Run SuperGlue matching
def run_superglue_matching(tensor1, tensor2):
    """Runs SuperGlue matching on two frame tensors."""
    with torch.no_grad():
        pred = matching({'image0': tensor1, 'image1': tensor2})
    return pred

# Display matches
def display_matches(image1, image2, pred):
    """Displays matched keypoints."""
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1

    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    new_width = w1 + w2
    new_img = np.zeros((max(h1, h2), new_width, 3), dtype=np.uint8)
    new_img[:h1, :w1] = image1
    new_img[:h2, w1:] = image2

    for i in range(len(mkpts0)):
        pt0 = tuple(map(int, mkpts0[i]))
        pt1 = tuple(map(int, mkpts1[i] + np.array([w1, 0])))
        cv2.line(new_img, pt0, pt1, (0, 255, 0), 1)
        cv2.circle(new_img, pt0, 2, (0, 0, 255), -1)
        cv2.circle(new_img, pt1, 2, (0, 0, 255), -1)

    cv2.imshow("SuperGlue Matches", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Video processing
video_path = r"D:\Documents\immune navigation\movies\DJI_0002.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {frame_count}")

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

    tensor1 = process_frame(frame1)
    tensor2 = process_frame(frame2)

    pred = run_superglue_matching(tensor1, tensor2)
    display_matches(frame1, frame2, pred)

cap.release()
print("[INFO] Video processing complete.")
