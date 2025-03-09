# SuperGlue with Pose Estimation for Video Processing
# This script processes a video file frame by frame using SuperGlue for keypoint matching and estimates the 6-DoF pose (rotation and translation) between frames.
# Github: https://github.com/magicleap/SuperGluePretrainedNetwork


import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys

# Constants
IMAGE_SIZE = (640, 480)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add SuperGluePretrainedNetwork to sys.path
superglue_path = r"D:\Documents\immune navigation\SuperGluePretrainedNetwork"  # Change to your path
sys.path.append(superglue_path)

from models.matching import Matching
from models.utils import frame2tensor

# SuperGlue setup
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

# Image preprocessing
def process_frame(frame):
    frame = cv2.resize(frame, IMAGE_SIZE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(frame_gray, DEVICE)
    return frame_tensor

# Run SuperGlue matching
def run_superglue_matching(tensor1, tensor2):
    with torch.no_grad():
        pred = matching({'image0': tensor1, 'image1': tensor2})
    return pred

# Estimate 6-DoF pose (rotation + translation)
def estimate_pose(pred, K):
    """Estimates rotation and translation between two frames using matched keypoints."""
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1

    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts0) < 8:  # Essential matrix requires at least 8 points
        print("[WARNING] Not enough matches for pose estimation.")
        return None, None

    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)

    return R, t

# Display matches
def display_matches(image1, image2, pred):
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

# Camera intrinsic matrix (adjust for your camera)
K = np.array([[700, 0, IMAGE_SIZE[0] / 2],  
              [0, 700, IMAGE_SIZE[1] / 2],  
              [0, 0, 1]])  

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
    R, t = estimate_pose(pred, K)

    print(f"[INFO] Frame {i} → {i + frame_jump}")
    print("Rotation (R):\n", R)
    print("Translation (t):\n", t.T)  # Transposed for easier reading
    print("-" * 50)

    display_matches(frame1, frame2, pred)

cap.release()
print("[INFO] Video processing complete.")
