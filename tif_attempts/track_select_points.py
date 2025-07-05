import os
import torch
import glob
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

import cv2
import numpy as np


def enhance_frame(frame):
    """
    Input: frame as a NumPy array (H, W, 3) in [0, 255]
    Output: enhanced frame
    """
    frame = frame.astype(np.uint8)
    
    # Convert to grayscale if needed
    if frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.squeeze()

    # CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Optional: sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Convert to 3-channel RGB again
    final = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    return final

video_path = "./tif_attempts/tracked_frames/stitched_tif.mp4"
video = read_video_from_path(video_path)
# video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
enhanced_frames = []

for frame in video:
    enhanced = enhance_frame(frame)
    enhanced_frames.append(enhanced)

# Stack back into torch tensor
video = torch.from_numpy(np.stack(enhanced_frames)).permute(0, 3, 1, 2)[None].float().cuda()


# Normalize to [0, 1]
video = video / 255.0
# --- Enhance contrast (especially dark outlines) ---
# Gamma correction (darkens mid-tones, enhances black outlines)
gamma = 0.6
video = torch.pow(video, gamma)

# Contrast stretching: increase contrast by remapping intensities
mean = video.mean()
video = (video - mean) * 2.0 + mean  # higher multiplier = more contrast

# Clip to valid range [0, 1] and rescale to [0, 255] for consistency
video = video.clamp(0, 1) * 255.0

# video = video.repeat_interleave(2, dim=1)


from cotracker.predictor import CoTrackerPredictor

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/scaled_offline.pth'
    )
)

if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()



# Load the CSV file from ImageJ
coords = np.loadtxt('./tif_attempts/Results.csv', delimiter=',', skiprows=1)  # adjust delimiter as needed
# coords[:, 0] = X, coords[:, 1] = Y

# Format for CoTracker: [frame_idx, x, y]
queries = torch.tensor([[0.0, x, y] for x, y in coords[:, -2:]], dtype=torch.float32)


# queries = torch.tensor([
#     [0., 614.0, 1700.0],
#     [0., 607.0, 1652.0],
#     # [0., 1323.0, 1810.0],
#     # [0., 1323.0, 1766.0],
#     [0., 2067.0, 1700.0],
#     [0., 2067.0, 1638.0],
#     [0., 2684.0, 1707.0],
#     [0., 2684.0, 1645.0],
#     [0., 4128.0, 1542.0],
#     [0., 4135.0, 1601.0],
#     # [0., 2489.0, 176.0],
#     # [0., 2476.0, 99.0],
#     [0., 2658.0, 0.0],
#     [0., 2665.0, 70.0],
#     [0., 3946.0, 503.0],
#     [0., 3949.0, 565.0],
#     [0., 3866.0, 514.0],
#     [0., 3863.0, 551.0],
#     [0., 4537.0, 606.0],
#     [0., 4544.0, 657.0],
#     [0., 1860.0, 510.0],
#     [0., 1863.0, 565.0],
#     [0., 4742.0, 1021.0],
#     [0., 4742.0, 1072.0],
#     [0., 3924.0, 2060.0],
#     [0., 3937.0, 2118.0],
#     [0., 4726.0, 2588.0],
#     [0., 4716.0, 2647.0],
#     [0., 2889.0, 2574.0],
#     [0., 2895.0, 2632.0],
#     [0., 3048.0, 2577.0],
#     [0., 3048.0, 2629.0],
#     [0., 345.0, 3102.0],
#     [0., 348.0, 3161.0],
#     [0., 4780.0, 3242.0],
#     [0., 4780.0, 3191.0],
# ])

if torch.cuda.is_available():
    queries = queries.cuda()

# pred_tracks, pred_visibility = model(video, queries=queries[None])
pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)


vis = Visualizer(
    save_dir='./tif_attempts/tracked_frames/select_points',
    linewidth=3, 
    mode='cool',
    tracks_leave_trace=-1
)


# make it automatically save a file with the next file name number
# Find existing files matching the pattern
save_dir = vis.save_dir
pattern = os.path.join(save_dir, "queries_backward_*.mp4")
existing_files = glob.glob(pattern)

# Extract numbers from filenames
existing_nums = []
for f in existing_files:
    base = os.path.basename(f)
    parts = base.replace(".mp4", "").split("_")
    if parts[-1].isdigit():
        existing_nums.append(int(parts[-1]))

next_num = max(existing_nums) + 1 if existing_nums else 1
filename = f'queries_backward_{next_num}'

vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename=filename
)