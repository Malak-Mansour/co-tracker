# from PIL import Image

# import os
# import torch
# import glob
# from cotracker.utils.visualizer import Visualizer, read_video_from_path

# import numpy as np

# grid_size = 100



# video_path = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/stitched_images.mp4"
# video = read_video_from_path(video_path)


# from cotracker.predictor import CoTrackerPredictor

# model = CoTrackerPredictor(
#     checkpoint=os.path.join(
#         'co-tracker/checkpoints/scaled_offline.pth'
#     )
# )

# # if torch.cuda.is_available():
# #     model = model.cuda()
# #     video = video.cuda()



# input_mask = 'organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks/masks0*.tif' #'./assets/apple_mask.png'
# segm_mask = np.array(Image.open(input_mask))


# pred_tracks, pred_visibility = model(video, grid_size=grid_size, segm_mask=torch.from_numpy(segm_mask)[None, None])
# vis = Visualizer(
#     save_dir='organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/tracked_masks',
#     pad_value=100,
#     linewidth=2,
# )

# # make it automatically save a file with the next file name number
# # Find existing files matching the pattern
# save_dir = vis.save_dir
# pattern = os.path.join(save_dir, "tracked_mask_*.mp4")
# existing_files = glob.glob(pattern)

# # Extract numbers from filenames
# existing_nums = []
# for f in existing_files:
#     base = os.path.basename(f)
#     parts = base.replace(".mp4", "").split("_")
#     if parts[-1].isdigit():
#         existing_nums.append(int(parts[-1]))

# next_num = max(existing_nums) + 1 if existing_nums else 1
# filename = f'tracked_mask_{next_num}'

# vis.visualize(
#     video=video,
#     tracks=pred_tracks,
#     visibility=pred_visibility,
#     filename=filename)



import os
import glob
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import cv2
import torch.nn.functional as F

grid_size = 100

# --- Load video ---
video_path = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/tif_stitched_images.mp4"
video = read_video_from_path(video_path)  # shape: [T, C, H, W]
video = torch.from_numpy(video).float().unsqueeze(0)  # [1, T, C, H, W]

if torch.cuda.is_available():
    video = video.cuda()

# --- Load model ---
model = CoTrackerPredictor(
    checkpoint=os.path.join('co-tracker/checkpoints/scaled_offline.pth')
)
if torch.cuda.is_available():
    model = model.cuda()

# --- Load and process first binary mask ---
input_mask_pattern = 'organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks/masks0*.tif'
mask_files = sorted(glob.glob(input_mask_pattern))
if not mask_files:
    raise FileNotFoundError(f"No mask files found matching: {input_mask_pattern}")

first_mask_path = mask_files[0]
print("First mask path:", first_mask_path)

mask_np = tifffile.imread(first_mask_path)
print("Original mask shape:", mask_np.shape, "dtype:", mask_np.dtype)
print("Original mask min/max:", mask_np.min(), mask_np.max())

# Optional: show histogram
unique, counts = np.unique(mask_np, return_counts=True)
print("Pixel value histogram:")
for u, c in zip(unique, counts):
    print(f"  Value {u}: {c} pixels")

if video.shape[-1] == 3:
    video = video.permute(0, 1, 4, 2, 3)
B, T, C, H, W = video.shape


print("Video shape      :", video.shape)  # [1, T, 3, H, W]
print("Mask shape       :", mask_np.shape)
print("Expected mask size:", (H, W))

if mask_np.shape != (H, W):
    diff_h = H - mask_np.shape[0]
    diff_w = W - mask_np.shape[1]
    if diff_h >= 0 and diff_w >= 0:
        print(f"Padding mask by (bottom={diff_h}, right={diff_w}) to match video shape...")
        mask_np = np.pad(mask_np, ((0, diff_h), (0, diff_w)), mode='constant', constant_values=0)
    else:
        raise ValueError(f"❌ Mask shape {mask_np.shape} is larger than video shape {(H, W)} — aborting.")

# Convert to binary mask and label
binary_mask = mask_np > 0
labeled_mask, num_cells = label(binary_mask)
print(f"✅ Detected {num_cells} cell(s) in the first frame mask.")

if num_cells == 0:
    raise ValueError("❌ No cells found in the mask. Please check the input.")

segm_mask = torch.from_numpy(labeled_mask)[None, None].long()
if torch.cuda.is_available():
    segm_mask = segm_mask.cuda()


# video = video[:, :60]  # Track only first 60 frames
# Optional: Downsample for memory efficiency
# Flatten time into batch dimension
video_2d = video.reshape(B * T, C, H, W)  # [B*T, C, H, W]

# Downsample
downscale_factor = 0.25
video_2d = torch.nn.functional.interpolate(
    video_2d, scale_factor=downscale_factor, mode='bilinear', align_corners=False
)

# Reshape back to [B, T, C, H, W]
H_ds, W_ds = video_2d.shape[-2:]
video = video_2d.reshape(B, T, C, H_ds, W_ds)

# Resize segmentation mask to match new spatial size
# Inside your CoTracker model (predictor.py), find where segm_mask is passed to F.interpolate
# Cast to float before interpolate, then back to long
segm_mask = F.interpolate(segm_mask.float(), size=(H_ds, W_ds), mode="nearest").long()



# --- Run CoTracker ---
pred_tracks, pred_visibility = model(video, grid_size=grid_size, segm_mask=segm_mask)

# --- Set up visualizer ---
vis = Visualizer(
    save_dir='organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/tracked_masks',
    pad_value=100,
    linewidth=2,
)

# --- Auto-increment output filename ---
save_dir = vis.save_dir
pattern = os.path.join(save_dir, "tracked_mask_*.mp4")
existing_files = glob.glob(pattern)

existing_nums = []
for f in existing_files:
    base = os.path.basename(f)
    parts = base.replace(".mp4", "").split("_")
    if parts[-1].isdigit():
        existing_nums.append(int(parts[-1]))

next_num = max(existing_nums) + 1 if existing_nums else 1
filename = f'tracked_mask_{next_num}'

# --- Visualize and save result ---
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename=filename
)
