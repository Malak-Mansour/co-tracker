# save images not videos (from .tif)
'''
import torch
import os
import numpy as np
import imageio.v3 as iio
from torchvision.transforms import functional as TF

# Parameters
folder_path = '/home/malak.mansour/Downloads/DEP/nd_077_right'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
grid_size = 10

# Step 1: Load all .tif files sorted by name
tif_files = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith('.tif')
])

# Step 2: Read each frame into a list
frames = [iio.imread(f) for f in tif_files]

# Step 3: Convert to numpy array and ensure correct shape
frames_np = np.stack(frames)  # Shape: (T, H, W) or (T, H, W, C)
if frames_np.ndim == 3:  # grayscale (T, H, W)
    frames_np = np.stack([frames_np]*3, axis=-1) # (T, H, W, 1)

# Step 4: Convert to tensor [B, T, C, H, W]



# # Resize to (384, 512)
# resized_frames = [TF.resize(torch.from_numpy(f).permute(2, 0, 1).float(), size=(384, 512)).permute(1, 2, 0).numpy()
#                   for f in frames_np]
# frames_np = np.stack(resized_frames)

# # Convert to tensor
# video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].float().to(device)
# video = video / 255.0



frames_np = frames_np.astype(np.float32)
frames_np /= frames_np.max()  # Normalize to [0, 1]
video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].to(device)  # B T C H W


# Optional: normalize [0, 255] to [0, 1] if needed
video = video / 255.0

# Step 5: Load CoTracker
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Step 6: Run CoTracker
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)

# print(f"video shape: {video.shape}, dtype: {video.dtype}")

for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1

    print(f"Frame {ind} to {ind + cotracker.step * 2}: Tracks shape: {pred_tracks.shape}")




#Visualize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Create output directory
os.makedirs("tracked_frames_tif", exist_ok=True)

# Convert video tensor back to numpy for visualization
video_np = (video.squeeze().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8) # (T, H, W, C)


# print("pred_tracks shape:", pred_tracks.shape)
# print("pred_visibility shape:", pred_visibility.shape)

# Loop through frames and plot
for t in range(pred_tracks.shape[1]):
    frame = video_np[t].squeeze()  # (H, W, C) or (H, W)
    
    # If grayscale
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(frame, cmap="gray")

    # Plot all tracked points for this frame
    for n in range(pred_tracks.shape[2]):
        x, y = pred_tracks[0, t, n].detach().cpu().numpy()
        vis = pred_visibility[0, t, n].item()
        if vis > 0.5:  # Show only visible points
            ax.plot(x, y, 'ro', markersize=3)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"tracked_frames/frame_{t:04d}.png", bbox_inches='tight')
    plt.close()
'''


# save images not videos (from jpg)
'''
import torch
import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

# Parameters
folder_path = '/home/malak.mansour/Downloads/DEP/tif_jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
grid_size = 10

# Step 1: Load all .jpg files sorted by name
jpg_files = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith('.jpg')
])

# Step 2: Read each frame into a list
frames = [iio.imread(f) for f in jpg_files]

# Step 3: Convert to numpy array and ensure correct shape
frames_np = np.stack(frames)  # (T, H, W) or (T, H, W, C)
if frames_np.ndim == 3:  # grayscale
    frames_np = np.stack([frames_np]*3, axis=-1)  # (T, H, W, 3)

# Step 4: Convert to tensor [B, T, C, H, W] and normalize once
frames_np = frames_np.astype(np.float32) / 255.0
video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].to(device)  # B T C H W

# Step 5: Load CoTracker
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Step 6: Run CoTracker
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)

# print(f"video shape: {video.shape}, dtype: {video.dtype}")

# Step 7: Iterate through and track
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )
    print(f"Frame {ind} to {ind + cotracker.step * 2}: Tracks shape: {pred_tracks.shape}")




# === Visualization ===
os.makedirs("tracked_frames_jpg", exist_ok=True)

# Convert back to [T, H, W, C]
video_np = (video.squeeze().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

# print("pred_tracks shape:", pred_tracks.shape)
# print("pred_visibility shape:", pred_visibility.shape)

for t in range(pred_tracks.shape[1]):
    frame = video_np[t]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(frame)  # RGB, no need for cmap
    for n in range(pred_tracks.shape[2]):
        x, y = pred_tracks[0, t, n].detach().cpu().numpy()
        vis = pred_visibility[0, t, n].item()
        if vis > 0.5:
            ax.plot(x, y, 'ro', markersize=3)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"tracked_frames/frame_{t:04d}.jpg", pad_inches=0)
    plt.close()
'''



# using and saving videos, keeping resolution

import torch
import torchvision
from torchvision.io import read_video
from PIL import Image, ImageDraw
import imageio
import numpy as np
import os

# === CONFIG ===
video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/yellow_cup.mp4"
output_video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/yellow_cup_tracked.mp4"
# video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/stitched_tif.mp4"
# output_video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/tracked_tif.mp4"


grid_size = 20
fps = 10  # You can adjust based on your original video
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
cotracker.eval()

# === LOAD VIDEO ===
video_frames, _, _ = read_video(video_path, pts_unit='sec')  # (T, H, W, C)
video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
video = video_frames.unsqueeze(0).to(device)  # (1, T, C, H, W)

# === RUN COTRACKER ===
with torch.no_grad():
    cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)
    all_tracks = []
    for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video[:, ind : ind + cotracker.step * 2]
        )
        all_tracks.append((pred_tracks, pred_visibility))

print("Tracking completed.")

# === CONCATENATE TRACKS ===
tracks_all = torch.cat([t[0] for t in all_tracks], dim=1)  # (B, T, N, 2)
tracks_all = tracks_all[0].cpu().numpy()  # (T, N, 2)

# === OVERLAY TRACKS AND SAVE VIDEO ===
frames_np = (video_frames * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)

with imageio.get_writer(output_video_path, fps=fps, codec="libx264") as writer:
    for t in range(frames_np.shape[0]):
        frame = Image.fromarray(frames_np[t])
        draw = ImageDraw.Draw(frame)

        if t < tracks_all.shape[0]:
            for n in range(tracks_all.shape[1]):
                x, y = tracks_all[t, n]
                if not np.isnan(x) and not np.isnan(y):
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

        writer.append_data(np.array(frame))

print(f"Saved tracked video to: {output_video_path}")