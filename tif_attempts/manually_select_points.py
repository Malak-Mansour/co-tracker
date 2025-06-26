import torch
import matplotlib.pyplot as plt
from cotracker.utils.visualizer import read_video_from_path

video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/stitched_tif.mp4"
video_np = read_video_from_path(video_path)  # shape: (T, H, W, C), type: np.ndarray

# Convert to torch tensor
video = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # (T, C, H, W)

# Get first frame (frame 0) as NumPy (H, W, 3)
first_frame = video[0].permute(1, 2, 0).float().numpy() / 255.0

# Show and click
plt.imshow(first_frame)
plt.title("Click on the image to select points (close window when done)")
clicked_points = plt.ginput(n=-1, timeout=0)
plt.close()

# Format as queries: [0, x, y] (0 = frame index)
queries = torch.tensor([[0., x, y] for x, y in clicked_points], dtype=torch.float32)

print("Selected query points:")
print(queries)
