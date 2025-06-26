import os
from PIL import Image
import imageio
import numpy as np

# Path to jpg folder and output
image_folder = "/home/malak.mansour/Downloads/DEP/tif_jpg"
output_video_path = "stitched_input_video.mp4"
fps = 5  # Adjust as needed

# Sort and load images
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
)
images = []

for f in image_files:
    img_path = os.path.join(image_folder, f)
    img = Image.open(img_path).convert("RGB")
    images.append(img)

# Use imageio to save video from PIL images
with imageio.get_writer(output_video_path, fps=fps, codec="libx264") as writer:
    for img in images:
        writer.append_data(np.array(img))  # Convert to NumPy before writing

print(f"Saved video to {output_video_path}")
