# STITCH MULTIPLE TIFF IMAGES INTO A .MP4 VIDEO
# import os
# from PIL import Image
# import imageio
# import numpy as np
# import tifffile

# # Input and output paths
# image_folder = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6"
# output_video_path = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/stitched_images.mp4"
# fps = 5

# # Ensure output folder exists
# os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# # Sort and load TIFFs
# image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".tif")])
# images = []

# for f in image_files:
#     img_path = os.path.join(image_folder, f)
#     img_np = tifffile.imread(img_path)

#     # Normalize to 0–255 and convert to uint8
#     img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)

#     # Convert grayscale to RGB
#     if img_np.ndim == 2:
#         img_rgb = np.stack([img_np] * 3, axis=-1)
#     else:
#         img_rgb = img_np  # If already RGB

#     images.append(img_rgb)

# # Save video
# with imageio.get_writer(output_video_path, fps=fps, codec="libx264") as writer:
#     for img in images:
#         writer.append_data(img)

# print(f"✅ Saved normalized video to {output_video_path}")


import tifffile
import os
import numpy as np
import imageio
import cv2

image_folder = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".tif")])
output_video_path = os.path.join(image_folder, "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/tif_stitched_images.mp4")

# Setup writer
writer = imageio.get_writer(output_video_path, fps=5, codec='libx264')

for f in image_files:
    img_path = os.path.join(image_folder, f)
    img = tifffile.imread(img_path)

    # Grayscale → RGB
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    # Normalize to [0, 255]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    writer.append_data(img)  # write frame

writer.close()
print("✅ Saved memory-efficient stitched video:", output_video_path)


'''
 - Resolution     : 4912 x 3264
 - FPS            : 5.0
 - Frame count    : 121
 - Duration       : 24.20 seconds
 - File size      : 23.00 MB
'''
# Re-open the saved video
cap = cv2.VideoCapture(output_video_path)

if not cap.isOpened():
    print("❌ Failed to open video for metadata check.")
else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print("📊 Video Properties:")
    print(f" - Resolution     : {width} x {height}")
    print(f" - FPS            : {fps}")
    print(f" - Frame count    : {frame_count}")
    print(f" - Duration       : {duration:.2f} seconds")
    cap.release()
# File size
file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
print(f" - File size      : {file_size:.2f} MB")



# STITCH MULTIPLE TIFF IMAGES INTO A MULTI-PAGE TIFF FILE
# import tifffile
# from tifffile import imwrite
# import os
# import numpy as np

# # Input/output paths
# image_folder = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6"
# output_tiff_path = "organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/stitched_images.tif"


# # Load and stack
# image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".tif")])
# images = []

# for f in image_files:
#     img_path = os.path.join(image_folder, f)
#     img_np = tifffile.imread(img_path)
#     images.append(img_np)

# # Stack and write
# imwrite(output_tiff_path, np.stack(images), photometric='minisblack')
# print(f"✅ Saved multi-page TIFF to {output_tiff_path}")
