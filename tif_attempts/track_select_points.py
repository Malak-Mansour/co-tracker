# '''
import os
import torch
import glob
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML


video_path = "./tif_attempts/tracked_frames/stitched_tif.mp4"
video = read_video_from_path(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()


# Normalize to [0, 1]
video = video / 255.0

'''
# Increase brightness and contrast
brightness_factor = 0.2   # shift all pixel values up (try 0.1 to 0.3)
contrast_factor = 1.4     # scale contrast (try 1.1 to 1.5)

video = (video - 0.5) * contrast_factor + 0.5  # contrast
video = video + brightness_factor              # brightness
'''

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



# queries = torch.tensor([
#     [0., 400., 350.],  # point tracked from the first frame
#     [10., 600., 500.], # frame number 10
#     [20., 750., 600.], # ...
#     [30., 900., 200.]
# ])
# queries = torch.tensor([
#     [0., 239., 276.],
#     [0., 74., 203.],
#     [0., 52., 163.],
#     [0., 417., 75.],
#     [0., 419., 111.],
#     [0., 1474., 374.],
#     [0., 1481., 426.]
# ])
# queries = torch.tensor([
#     [0., 2064., 1165.],
#     [0., 2061., 1116.],
#     [0., 3069., 961.],
#     [0., 3069., 909.],
#     [0., 1342., 2637.],
#     [0., 1342., 2704.],
#     [0., 1319., 1810.],
#     [0., 1322., 1762.],
#     [0., 2684., 1706.],
#     [0., 2687., 1651.],
# ])

# queries = torch.tensor([
#     [0., 2086.0, 1705.0],
#     [0., 2059.0, 1646.0],
#     [0., 2684.0, 1700.0],
#     [0., 2684.0, 1646.0],
#     [0., 3071.0, 961.0],
#     [0., 3068.0, 907.0],
# ])

# queries = torch.tensor([
#     [0., 614.0, 1704.0],
#     [0., 614.0, 1645.0],
#     [0., 2687.0, 1711.0],
#     [0., 2694.0, 1656.0],
#     [0., 2074.0, 1704.0],
#     [0., 2071.0, 1649.0],
#     [0., 3608.0, 2056.0],
#     [0., 3614.0, 2104.0],
#     [0., 3937.0, 2060.0],
#     [0., 3930.0, 2111.0],
#     [0., 4138.0, 1601.0],
#     [0., 4138.0, 1546.0],
#     [0., 2064.0, 1164.0],
#     [0., 2064.0, 1123.0],
#     [0., 4745.0, 1021.0],
#     [0., 4742.0, 1076.0],
#     [0., 4723.0, 2636.0],
#     [0., 4729.0, 2577.0],
#     [0., 2892.0, 2636.0],
#     [0., 2895.0, 2585.0],
#     [0., 3055.0, 2581.0],
#     [0., 3061.0, 2632.0],
#     [0., 1860.0, 510.0],
#     [0., 1866.0, 558.0],
# ])


queries = torch.tensor([
    [0., 614.0, 1700.0],
    [0., 607.0, 1652.0],
    [0., 1323.0, 1810.0],
    [0., 1323.0, 1766.0],
    [0., 2067.0, 1700.0],
    [0., 2067.0, 1638.0],
    [0., 2684.0, 1707.0],
    [0., 2684.0, 1645.0],
    [0., 4128.0, 1542.0],
    [0., 4135.0, 1601.0],
    [0., 2489.0, 176.0],
    [0., 2476.0, 99.0],
    [0., 2658.0, 0.0],
    [0., 2665.0, 70.0],
    [0., 3946.0, 503.0],
    [0., 3949.0, 565.0],
    [0., 3866.0, 514.0],
    [0., 3863.0, 551.0],
    [0., 4537.0, 606.0],
    [0., 4544.0, 657.0],
    [0., 1860.0, 510.0],
    [0., 1863.0, 565.0],
    # [0., 2058.0, 1120.0],
    # [0., 2061.0, 1164.0],
    [0., 4742.0, 1021.0],
    [0., 4742.0, 1072.0],
    [0., 3924.0, 2060.0],
    [0., 3937.0, 2118.0],
    [0., 4726.0, 2588.0],
    [0., 4716.0, 2647.0],
    [0., 2889.0, 2574.0],
    [0., 2895.0, 2632.0],
    [0., 3048.0, 2577.0],
    [0., 3048.0, 2629.0],
    [0., 345.0, 3102.0],
    [0., 348.0, 3161.0],
    [0., 4780.0, 3242.0],
    [0., 4780.0, 3191.0],
])

if torch.cuda.is_available():
    queries = queries.cuda()

# pred_tracks, pred_visibility = model(video, queries=queries[None])
pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)



#  changed co-tracker/cotracker/utils/visualizer.py and linewidth
#     old visualizer file's draw_line function with linewidth 1: queries_backward_1.mp4
#                                                      2: queries_backward.mp4
#     new visualizer file's draw_line function with linewidth 3: queries_backward_2.mp4 - BEST    
    
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


'''
# works with stride 1 in co-tracker/models/build_cotracker.py

import os
import torch
import torch.nn.functional as F
import glob

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

# ↓ Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

video_path = "/l/users/malak.mansour/DEP/co-tracker/tif_attempts/tracked_frames/stitched_tif.mp4"
video = read_video_from_path(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()


# Normalize to [0, 1] range if not already
video = video / 255.0

# Increase brightness and contrast
brightness_factor = 0.2   # shift all pixel values up (try 0.1 to 0.3)
contrast_factor = 1.4     # scale contrast (try 1.1 to 1.5)

video = (video - 0.5) * contrast_factor + 0.5  # contrast
video = video + brightness_factor              # brightness

# Clip to valid range [0, 1] and rescale to [0, 255] for consistency
video = video.clamp(0, 1) * 255.0


# ↓ Resize video to reduce memory usage (adjust resolution as needed)
# Assume: video shape = (1, T, C, H, W)
B, T, C, H, W = video.shape

# Flatten time into batch
video = video.view(B * T, C, H, W)

# Resize each frame
video = F.interpolate(video, size=(720, 1280), mode='bilinear', align_corners=False)

# Restore shape
video = video.view(B, T, C, 720, 1280)

from cotracker.predictor import CoTrackerPredictor

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/scaled_offline.pth'
    )
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # clear cache to free up memory
    model = model.cuda()
    video = video.cuda()


queries = torch.tensor([
    [0., 614.0, 1700.0],
    [0., 607.0, 1652.0],
    [0., 1323.0, 1810.0],
    [0., 1323.0, 1766.0],
    [0., 2067.0, 1700.0],
    [0., 2067.0, 1638.0],
    [0., 2684.0, 1707.0],
    [0., 2684.0, 1645.0],
    [0., 4128.0, 1542.0],
    [0., 4135.0, 1601.0],
    [0., 2489.0, 176.0],
    [0., 2476.0, 99.0],
    [0., 2658.0, 0.0],
    [0., 2665.0, 70.0],
    [0., 3946.0, 503.0],
    [0., 3949.0, 565.0],
    [0., 3866.0, 514.0],
    [0., 3863.0, 551.0],
    [0., 4537.0, 606.0],
    [0., 4544.0, 657.0],
    [0., 1860.0, 510.0],
    [0., 1863.0, 565.0],
    [0., 2058.0, 1120.0],
    [0., 2061.0, 1164.0],
    [0., 4742.0, 1021.0],
    [0., 4742.0, 1072.0],
    [0., 3924.0, 2060.0],
    [0., 3937.0, 2118.0],
    [0., 4726.0, 2588.0],
    [0., 4716.0, 2647.0],
    [0., 2889.0, 2574.0],
    [0., 2895.0, 2632.0],
    [0., 3048.0, 2577.0],
    [0., 3048.0, 2629.0],
    [0., 345.0, 3102.0],
    [0., 348.0, 3161.0],
    [0., 4780.0, 3242.0],
    [0., 4780.0, 3191.0],
])

if torch.cuda.is_available():
    queries = queries.cuda()

pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)

# Visualization
vis = Visualizer(
    save_dir='/l/users/malak.mansour/DEP/co-tracker/tif_attempts/tracked_frames/select_points',
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

'''