import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML


video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/stitched_tif.mp4"
video = read_video_from_path(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()



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

queries = torch.tensor([
    [0., 614.0, 1704.0],
    [0., 614.0, 1645.0],
    [0., 2687.0, 1711.0],
    [0., 2694.0, 1656.0],
    [0., 2074.0, 1704.0],
    [0., 2071.0, 1649.0],
    [0., 3608.0, 2056.0],
    [0., 3614.0, 2104.0],
    [0., 3937.0, 2060.0],
    [0., 3930.0, 2111.0],
    [0., 4138.0, 1601.0],
    [0., 4138.0, 1546.0],
    [0., 2064.0, 1164.0],
    [0., 2064.0, 1123.0],
    [0., 4745.0, 1021.0],
    [0., 4742.0, 1076.0],
    [0., 4723.0, 2636.0],
    [0., 4729.0, 2577.0],
    [0., 2892.0, 2636.0],
    [0., 2895.0, 2585.0],
    [0., 3055.0, 2581.0],
    [0., 3061.0, 2632.0],
    [0., 1860.0, 510.0],
    [0., 1866.0, 558.0],
])


if torch.cuda.is_available():
    queries = queries.cuda()

# pred_tracks, pred_visibility = model(video, queries=queries[None])
pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)



''' changed co-tracker/cotracker/utils/visualizer.py and linewidth
    old visualizer draw_line function with linewidth 1: queries_backward_1.mp4
                                                     2: queries_backward.mp4
    new visualizer draw_line function with linewidth 3: queries_backward_2.mp4 - BEST    
    '''
vis = Visualizer(
    save_dir='/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/select_points',
    linewidth=3, 
    mode='cool',
    tracks_leave_trace=-1
)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='queries_backward_2')


'''
# works with stride 1 in co-tracker/models/build_cotracker.py

import os
import torch
import torch.nn.functional as F

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

# ↓ Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

video_path = "/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/stitched_tif.mp4"
video = read_video_from_path(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

# ↓ Resize video to reduce memory usage (adjust resolution as needed)
video = F.interpolate(video, size=(720, 1280), mode='bilinear', align_corners=False)

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
    [0., 614.0, 1704.0],
    [0., 614.0, 1645.0],
    [0., 2687.0, 1711.0],
    [0., 2694.0, 1656.0],
    [0., 2074.0, 1704.0],
    [0., 2071.0, 1649.0],
    [0., 3608.0, 2056.0],
    [0., 3614.0, 2104.0],
    [0., 3937.0, 2060.0],
    [0., 3930.0, 2111.0],
    [0., 4138.0, 1601.0],
    [0., 4138.0, 1546.0],
    [0., 2064.0, 1164.0],
    [0., 2064.0, 1123.0],
    [0., 4745.0, 1021.0],
    [0., 4742.0, 1076.0],
    [0., 4723.0, 2636.0],
    [0., 4729.0, 2577.0],
    [0., 2892.0, 2636.0],
    [0., 2895.0, 2585.0],
    [0., 3055.0, 2581.0],
    [0., 3061.0, 2632.0],
    [0., 1860.0, 510.0],
    [0., 1866.0, 558.0],
])

if torch.cuda.is_available():
    queries = queries.cuda()

pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)

# Visualization
vis = Visualizer(
    save_dir='/home/malak.mansour/Downloads/DEP/co-tracker/tif_attempts/tracked_frames/select_points',
    linewidth=3, 
    mode='cool',
    tracks_leave_trace=-1
)

vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='queries_backward_2'
)

'''