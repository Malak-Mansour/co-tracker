'''
pip install mediapy tensorflow-cpu==2.12.0 tensorflow-datasets==4.9.2

cd ..
git clone https://github.com/google-deepmind/tapnet.git
cd tapnet
pip install -e .
cd ../co-tracker
'''


import os
import torch
import glob
import numpy as np
from torchvision.io import read_video
from cotracker.datasets.utils import CoTrackerData


class RealDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        random_frame_rate=False,
        random_seq_len=False,
        limit_samples=10000,
    ):
        super(RealDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        video_dir = "/path/to/videos"
        annotation_dir = "/path/to/annotations"
        self.video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        self.annotation_files = sorted(glob.glob(os.path.join(annotation_dir, "*.csv")))

        assert len(self.video_files) == len(self.annotation_files), "Mismatch between videos and CSVs"
        self.traj_per_sample = traj_per_sample
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.random_frame_rate = random_frame_rate
        self.random_seq_len = random_seq_len

    def crop(self, rgbs):
        S = len(rgbs)
        H, W = rgbs.shape[2:]
        y0 = 0 if self.crop_size[0] >= H else np.random.randint(0, H - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W else np.random.randint(0, W - self.crop_size[1])
        rgbs = [rgb[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for rgb in rgbs]
        return torch.stack(rgbs)

    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            sample = CoTrackerData(
                video=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1])),
                trajectory=torch.ones(1, 1, 1, 2),
                visibility=torch.ones(1, 1, 1),
                valid=torch.ones(1, 1, 1),
            )
        return sample, gotit

    def getitem_helper(self, index):
        video_path = self.video_files[index]
        annotation_path = self.annotation_files[index]

        rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
        if rgbs.numel() == 0:
            return None, False

        video = self.crop(rgbs)
        seq_len = len(video)

        coords = np.loadtxt(annotation_path, delimiter=",", skiprows=1)
        frames = coords[:, 0].astype(int)
        points = coords[:, 1:3]

        traj_count = len(points)
        trajectory = torch.zeros(seq_len, traj_count, 2)
        visibility = torch.zeros(seq_len, traj_count)
        valid = torch.zeros(seq_len, traj_count)

        for i, frame in enumerate(frames):
            if frame >= seq_len:
                continue
            trajectory[frame, i, :] = torch.tensor(points[i])
            visibility[frame, i] = 1.0
            valid[frame, i] = 1.0

        sample = CoTrackerData(
            video=video,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid,
            seq_name=video_path,
        )
        return sample, True

    def __len__(self):
        return len(self.video_files)
