import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from .optical_flow import OpticalFlow
from .point_tracking import PointTracker
from utils.torch import get_grid
import numpy as np
import matplotlib.pyplot as plt

class DenseOpticalTracker(nn.Module):
    def __init__(self,
                 height=512,
                 width=512,
                 tracker_config="configs/cotracker2_patch_4_wind_8.json",
                 tracker_path="checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                 estimator_config="configs/raft_patch_8.json",
                 estimator_path="checkpoints/cvo_raft_patch_8.pth",
                 refiner_config="configs/raft_patch_4_alpha.json",
                 refiner_path="checkpoints/movi_f_raft_patch_4_alpha.pth"):
        super().__init__()
        self.point_tracker = PointTracker(height, width, tracker_config, tracker_path, estimator_config, estimator_path)
        self.optical_flow_refiner = OpticalFlow(height, width, refiner_config, refiner_path)
        self.name = self.point_tracker.name + "_" + self.optical_flow_refiner.name
        self.resolution = [height, width]

    def forward(self, data, mode, u, v, **kwargs):
        if mode == "get_flow_frame_to_frame":
            return self.get_flow_frame_to_frame(data, u, v, **kwargs)
        if mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_flow_frame_to_frame(self, data, u, v, **kwargs):
        video = data["video"]
        T, C, h, w = data["video"].shape
        tracks = []
        H, W = self.resolution

        init = self.point_tracker(data, mode= "tracks_at_motion_boundaries" , **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        grid = get_grid(H, W, device=video.device)
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)

        data = {
            "src_frame": data["video"][:, u],
            "tgt_frame": data["video"][:, v],
            "src_points": init[:, u],
            "tgt_points": init[:, v]
        }
        pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
        pred["src_points"] = data["src_points"]
        pred["tgt_points"] = data["tgt_points"]
        flow, alpha = pred["flow"], pred["alpha"]
        flow = flow + grid

        return flow


    def get_flow_from_last_to_first_frame(self, data, **kwargs):
        video = data["video"]
        T, C, h, w = data["video"].shape
        tracks = []
        H, W = self.resolution

        init = self.point_tracker(data, mode= "tracks_at_motion_boundaries" , **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        grid = get_grid(H, W, device=video.device)
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)


        data = {
            "src_frame": data["video"][:, 0],
            "tgt_frame": data["video"][:, 1],
            "src_points": init[:, 0],
            "tgt_points": init[:, 1]
        }
        pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
        pred["src_points"] = data["src_points"]
        pred["tgt_points"] = data["tgt_points"]
        flow, alpha = pred["flow"], pred["alpha"]
        flow = flow + grid
        print(flow[0,:,:,0])  
        return flow
