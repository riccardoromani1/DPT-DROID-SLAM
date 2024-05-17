import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math

from .shelf import RAFT
from .interpolation import interpolate
from utils.io import read_config
from utils.torch import get_grid, get_sobel_kernel


class OpticalFlow(nn.Module):
    def __init__(self, 
                 height, 
                 width, 
                 config="droid_slam/configs/raft_patch_4_alpha.json",
                 load_path="droid_slam/checkpoints/movi_f_raft_patch_4_alpha.pth"):
        super().__init__()
        model_args = read_config(config)
        model_dict = {"raft": RAFT}
        self.model = model_dict[model_args.name](model_args)
        self.name = model_args.name
        if load_path is not None:
            device = next(self.model.parameters()).device
            self.model.load_state_dict(torch.load(load_path, map_location=device))
        coarse_height, coarse_width = height // model_args.patch_size, width // model_args.patch_size
        self.register_buffer("coarse_grid", get_grid(coarse_height, coarse_width))

    def forward(self, data, mode, **kwargs):
        if mode == "harris_corner":
            return self.getcorners(data, **kwargs)
        if mode == "flow_with_tracks_init":
            return self.get_flow_with_tracks_init(data, **kwargs)
        elif mode == "motion_boundaries":
            return self.get_motion_boundaries(data, **kwargs)
        elif mode == "tracks_for_queries":
            return self.get_tracks_for_queries(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_motion_boundaries(self, data, boundaries_size=1, boundaries_dilation=4, boundaries_thresh=0.025, **kwargs):
        eps = 1e-12
        src_frame, tgt_frame = data["src_frame"], data["tgt_frame"]
        K = boundaries_size * 2 + 1
        D = boundaries_dilation
        B,_, H, W = src_frame.shape
        reflect = torch.nn.ReflectionPad2d(K // 2)
        sobel_kernel = get_sobel_kernel(K).to(src_frame.device)
        flow, _ = self.model(src_frame, tgt_frame)
        norm_flow = torch.stack([flow[..., 0] / (W - 1), flow[..., 1] / (H - 1)], dim=-1)
        flow_permuted = flow.permute(0, 3, 1, 2)  # Now the shape is [1, 2, 48, 72]

        # Step 2: Use interpolate to resize to the new height H and width W
        norm_flow = F.interpolate(flow_permuted, size=(H, W), mode='bilinear', align_corners=False)

        norm_flow = norm_flow.reshape(-1, 1, H, W)
        boundaries = F.conv2d(reflect(norm_flow), sobel_kernel)
        boundaries = ((boundaries ** 2).sum(dim=1, keepdim=True) + eps).sqrt()
        boundaries = boundaries.view(-1, 2, H, W).mean(dim=1, keepdim=True)
        if boundaries_dilation > 1:
            boundaries = torch.nn.functional.max_pool2d(boundaries, kernel_size=D * 2, stride=1, padding=D)
            boundaries = boundaries[:, :, -H:, -W:]
        boundaries = boundaries[:, 0]
        boundaries = boundaries - boundaries.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1)
        boundaries = boundaries / boundaries.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1)
        boundaries = boundaries > boundaries_thresh
        return {"motion_boundaries": boundaries, "flow": flow}
    
    def get_sobel_kernels(self):
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        return kernel_x.unsqueeze(0).unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0)

    

    def getcorners(self, data, k=0.04, threshold=0.01, **kwargs):
        src_frame, tgt_frame = data["src_frame"], data["tgt_frame"]
        B, _, H, W = src_frame.shape

        # Convert to grayscale if necessary (assuming input is RGB)
        if src_frame.shape[1] == 3:
            src_gray = 0.299 * src_frame[:, 0] + 0.587 * src_frame[:, 1] + 0.114 * src_frame[:, 2]
            tgt_gray = 0.299 * tgt_frame[:, 0] + 0.587 * tgt_frame[:, 1] + 0.114 * tgt_frame[:, 2]
        else:
            src_gray = src_frame
            tgt_gray = tgt_frame

        # Add channel dimension to grayscale images
        src_gray = src_gray.unsqueeze(1)  # Shape: (B, 1, H, W)

        # Get Sobel kernels
        sobel_kernel_x, sobel_kernel_y = self.get_sobel_kernels()
        device = src_frame.device
        sobel_kernel_x = sobel_kernel_x.to(device)
        sobel_kernel_y = sobel_kernel_y.to(device)

        # Compute gradients using Sobel operator
        grad_x = F.conv2d(src_gray, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(src_gray, sobel_kernel_y, padding=1)

        # Compute products of derivatives
        Ixx = grad_x *2
        Iyy = grad_y *2
        Ixy = grad_x * grad_y

        # Apply Gaussian filter to the products of derivatives
        kernel_size = 3
        sigma = 1
        Ixx = self.apply_gaussian_blur(Ixx, kernel_size=kernel_size, sigma=sigma)
        Iyy = self.apply_gaussian_blur(Iyy, kernel_size=kernel_size, sigma=sigma)
        Ixy = self.apply_gaussian_blur(Ixy, kernel_size=kernel_size, sigma=sigma)

        # Compute the Harris response
        det_M = Ixx * Iyy - Ixy *2
        trace_M = Ixx + Iyy
        harris_response = det_M - k * (trace_M *2)

        # Threshold the Harris response to detect corners
        harris_threshold = threshold * harris_response.max()
        corners = harris_response > harris_threshold
        corners = corners.squeeze(0)

        return {"corners": corners, "harris_response": harris_response}


    def get_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        # Create a 1D Gaussian kernel
        def gauss(x, sigma):
            return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * torch.exp(-0.5 * (x / sigma) ** 2)

        # Create range
        kernel_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        # Apply Gaussian formula
        kernel_1d = gauss(kernel_range, sigma)
        kernel_1d /= kernel_1d.sum()  # Normalize

        # Create 2D Gaussian kernel by outer product
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)

        return kernel_2d

    def apply_gaussian_blur(self, input, kernel_size=3, sigma=1.0):
        B, C, H, W = input.shape
        gaussian_kernel = self.get_gaussian_kernel(kernel_size, sigma).to(input.device)

        # Apply the Gaussian kernel to each channel
        blurred = F.conv2d(input, gaussian_kernel, padding=kernel_size // 2, groups=C)
        return blurred

    def get_flow_with_tracks_init(self, data, is_train=False, interpolation_version="torch3d", alpha_thresh=0.8, **kwargs):
        coarse_flow, coarse_alpha = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                version=interpolation_version)
        flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                 tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                 src_feats=data["src_feats"] if "src_feats" in data else None,
                                 tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                 coarse_flow=coarse_flow,
                                 coarse_alpha=coarse_alpha,
                                 is_train=is_train)
        if not is_train:
            alpha = (alpha > alpha_thresh).float()
        return {"flow": flow, "alpha": alpha, "coarse_flow": coarse_flow, "coarse_alpha": coarse_alpha}

    def get_tracks_for_queries(self, data, **kwargs):
        raise NotImplementedError




