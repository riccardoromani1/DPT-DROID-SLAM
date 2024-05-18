import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
import cv2

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
        coarse_height, coarse_width = 4*height // model_args.patch_size, 4*width // model_args.patch_size
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
    

    def create_boolean_mask(self, image_shape, coordinates):
        """
        Create a boolean mask of given image shape with True at specified coordinates.

        Parameters:
        image_shape (tuple): Shape of the image (height, width).
        coordinates (numpy.ndarray): Array of (x, y) coordinates.

        Returns:
        numpy.ndarray: Boolean mask of shape (height, width).
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)

        for coord in coordinates:
            x,y = coord
            x = round(x)
            y = round(y)
            if 0 <= y < height and 0 <= x < width:
                mask[y, x] = True

        return mask


    def get_sobel_kernels(self):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        return kernel_x, kernel_y
    
    def gaussian_filter(size, sigma):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()
    
    def get_harris_R(self, img, k):
        sobel_kernel_x, sobel_kernel_y = self.get_sobel_kernels()
        # Compute gradients using Sobel operator
        grad_x = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_x)
        grad_y = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_y)
        #grad_y = signal.convolve2d(img, sobel_kernel_y, boundary="fill", mode="same")

        # Compute products of derivatives
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y

        # Apply Gaussian filter to the products of derivatives
        kernel_size = 3
        # Gaussian Kernel
        G = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])/16
        sigma = 1
        Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=G)
        Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=G)
        Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=G)

        # Compute the Harris response
        det_M = Ixx * Iyy - Ixy * Ixy
        trace_M = Ixx + Iyy
        harris_response = det_M - k * (trace_M * trace_M)
        return harris_response

    def get_harris_corners(self, image, R):

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(R > 1e-2))
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        return cv2.cornerSubPix(image, np.float32(centroids), (9,9), (-1,-1), criteria)



    def getcorners(self, data, k=0.08, threshold=0.035, **kwargs):
        src_frame, tgt_frame = data["src_frame"], data["tgt_frame"]
        B, _, H, W = src_frame.shape

        # Convert to grayscale if necessary (assuming input is RGB)
        if src_frame.shape[1] == 3:
            src_gray = 0.299 * src_frame[:, 0] + 0.587 * src_frame[:, 1] + 0.114 * src_frame[:, 2]
            tgt_gray = 0.299 * tgt_frame[:, 0] + 0.587 * tgt_frame[:, 1] + 0.114 * tgt_frame[:, 2]
        else:
            src_gray = src_frame
            tgt_gray = tgt_frame
        device = src_frame.device
        #convert to np.float32
        im = src_gray.detach().cpu().numpy()
        im = np.squeeze(im, 0)
        im = im.astype(np.float32)

        #normalize 
        im /= im.max()


        R = self.get_harris_R(im, k)
        corners = self.get_harris_corners(im, R)
        
        corner_mask = self.create_boolean_mask((H,W), corners)

        R = torch.from_numpy(R)
        R = torch.unsqueeze(R, 0).to(device)
        corner_mask = torch.from_numpy(corner_mask)
        corner_mask = torch.unsqueeze(corner_mask, 0).to(device)
        return {"corners": corner_mask, "harris_response": R}


   

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




