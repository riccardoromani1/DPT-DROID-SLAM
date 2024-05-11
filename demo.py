import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    #cv2.imshow('image', image / 255.0)
    #cv2.waitKey(0)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        #h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        #w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = 480
        h1 = 320
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

    parser.add_argument("--name", type=str)
    parser.add_argument("--model", type=str, default="dot", choices=["dot", "of", "pt"])
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--aspect_ratio", type=float, default=1)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_tracks", type=int, default=60)
    parser.add_argument("--sim_tracks", type=int, default=60)
    parser.add_argument("--alpha_thresh", type=float, default=0.8)
        #parser.add_argument("--is_train", type=str2bool, nargs='?', const=True, default=False)

        # Parallelization
    parser.add_argument('--worker_idx', type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

        # Optical flow estimator
    parser.add_argument("--estimator_config", type=str, default="configs/raft_patch_8.json")
    parser.add_argument("--estimator_path", type=str, default="checkpoints/cvo_raft_patch_8.pth")
    parser.add_argument("--flow_mode", type=str, default="direct", choices=["direct", "chain", "warm_start"])

        # Optical flow refiner
    parser.add_argument("--refiner_config", type=str, default="configs/raft_patch_4_alpha.json")
    parser.add_argument("--refiner_path", type=str, default="checkpoints/movi_f_raft_patch_4_alpha.pth")

        # Point tracker
    parser.add_argument("--tracker_config", type=str, default="configs/cotracker2_patch_4_wind_8.json")
    parser.add_argument("--tracker_path", type=str, default="checkpoints/movi_f_cotracker2_patch_4_wind_8.pth")
    parser.add_argument("--sample_mode", type=str, default="all", choices=["all", "first", "last"])

        # Dense optical tracker
    parser.add_argument("--cell_size", type=int, default=1)
    parser.add_argument("--cell_time_steps", type=int, default=20)

        # Interpolation
    parser.add_argument("--interpolation_version", type=str, default="torch3d", choices=["torch3d", "torch"])
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
