import argparse
import os

import torch

from face_box import face_box
from model.recon import face_model
from easydict import EasyDict
import cv2

k_iscrop = True
k_detector = "retinaface"
k_ldm68 = True
k_ldm106_2d = True
k_ldm134 = True
k_seg = True
k_seg_visible = True
k_use_tex = True
k_extract_tex = True
k_backbone = "resnet50"
# default device is gpu
k_device = "cuda"

k_model_args = EasyDict(
    device=k_device,
    ldm68=k_ldm68,
    ldm106=k_ldm106_2d,
    ldm106_2d=k_ldm106_2d,
    ldm134=k_ldm134,
    seg_visible=k_seg_visible,
    seg=k_seg,
    backbone=k_backbone,
    extractTex=k_extract_tex
)

k_facebox_args = EasyDict(
    iscrop=k_iscrop,
    detector=k_detector,
    device=k_device
)


class VideoPlayer:
    def __init__(self, url: str=None):
        """
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fit 3DDFA v3 on a face video")
    parser.add_argument("--input_video", type=str, help="The path to the video to load.")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory")
    args = parser.parse_args()

    # sanity check input video
    input_video_path = args.input_video
    output_dir = args.output_dir
    if not os.path.exists(input_video_path):
        print(f"ERROR: The specified video path is invalid: {input_video_path}")
        exit(-1)

    # output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Creating model...")
    recon_model = face_model(k_model_args)
    facebox_detector = face_box(k_facebox_args).detector



