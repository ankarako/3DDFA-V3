import argparse
import os

import torch
from torchvision.utils import make_grid
import numpy as np
from face_box import face_box
from model.recon import face_model
from easydict import EasyDict
import cv2
from PIL import Image
from util.io import back_resize_ldms, back_resize_crop_img, plot_kpts
from tqdm import tqdm
import json

import mediapipe as mp
# mediapipe lmks indices
# https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

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

k_class_labels = {
    'leye': 1,
    'reye': 2,
    'leyebrow': 3,
    'reyebrow': 4,
    'nose': 5,
    'ulip': 6,
    'blip': 7,
    'face': 8
}
k_class_colors = {
    0: np.array([0, 0, 255]),
    1: np.array([0, 255, 0]),
    2: np.array([0, 255, 255]),
    3: np.array([255, 0, 0]),
    4: np.array([255, 0, 255]),
    5: np.array([255, 255, 0]),
    6: np.array([255, 255, 0]),
    7: np.array([255, 255, 255]),
}

class VideoPlayer:
    def __init__(
        self,
        url: str
    ):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print(f"Failed to read video file: {url}")
            self.cap = None
            return
    
    @property
    def nframes(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0
    
    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
    
    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0

    def get_frame(self, idx: int) -> torch.Tensor:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Failed to read the specified frame idx: {idx}")
            return torch.zeros([3, self.height, self.width])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(torch.float32) / 255
        return frame_tensor
    
    def frames(self, start: int=0, end: int=None, step: int=1):
        """
        Get an iterable of the video's frames
        """
        if not self.cap:
            return
        
        if end is None:
            end = self.nframes
        
        for idx in range(start, end, step):
            yield self.get_frame(idx)
    
    def __len__(self) -> int:
        return self.nframes - 1
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.get_frame(index)


def process_lmks(lmks, trans_params, img):
    """
    Interpolate landmarks on image dimensions

    :param lmks The landmarks to process
    :param trans_params transformation parameters from face_box
    :param img The frame as a Pillow.Image
    :return The processed landmarks, Image plotted landmarks
    """
    lmks[:, 1] = 224 - 1 - lmks[:, 1]
    if trans_params is not None:
        lmks = back_resize_ldms(lmks, trans_params)
    img_lmk = plot_kpts(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), lmks)
    return lmks, img_lmk

def process_seg(seg, img, trans_params):
    img = np.asarray(img)
    new_seg = np.zeros([img.shape[0], img.shape[1], 8])
    for i in range(8):
        temp = np.stack([seg[:, :, i], ] * 3, axis=-1)
        temp = back_resize_crop_img((temp).astype(np.uint8), trans_params, np.zeros_like(img), resample_method=Image.NEAREST)[:, :, ::-1]
        new_seg[:, :, i] = temp[:, :, 0] * 255

        temp2 = img.copy()
        temp2[new_seg[:, :, i] == 255] = np.array([200, 200, 100])
    return new_seg

def seg_to_png(seg):
    output = np.zeros([seg.shape[0], seg.shape[1], 3])
    seg_viz = np.zeros_like(output)
    for channel in range(8):
        output[seg[..., channel] == 255] = channel + 1
        seg_viz[seg[..., channel] == 255] = k_class_colors[channel]
    output = output.astype(np.uint8)
    seg_viz = seg_viz.astype(np.uint8)
    return output, seg_viz

def make_proc_grid(img, img_lmks68, img_lmks106, img_lmks1062d, img_lmks134, seg, seg_visible) -> np.ndarray:
    """
    Put all the specified images in an image grid

    :param img The original image
    :param img_lmks... The images with the landmarks plotted
    :param seg The segmentation image
    :param seg_visible
    :return An image with the specified images in a grid
    """
    # convert every image to tensor
    frame_im = np.asarray(img)[:, :, [2, 1, 0]]
    frame_im = torch.from_numpy(frame_im).permute(2, 0, 1).to(torch.float32) / 255.0
    img_lmks68 = torch.from_numpy(img_lmks68).permute(2, 0, 1).to(torch.float32) / 255.0
    img_lmks106 = torch.from_numpy(img_lmks106).permute(2, 0, 1).to(torch.float32) / 255.0
    img_lmks1062d = torch.from_numpy(img_lmks1062d).permute(2, 0, 1).to(torch.float32) / 255.0
    img_lmks134 = torch.from_numpy(img_lmks134).permute(2, 0, 1).to(torch.float32) / 255.0
    seg = torch.from_numpy(seg).permute(2, 0, 1).to(torch.float32) / 255
    seg_vis = torch.from_numpy(seg_visible).permute(2, 0, 1).to(torch.float32) / 255

    grid = make_grid([frame_im, img_lmks68, img_lmks106, img_lmks1062d, img_lmks134, seg, seg_vis])
    grid = (grid.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return grid


class EyeTracking:
    def __init__(self):
        """
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.eye_indices = [473, 468]
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        """
        results = self.face_mesh.process(img)
        if not results.multi_face_landmarks:
            return None
        else:
            eye_lmks = []
            # convert lmks to tensor
            lmks = results.multi_face_landmarks[0]
            for idx in self.eye_indices:
                lm = lmks.landmark[idx]

                # convert to image coordinates
                x_px = lm.x * img.shape[0]
                y_px = lm.y * img.shape[1]
                z_rel = lm.z

                eye_lmks.append([x_px, y_px, z_rel])
            eye_lmks = np.array(eye_lmks)
            return eye_lmks

k_colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

def draw_lmks_2d(img: np.ndarray, lmks: np.ndarray, color: str) -> torch.Tensor:
    """
    Draw the specified landmarks on the given image

    :param img The image on which to draw the landmarks
    :param lmks The landmarks to draw
    :param color The color to use
    """
    # convert to numpy
    for lmk in lmks:
        img = cv2.circle(img, (round(lmk[0].item()), round(lmk[1].item())), 2, color=k_colors[color], thickness=-1)
    # convert to torch again
    return img

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

    print("Initializing modules...")
    recon_model = face_model(k_model_args)
    facebox_detector = face_box(k_facebox_args).detector
    video_player = VideoPlayer(input_video_path)
    eye_tracking = EyeTracking()

    height, width = video_player.height, video_player.width
    print("Modules Initialized.")

    k_output_video_frames = []


    device = torch.device(k_device)
    video_loop = tqdm(enumerate(video_player), total=video_player.nframes, desc="Processing video")
    for idx, frame in video_loop:
        # create output directory and check if output
        # data exist so we can skip them
        output_path = os.path.join(output_dir, f"frame-{idx}")
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_frame_filepath = os.path.join(output_path, f"frame-{idx}.png")
        output_seg_path = os.path.join(output_path, f"seg-{idx}.png")
        output_seg_visible_path = os.path.join(output_path, f'seg-vis-{idx}.png')
        output_lmk_path = os.path.join(output_path, f"lmks-{idx}.json")
        
        out_frame_exists = os.path.exists(output_frame_filepath)
        out_seg_exists = os.path.exists(output_seg_path)
        out_lmk_data_exists = os.path.exists(output_lmk_path)
        if out_frame_exists and out_seg_exists and out_lmk_data_exists:
            continue

        # convert frame to Pillow.Image
        frame_im = (frame.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_im = Image.fromarray(frame_im).convert('RGB')


        trans_params, im_tensor = facebox_detector(frame_im)
        recon_model.input_img = im_tensor.to(device)
        results = recon_model.forward()

        lmks_68 = results['ldm68'][0]
        lmks_106 = results['ldm106'][0]
        lmks_1062d = results['ldm106_2d'][0]
        lmks_134 = results['ldm134'][0]
        seg = results['seg']
        seg_visible = results['seg_visible']

        # infer eye landmarks with mediapipe
        eye_lmks = eye_tracking.process(np.asarray(frame_im))

        # interpolate landmarks and segmentations to video dimensions
        lmks68, img_lmks68 = process_lmks(lmks_68, trans_params, frame_im)
        lmks106, img_lmks106 = process_lmks(lmks_106, trans_params, frame_im)
        lmks1062d, img_lmks1062d = process_lmks(lmks_1062d, trans_params, frame_im)
        lmks134, img_lmks134 = process_lmks(lmks_1062d, trans_params, frame_im)
        
        seg = process_seg(seg, frame_im, trans_params)
        seg_visible = process_seg(seg_visible, frame_im, trans_params)

        # save outputs
        output_lmk_dict = {
            'lmks68': lmks68.tolist(),
            'lmks106': lmks106.tolist(),
            'lmks1062d': lmks1062d.tolist(),
            'lmks134': lmks134.tolist(),
            'eyes': eye_lmks.tolist()
        }
        with open(output_lmk_path, 'w') as outfd:
            json.dump(output_lmk_dict, outfd)
        
        # save output frame
        frame_im.save(output_frame_filepath)
        seg, seg_viz = seg_to_png(seg)
        seg_visible, seg_visible_viz = seg_to_png(seg_visible)
        
        seg_im = Image.fromarray(seg)
        seg_im.save(output_seg_path)
        seg_vis_im = Image.fromarray(seg_visible)
        seg_vis_im.save(output_seg_visible_path)

        # save lmk image
        img_lmks_face = draw_lmks_2d(np.asarray(frame_im).copy(), lmks68, 'green')
        img_lmks_face = draw_lmks_2d(img_lmks_face, eye_lmks[0:1], 'red') # red should be left
        img_lmks_face = draw_lmks_2d(img_lmks_face, eye_lmks[1:], 'blue') # blue should be right
        img_lmks = Image.fromarray(img_lmks_face)
        output_lmk_path = os.path.join(output_path, f'lmks-pred-{idx}.png')
        img_lmks.save(output_lmk_path)
        img_grid = make_proc_grid(frame_im, img_lmks68, img_lmks106, img_lmks1062d, img_lmks134, seg_viz, seg_visible_viz)
        k_output_video_frames += [img_grid]
    
    print("Outputing processed video")
    height, width = k_output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_proc_video_filepath = os.path.join(output_dir, 'video-processed.mp4')
    video_writer = cv2.VideoWriter(output_proc_video_filepath, fourcc, 30.0, (width, height))

    for frame in tqdm(k_output_video_frames, total=len(k_output_video_frames), desc="Exporting video"):
        video_writer.write(frame)
    video_writer.release()





