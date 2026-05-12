"""High-level Python API for UniDAC video / image depth inference.

Usage:
    from unidac.api import UniDACPipeline

    pipe = UniDACPipeline()                      # loads with Preset A defaults
    depth = pipe.predict_frame(bgr_image)        # (H, W) float32 metric depth
    result = pipe.predict_video("path/to/video.mp4")
    # result.depth: (T, H, W) float32 meters
    # result.fps: float
    # result.input_frame_count: int

Override parameters:
    pipe = UniDACPipeline(preset="B")            # legacy isotropic params
    pipe = UniDACPipeline(cam_overrides=dict(fl_x_ref=1820, fl_y_ref=1275, k1=0.0),
                          crop_wFoV=150)

The output depth's spatial size matches `fwd_sz` from the underlying config
(e.g. (512, 704) for Preset A), not the input video resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from unidac.dataloaders.dataset import resize_for_input
from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast


REF_W_NATIVE = 5312.0  # GoPro 5.3K native width; fl values are calibrated at this W


PRESETS = {
    # tuned via demo/sweeps/sweep_GX010086_lowfly for HyperView + Max Lens
    "A": dict(
        config="configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json",
        fl_x_ref=1820.0, fl_y_ref=1275.0,
        k1=0.0, k2=0.0, k3=0.0, k4=0.0,
        crop_wFoV=150,
        camera_model="OPENCV_FISHEYE",
    ),
    # original 32-video batch params (isotropic, kitti360 outdoor); legacy
    "B": dict(
        config="configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json",
        fl_x_ref=1909.0, fl_y_ref=1909.0,
        k1=-0.20, k2=0.0, k3=0.0, k4=0.0,
        crop_wFoV=120,
        camera_model="OPENCV_FISHEYE",
    ),
}

DEFAULT_MODEL_FILE = "checkpoints/unidac.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class VideoResult:
    depth: np.ndarray            # (T, H_out, W_out) float32 meters
    fps: float
    input_frame_count: int
    stride: int


class UniDACPipeline:
    """Stateful inference pipeline. Construct once, call predict_* many times."""

    def __init__(
        self,
        preset: str = "A",
        model_file: str = DEFAULT_MODEL_FILE,
        device: Optional[str] = None,
        cam_overrides: Optional[dict] = None,
        crop_wFoV: Optional[float] = None,
        config_file: Optional[str] = None,
    ):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; available: {list(PRESETS)}")
        p = dict(PRESETS[preset])
        if cam_overrides:
            p.update(cam_overrides)
        if crop_wFoV is not None:
            p["crop_wFoV"] = float(crop_wFoV)
        if config_file is not None:
            p["config"] = config_file
        self.params = p

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        with open(p["config"]) as f:
            cfg = json.load(f)
        self.cano_sz = cfg["data"]["cano_sz"]
        self.fwd_sz = tuple(cfg["data"]["fwd_sz"])

        model = UniDAC.build(cfg)
        model.load_pretrained(model_file)
        self.model = model.to(self.device).eval()

    def _cam_params(self, W: int, H: int) -> dict:
        scale = W / REF_W_NATIVE
        p = self.params
        return {
            "dataset": "unidac_api",
            "fl_x": float(p["fl_x_ref"] * scale),
            "fl_y": float(p["fl_y_ref"] * scale),
            "cx": float(W / 2.0), "cy": float(H / 2.0),
            "k1": float(p["k1"]), "k2": float(p["k2"]),
            "k3": float(p["k3"]), "k4": float(p["k4"]),
            "camera_model": p["camera_model"],
        }

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR image (H, W, 3) uint8 → metric depth (H_out, W_out) float32 in meters."""
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        depth_dummy = np.zeros((H, W, 1), dtype=np.float32)
        image_f = image.astype(np.float32) / 255.0
        mask_valid_depth = depth_dummy > 0.01

        cam_params = self._cam_params(W, H)
        crop_w = int(self.cano_sz[0] * self.params["crop_wFoV"] / 180)
        crop_h = int(crop_w * self.fwd_sz[0] / self.fwd_sz[1])

        image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image_f, depth_dummy, (mask_valid_depth * 1.0).astype(np.float32),
            0, np.array(0).astype(np.float32),
            crop_h, crop_w, self.cano_sz[0], self.cano_sz[0] * 2,
            cam_params, np.array(0).astype(np.float32), scale_fac=None,
        )
        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

        image_r, _, _, pred_scale_factor, attn_mask, lat_grid, _ = resize_for_input(
            (image_erp * 255.0).astype(np.uint8), depth_erp, self.fwd_sz, None,
            [image_erp.shape[0], image_erp.shape[1]], 1.0,
            padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude,
        )

        image_t = TF.normalize(TF.to_tensor(image_r), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(self.device)
        attn_t = TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0).to(self.device)
        lat_grid_t = torch.tensor(lat_grid).unsqueeze(0).to(self.device)
        lat_range_t = lat_range.unsqueeze(0).to(self.device)
        long_range_t = long_range.unsqueeze(0).to(self.device)

        preds, _, _ = self.model(image_t, lat_range_t, long_range_t,
                                 attn_mask=attn_t, lat_grid=lat_grid_t)
        preds = preds * pred_scale_factor
        return preds[0].squeeze().detach().cpu().numpy().astype(np.float32)

    def predict_video(self, video_path: str, stride: int = 1) -> VideoResult:
        """Process every `stride`-th frame; returns (T, H, W) float32 depth in meters."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_fps = fps / max(stride, 1)
        depths = []
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    depths.append(self.predict_frame(frame))
                idx += 1
        finally:
            cap.release()
        return VideoResult(
            depth=np.stack(depths, axis=0),
            fps=float(out_fps),
            input_frame_count=total,
            stride=stride,
        )


__all__ = ["UniDACPipeline", "VideoResult", "PRESETS"]
