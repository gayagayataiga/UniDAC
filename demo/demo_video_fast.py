"""Fast variant of demo_video.py: caches the cam->ERP warp per video.

Per-video, the (new_grid, mask_active) tensors and resize_for_input's pad/scale
factors are constant (since cam_params, fwd_sz, cano_sz, theta=phi=roll=0 are
all fixed). We precompute them once per video and reuse for every frame.

Validation goal: byte-identical depth output (in uint16 mm) vs demo_video.py.
"""
import argparse
import glob
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import make_cam_to_erp_grid, apply_cam_to_erp_grid
from unidac.dataloaders.dataset import resize_for_input

from demo.demo_video import build_sample_for_frame, depth_to_bgr


class FramePreprocessor:
    """Per-video reusable cam->ERP->fwd_sz pipeline.

    Constructed from the first frame's shape. All warp/resize params are
    cached; per-frame cost is just one GPU F.grid_sample + a CPU resize.
    """

    def __init__(self, first_frame_bgr, cano_sz, device):
        sample = build_sample_for_frame(first_frame_bgr, dataset_name="")
        self.cam_params = sample["cam_params"]
        self.fwd_sz = tuple(sample["fwd_sz"])
        self.crop_wFoV = sample["crop_wFoV"]
        self.cano_sz = list(cano_sz)
        self.device = device

        H, W = first_frame_bgr.shape[:2]
        self.img_h, self.img_w = H, W

        crop_width = int(cano_sz[0] * self.crop_wFoV / 180)
        crop_height = int(crop_width * self.fwd_sz[0] / self.fwd_sz[1])
        self.crop_h, self.crop_w = crop_height, crop_width

        # 1. Cache the cam -> ERP grid (geometry-only, frame-independent)
        new_grid, mask_active, lat_grid_np, lon_grid_np = make_cam_to_erp_grid(
            H, W, 0, np.array(0).astype(np.float32),
            crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
            self.cam_params, np.array(0).astype(np.float32), scale_fac=None,
        )
        self.new_grid = new_grid.to(device)
        self.mask_active = mask_active.to(device)
        self.erp_mask_np = mask_active[0, 0].cpu().numpy().astype(np.float32)
        self.latitude = lat_grid_np
        self.longitude = lon_grid_np

        # 2. Cache resize_for_input outputs that don't depend on the frame.
        #    We feed a zero RGB placeholder; we'll overwrite image_r per frame,
        #    but pad/pred_scale_factor/attn_mask/lat_grid are constants.
        dummy_rgb = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
        dummy_depth = np.zeros((crop_height, crop_width, 1), dtype=np.float32)
        _img_r, _depth_r, pad, pred_scale_factor, attn_mask, lat_grid_after, _long_after = resize_for_input(
            dummy_rgb, dummy_depth, self.fwd_sz, None,
            [crop_height, crop_width], 1.0,
            padding_rgb=[0, 0, 0], mask=self.erp_mask_np,
            lat_grid=self.latitude, long_grid=self.longitude,
        )
        self.pad = pad
        self.pred_scale_factor = pred_scale_factor
        self.attn_mask = attn_mask
        self.lat_grid_after = lat_grid_after

        # Pre-build the tensors that go into the model that are also frame-invariant
        self.lat_range = torch.tensor(
            [float(np.min(self.latitude)), float(np.max(self.latitude))]
        ).unsqueeze(0).to(device)
        self.long_range = torch.tensor(
            [float(np.min(self.longitude)), float(np.max(self.longitude))]
        ).unsqueeze(0).to(device)
        self.attn_mask_t = TF.to_tensor(
            (attn_mask > 0).astype(np.float32)
        ).unsqueeze(0).to(device)
        self.lat_grid_t = torch.tensor(lat_grid_after).unsqueeze(0).to(device)

        # Normalization constants on device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, model, frame_bgr):
        # frame_bgr -> RGB -> float32 in [0,1] -> (1,3,H,W) on device
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_t = torch.from_numpy(image).to(self.device, non_blocking=True)
        image_t = image_t.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1,3,H,W)

        # cached warp
        erp_img = apply_cam_to_erp_grid(image_t, self.new_grid, self.mask_active)
        # erp_img: (1,3,crop_h,crop_w) float32 in [0,1] (already masked)

        # bit-exact path to match demo_video.py: convert to uint8 numpy, then
        # cv2.resize+pad on CPU, then back to float on GPU.
        erp_np = erp_img[0].permute(1, 2, 0).cpu().numpy()  # H, W, 3 float
        erp_uint8 = (erp_np * 255.0).astype(np.uint8)
        dummy_depth = np.zeros((self.crop_h, self.crop_w, 1), dtype=np.float32)
        image_r, _depth_r, _pad, _scale, _mask, _lat, _long = resize_for_input(
            erp_uint8, dummy_depth, self.fwd_sz, None,
            [self.crop_h, self.crop_w], 1.0,
            padding_rgb=[0, 0, 0], mask=self.erp_mask_np,
            lat_grid=self.latitude, long_grid=self.longitude,
        )

        # Normalize on device
        image_in = torch.from_numpy(image_r).to(self.device, non_blocking=True)
        image_in = image_in.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_in = (image_in - self.mean) / self.std

        with torch.no_grad():
            preds, _, _ = model(image_in, self.lat_range, self.long_range,
                                attn_mask=self.attn_mask_t, lat_grid=self.lat_grid_t)
        preds = preds * self.pred_scale_factor
        return preds[0, 0].detach().cpu().numpy()


def process_video(model, device, cano_sz, video_path, out_mp4, out_npz,
                  depth_max, stride, save_raw):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  skip (cannot open): {video_path}")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = src_fps / max(stride, 1)

    ret, first = cap.read()
    if not ret:
        print(f"  skip (empty): {video_path}")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    preproc = FramePreprocessor(first, cano_sz, device)

    writer = None
    raw_frames = [] if save_raw else None
    name = osp.splitext(osp.basename(video_path))[0]
    pbar = tqdm(total=total, desc=name, unit="f", leave=False)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                depth = preproc(model, frame)
                if save_raw:
                    raw_frames.append(depth)
                bgr = depth_to_bgr(depth, depth_max)
                if writer is None:
                    h, w = bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_mp4, fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter: {out_mp4}")
                writer.write(bgr)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if save_raw and raw_frames:
            arr = np.stack(raw_frames, axis=0)
            arr_mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
            np.savez_compressed(out_npz, depth=arr_mm,
                                fps=np.float32(out_fps), scale=np.float32(1000.0))


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(args.config_file) as f:
        config = json.load(f)
    model = UniDAC.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device).eval()
    cano_sz = config["data"]["cano_sz"]

    videos = sorted(glob.glob(osp.join(args.video_dir, "*.MP4"))) + \
             sorted(glob.glob(osp.join(args.video_dir, "*.mp4")))
    if args.limit > 0:
        videos = videos[: args.limit]
    print(f"Found {len(videos)} videos")

    os.makedirs(args.out_dir, exist_ok=True)
    for v in videos:
        name = osp.splitext(osp.basename(v))[0]
        out_mp4 = osp.join(args.out_dir, f"{name}_depth.mp4")
        out_npz = osp.join(args.out_dir, f"{name}_depth_raw.npz")
        if args.skip_existing:
            need = (not osp.exists(out_mp4)) or (args.save_raw and not osp.exists(out_npz))
            if not need:
                print(f"  skip (all outputs exist): {name}")
                continue
        process_video(model, device, cano_sz, v, out_mp4, out_npz,
                      args.depth_max, args.stride, args.save_raw)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="/home/gayagaya/video")
    p.add_argument("--out-dir", default="demo/output_video_fast")
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--config-file", default="configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json")
    p.add_argument("--depth-max", type=float, default=3.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--save-raw", action="store_true")
    args = p.parse_args()
    main(args)
