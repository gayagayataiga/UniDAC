"""Run UniDAC depth inference on every video in a folder.

For each input video, writes one MP4 of the depth colormap to --out-dir at the
model's native output resolution (e.g. 512x704). RGB and raw depth are not saved.
"""
import os
import os.path as osp
import json
import glob
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
from unidac.dataloaders.dataset import resize_for_input

def build_sample_for_frame(image_bgr, dataset_name="gopro_maxlens_hyperview"):
    """GoPro Hero 11 Black, Max Lens Mode + HyperView (16:9 stretched).

    HyperView stretches the native 8:7 sensor capture horizontally, so the
    effective lens projection is anisotropic: fl_x > fl_y. Values picked from
    demo/output_video/sweep_GX010086_lowfly on GX010086 frame 0.
    """
    H, W = image_bgr.shape[:2]
    REF_W, REF_FLX, REF_FLY = 5312.0, 1820.0, 1275.0
    fl_x = REF_FLX * (W / REF_W)
    fl_y = REF_FLY * (W / REF_W)
    cam_params = {
        "dataset": dataset_name,
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "camera_model": "OPENCV_FISHEYE",
    }
    return {
        "fishey_grid": None,
        "crop_wFoV": 150,
        "fwd_sz": (512, 704),
        "erp": False,
        "cam_params": cam_params,
    }


def infer_depth(model, device, frame_bgr, sample, cano_sz):
    """Run a single frame through UniDAC and return metric depth as HxW np.float32."""
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    depth = np.zeros((H, W, 1), dtype=np.float32)
    image_f = image.astype(np.float32) / 255.0
    mask_valid_depth = depth > 0.01

    fwd_sz = sample["fwd_sz"]
    crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
    crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

    image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image_f, depth, (mask_valid_depth * 1.0).astype(np.float32),
        0, np.array(0).astype(np.float32),
        crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
        sample["cam_params"], np.array(0).astype(np.float32), scale_fac=None,
    )
    lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
    long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

    image_r, _depth_r, _pad, pred_scale_factor, attn_mask, lat_grid, _long_grid = resize_for_input(
        (image_erp * 255.0).astype(np.uint8), depth_erp, fwd_sz, None,
        [image_erp.shape[0], image_erp.shape[1]], 1.0,
        padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude,
    )

    norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    image_t = TF.normalize(TF.to_tensor(image_r), **norm).unsqueeze(0).to(device)
    attn_mask_t = TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0).to(device)
    lat_grid_t = torch.tensor(lat_grid).unsqueeze(0).to(device)
    lat_range_t = lat_range.unsqueeze(0).to(device)
    long_range_t = long_range.unsqueeze(0).to(device)

    with torch.no_grad():
        preds, _, _ = model(image_t, lat_range_t, long_range_t, attn_mask=attn_mask_t, lat_grid=lat_grid_t)
    preds = preds * pred_scale_factor
    return preds[0, 0].detach().cpu().numpy()


def depth_to_bgr(depth, depth_max):
    d = np.clip(depth, 0.0, depth_max) / depth_max
    d8 = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def process_video(model, device, cano_sz, video_path, out_mp4_path, out_npz_path,
                  depth_max, stride, save_raw):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  skip (cannot open): {video_path}")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = src_fps / max(stride, 1)

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
                sample = build_sample_for_frame(frame, dataset_name=name)
                depth = infer_depth(model, device, frame, sample, cano_sz)
                if save_raw:
                    raw_frames.append(depth)
                bgr = depth_to_bgr(depth, depth_max)
                if writer is None:
                    h, w = bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_mp4_path, fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter: {out_mp4_path}")
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
            np.savez_compressed(out_npz_path, depth=arr_mm, fps=np.float32(out_fps),
                                scale=np.float32(1000.0))


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
    print(f"Found {len(videos)} videos in {args.video_dir}")

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
    p.add_argument("--video-dir", default="/home/gayagaya/video/new")
    p.add_argument("--out-dir", default="demo/output_video_new")
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--config-file", default="configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json")
    p.add_argument("--depth-max", type=float, default=8.0)
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--limit", type=int, default=0, help="limit number of videos (0 = all)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--save-raw", action="store_true",
                   help="also write per-video <name>_depth_raw.npz (uint16 mm)")
    args = p.parse_args()
    main(args)
