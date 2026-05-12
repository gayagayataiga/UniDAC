"""Per-block timing for the demo_video.py inference pipeline.

Run on a free GPU separate from any production job:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python demo/profile_inference.py \
        --video /home/gayagaya/video/GX010001.MP4 --n-frames 60
"""
import argparse
import json
import os.path as osp
import statistics
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
from unidac.dataloaders.dataset import resize_for_input

from demo.demo_video import build_sample_for_frame


class Timer:
    def __init__(self, device):
        self.device = device
        self.records = {}

    def __call__(self, name):
        return _Block(self, name)

    def report(self, skip_warmup=5):
        print(f"\nPer-block timings (n={len(next(iter(self.records.values())))} samples, skipping first {skip_warmup}):")
        print(f"  {'block':<32s} {'mean (ms)':>12s} {'std (ms)':>12s} {'pct':>8s}")
        means = {}
        for name, vals in self.records.items():
            vals = vals[skip_warmup:]
            means[name] = statistics.mean(vals) * 1000
        total = sum(means.values())
        for name, vals in self.records.items():
            vals = vals[skip_warmup:]
            m = statistics.mean(vals) * 1000
            s = statistics.stdev(vals) * 1000 if len(vals) > 1 else 0.0
            pct = m / total * 100 if total > 0 else 0.0
            print(f"  {name:<32s} {m:>12.2f} {s:>12.2f} {pct:>7.1f}%")
        print(f"  {'TOTAL':<32s} {total:>12.2f}")
        print(f"  ≈ {1000.0 / total:.2f} f/s if all blocks serial\n")


class _Block:
    def __init__(self, timer, name):
        self.t = timer
        self.name = name

    def __enter__(self):
        if self.t.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, *a):
        if self.t.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        self.t.records.setdefault(self.name, []).append(dt)


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(args.config_file) as f:
        config = json.load(f)
    model = UniDAC.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device).eval()
    cano_sz = config["data"]["cano_sz"]

    timer = Timer(device)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    name = osp.splitext(osp.basename(args.video))[0]

    n_done = 0
    norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    while n_done < args.n_frames:
        with timer("a_read_decode"):
            ret, frame = cap.read()
            if not ret:
                break

        with timer("a_cvtcolor"):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sample = build_sample_for_frame(frame, dataset_name=name)
        fwd_sz = sample["fwd_sz"]
        crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
        crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

        H, W = image.shape[:2]
        depth = np.zeros((H, W, 1), dtype=np.float32)
        image_f = image.astype(np.float32) / 255.0
        mask_valid_depth = (depth > 0.01).astype(np.float32)

        with timer("b_cam_to_erp"):
            image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
                image_f, depth, mask_valid_depth,
                0, np.array(0).astype(np.float32),
                crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
                sample["cam_params"], np.array(0).astype(np.float32), scale_fac=None,
            )

        with timer("c_resize_for_input"):
            image_r, _depth_r, _pad, pred_scale_factor, attn_mask, lat_grid, _long_grid = resize_for_input(
                (image_erp * 255.0).astype(np.uint8), depth_erp, fwd_sz, None,
                [image_erp.shape[0], image_erp.shape[1]], 1.0,
                padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude,
            )

        with timer("d_to_tensor_gpu"):
            lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
            long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
            image_t = TF.normalize(TF.to_tensor(image_r), **norm).unsqueeze(0).to(device)
            attn_mask_t = TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0).to(device)
            lat_grid_t = torch.tensor(lat_grid).unsqueeze(0).to(device)
            lat_range_t = lat_range.unsqueeze(0).to(device)
            long_range_t = long_range.unsqueeze(0).to(device)

        with timer("e_model_forward"):
            with torch.no_grad():
                preds, _, _ = model(image_t, lat_range_t, long_range_t,
                                    attn_mask=attn_mask_t, lat_grid=lat_grid_t)
            preds = preds * pred_scale_factor

        with timer("f_postprocess"):
            _ = preds[0, 0].detach().cpu().numpy()

        n_done += 1
        if n_done % 10 == 0:
            print(f"  [{n_done}/{args.n_frames}]")

    cap.release()
    timer.report(skip_warmup=args.skip_warmup)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/home/gayagaya/video/GX010001.MP4")
    p.add_argument("--config-file", default="configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json")
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--skip-warmup", type=int, default=5)
    args = p.parse_args()
    main(args)
