"""Grid search cam_params (k1, fl, crop_wFoV) on a single frame of ep204.

Run from /home/gayagaya/UniDAC.
"""
import os
import os.path as osp
import json
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
from unidac.dataloaders.dataset import resize_for_input


CONFIG_FILE = "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json"


def build_cam_params(W, H, fl, k1, dataset_name="ep204_grid"):
    return {
        "dataset": dataset_name,
        "fl_x": float(fl),
        "fl_y": float(fl),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "k1": float(k1), "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "camera_model": "OPENCV_FISHEYE",
    }


def infer_one(model, device, image_bgr, cam_params, crop_wfov, cano_sz, fwd_sz=(512, 704)):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    depth = np.zeros((H, W), dtype=np.float32)
    image_f = image.astype(np.float32) / 255.0
    depth = np.expand_dims(depth, axis=2)
    mask_valid_depth = depth > 0.01

    crop_width = int(cano_sz[0] * crop_wfov / 180)
    crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

    image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image_f, depth, (mask_valid_depth * 1.0).astype(np.float32),
        0, np.array(0).astype(np.float32),
        crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
        cam_params, np.array(0).astype(np.float32), scale_fac=None,
    )
    lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
    long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

    image_r, depth_r, pad, pred_scale_factor, attn_mask, lat_grid, long_grid = resize_for_input(
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

    pred_np = preds[0].squeeze().cpu().numpy()
    mask_np = (attn_mask > 0).astype(np.float32)
    rgb_disp = image_r  # uint8, HxWx3, RGB
    return rgb_disp, pred_np, mask_np


def main(args):
    device = torch.device("cuda")

    with open(CONFIG_FILE) as f:
        config = json.load(f)
    model = UniDAC.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device).eval()
    cano_sz = config["data"]["cano_sz"]

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"could not read frame {args.frame} from {args.video}"
    H, W = frame.shape[:2]
    print(f"frame {args.frame}: {W}x{H}")

    k1_list = [-0.10, -0.20, -0.35]
    fl_list = [180.0, 230.0, 280.0]
    crop_list = [120, 150]

    cmap_depth = cm.magma_r
    depth_max = float(args.depth_max)

    os.makedirs(args.out_dir, exist_ok=True)

    for crop in crop_list:
        fig, axes = plt.subplots(len(k1_list), len(fl_list), figsize=(4 * len(fl_list), 3 * len(k1_list)))
        fig.suptitle(f"ep204 frame {args.frame} | crop_wFoV={crop}", fontsize=14)
        for i, k1 in enumerate(k1_list):
            for j, fl in enumerate(fl_list):
                cam = build_cam_params(W, H, fl, k1)
                rgb_disp, pred, mask = infer_one(model, device, frame, cam, crop, cano_sz)
                # show ERP rgb with depth overlay-style: side-by-side in same cell via hstack
                pred_vis = cmap_depth(np.clip(pred / depth_max, 0, 1))[..., :3]
                pred_vis = (pred_vis * 255).astype(np.uint8)
                # ensure same height
                vis = np.concatenate([rgb_disp, pred_vis], axis=1)
                ax = axes[i, j]
                ax.imshow(vis)
                ax.set_title(f"k1={k1:+.2f}, fl={fl:.0f}", fontsize=9)
                ax.axis("off")
                print(f"  done crop={crop} k1={k1} fl={fl}  depth=[{pred[mask>0].min():.2f},{pred[mask>0].max():.2f}]")
        plt.tight_layout()
        out_path = osp.join(args.out_dir, f"grid_crop{crop}.jpg")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/home/gayagaya/video/resized/episode_000204.mp4")
    p.add_argument("--frame", type=int, default=200)
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--out-dir", default="demo/output_video/grid_ep204")
    p.add_argument("--depth-max", type=float, default=5.0)
    args = p.parse_args()
    main(args)
