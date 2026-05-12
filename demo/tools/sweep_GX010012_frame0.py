"""Sweep (config, k1, fl, crop_wFoV) on GX010012 frame 0, one subplot per combo.

Run from /home/gayagaya/UniDAC (or repo root). Outputs:
  demo/output/old32_presetB/sweep_GX010012/{cfg}_k1{k1}_fl{REF_FL}_crop{crop}.jpg
"""
import os
import os.path as osp
import json
import argparse
import itertools

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
from unidac.dataloaders.dataset import resize_for_input


REF_W = 5312.0  # user-provided REF_FL is calibrated for W=5312


def build_cam_params(W, H, fl, k1):
    return {
        "dataset": "sweep_GX010012",
        "fl_x": float(fl), "fl_y": float(fl),
        "cx": float(W / 2.0), "cy": float(H / 2.0),
        "k1": float(k1), "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "camera_model": "OPENCV_FISHEYE",
    }


def infer_one(model, device, image_bgr, cam_params, crop_wfov, cano_sz, fwd_sz):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    depth = np.zeros((H, W, 1), dtype=np.float32)
    image_f = image.astype(np.float32) / 255.0
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
    return image_r, pred_np, mask_np


def make_subplot(rgb_disp, pred, mask, depth_max, title, out_path):
    cmap = cm.magma_r
    pred_vis = cmap(np.clip(pred / depth_max, 0, 1))[..., :3]
    pred_vis = (pred_vis * 255).astype(np.uint8)
    # mask out invalid (black) pixels
    m = (mask > 0)[..., None]
    pred_vis = np.where(m, pred_vis, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb_disp); axes[0].set_title("ERP RGB"); axes[0].axis("off")
    im = axes[1].imshow(pred_vis); axes[1].set_title(f"depth (max={depth_max:.1f}m)"); axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"could not read frame {args.frame} from {args.video}"
    H, W = frame.shape[:2]
    print(f"frame {args.frame}: {W}x{H}")
    cv2.imwrite(osp.join(args.out_dir, "_input_frame0.jpg"), frame)

    configs = {
        "scannetpp": ("configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json", args.depth_max_indoor),
        "kitti360":  ("configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json", args.depth_max_outdoor),
    }

    k1_list = [-0.02, -0.05, -0.10, -0.20]
    ref_fl_list = [1620.0, 1820.0, 1909.0, 2000.0]
    crop_list = [110, 115, 120, 125]

    fl_list = [rf * (W / REF_W) for rf in ref_fl_list]
    total = len(configs) * len(k1_list) * len(ref_fl_list) * len(crop_list)
    print(f"sweep total = {total}")

    done = 0
    for cfg_name, (cfg_path, depth_max) in configs.items():
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = UniDAC.build(cfg)
        model.load_pretrained(args.model_file)
        model = model.to(device).eval()
        cano_sz = cfg["data"]["cano_sz"]
        fwd_sz = tuple(cfg["data"]["fwd_sz"])
        print(f"[{cfg_name}] fwd_sz={fwd_sz} cano_sz={cano_sz} depth_max={depth_max}")

        for k1, (ref_fl, fl), crop in itertools.product(k1_list, zip(ref_fl_list, fl_list), crop_list):
            tag = f"{cfg_name}_k1{k1:+.2f}_refFL{int(ref_fl)}_crop{crop}"
            out_path = osp.join(args.out_dir, f"{tag}.jpg")
            if args.skip_existing and osp.exists(out_path):
                done += 1
                continue
            cam = build_cam_params(W, H, fl, k1)
            try:
                rgb_disp, pred, mask = infer_one(model, device, frame, cam, crop, cano_sz, fwd_sz)
                title = f"{cfg_name} | k1={k1:+.2f} | REF_FL={ref_fl:.0f} (fl={fl:.0f}@W={W}) | crop_wFoV={crop}"
                make_subplot(rgb_disp, pred, mask, depth_max, title, out_path)
                dmin = float(pred[mask > 0].min()) if (mask > 0).any() else 0.0
                dmax = float(pred[mask > 0].max()) if (mask > 0).any() else 0.0
                done += 1
                print(f"[{done}/{total}] {tag}  depth=[{dmin:.2f},{dmax:.2f}]")
            except Exception as e:
                print(f"FAIL {tag}: {e}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/home/gayagaya/video/GX010012.MP4")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--out-dir", default="demo/output/old32_presetB/sweep_GX010012")
    p.add_argument("--depth-max-indoor", type=float, default=8.0)
    p.add_argument("--depth-max-outdoor", type=float, default=20.0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    main(args)
