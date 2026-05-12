import os
import os.path as osp
import json
import argparse
import glob

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
from unidac.utils.visualization import save_val_imgs_v2
from unidac.dataloaders.dataset import resize_for_input


def build_sample_for_frame(image_bgr, dataset_name="gopro_wide"):
    """GoPro Hero 11 Black, 5.3K/2.7K Wide, EIS HS on.

    Reference 5.3K Wide intrinsics (community-measured; Hero 11 Wide w/ EIS):
      W=5312, H=2988, fl_x=fl_y≈1820, cx≈W/2, cy≈H/2,
      k1≈-0.02, k2≈0.0, k3≈0.0, k4≈0.0  (distortion is mild due to EIS rectification)
    Other resolutions are scaled linearly.
    """
    H, W = image_bgr.shape[:2]
    REF_W, REF_FL = 5312.0, 1820.0
    fl = REF_FL * (W / REF_W)
    cam_params = {
        "dataset": dataset_name,
        "fl_x": float(fl),
        "fl_y": float(fl),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "k1": -0.02, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "camera_model": "OPENCV_FISHEYE",
    }
    # crop_wFoV slightly less than the lens HFOV (~122°) to avoid pulling in invalid edges.
    return {
        "config_file": "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json",
        "dataset_name": dataset_name,
        "fishey_grid": None,
        "crop_wFoV": 115,
        "fwd_sz": (512, 704),
        "erp": False,
        "cam_params": cam_params,
        "image_bgr": image_bgr,
    }


def run_inference(model, device, sample, cano_sz, out_dir, out_name):
    image = cv2.cvtColor(sample["image_bgr"], cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    depth = np.zeros((H, W), dtype=np.float32)
    image_f = image.astype(np.float32) / 255.0
    depth = np.expand_dims(depth, axis=2)
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
    gt_t = TF.to_tensor(depth_r).unsqueeze(0).to(device)
    mask_t = TF.to_tensor((depth_r > 0.01).astype(np.uint8)).unsqueeze(0).to(device)

    with torch.no_grad():
        preds, _, _ = model(image_t, lat_range_t, long_range_t, attn_mask=attn_mask_t, lat_grid=lat_grid_t)
    preds = preds * pred_scale_factor

    os.makedirs(out_dir, exist_ok=True)
    save_val_imgs_v2(
        0, preds[0], gt_t[0], image_t[0], out_name, out_dir,
        active_mask=attn_mask_t[0], valid_depth_mask=mask_t[0],
        depth_max=20.0, arel_max=0.5,
    )


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config_file = "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json"
    with open(config_file) as f:
        config = json.load(f)
    model = UniDAC.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device).eval()
    cano_sz = config["data"]["cano_sz"]

    videos = sorted(glob.glob(osp.join(args.video_dir, "*.MP4"))) + sorted(glob.glob(osp.join(args.video_dir, "*.mp4")))
    if args.limit > 0:
        videos = videos[: args.limit]
    print(f"Found {len(videos)} videos")

    for v in videos:
        cap = cv2.VideoCapture(v)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  skip (could not read): {v}")
            continue
        name = osp.splitext(osp.basename(v))[0]
        print(f"Processing {name} (frame 0, shape={frame.shape})")
        sample = build_sample_for_frame(frame, dataset_name=name)
        run_inference(model, device, sample, cano_sz, args.out_dir, f"{name}_frame0.jpg")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="/home/gayagaya/video")
    p.add_argument("--model-file", default="checkpoints/unidac.pt")
    p.add_argument("--out-dir", default="demo/output/old32_presetB")
    p.add_argument("--limit", type=int, default=1, help="limit number of videos (0 = all)")
    args = p.parse_args()
    main(args)
