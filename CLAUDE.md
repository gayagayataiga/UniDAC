# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UniDAC (CVPR 2026) — universal metric depth estimation across perspective, fisheye, and 360°/equirectangular (ERP) cameras, using a unified model trained only on perspective images. The core idea: project any camera's image into an ERP patch, run a DINOv3 + DPT backbone, predict a relative depth, then a separate scale head produces metric depth.

## Environment setup

```bash
conda create -n unidac python=3.10.18 -y
conda activate unidac
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"   # eval.sh also sets this
```

Checkpoint: download `unidac.pt` from HuggingFace (`girish1511/UniDAC`) into `checkpoints/`. Configs additionally reference `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (the backbone weights).

## Commands

**Evaluation** (reproduce paper numbers; `<domain>` ∈ `indoor|outdoor`, `<dataset>` ∈ `scannetpp|gv2|kitti360|kitti|nyu|nuscenes|ibims`; the pair selects `configs/test/dac_dinov3l+dpt_<domain>_test_<dataset>.json`):

```bash
bash eval.sh <domain> <dataset>
# expands to:
# python scripts/test.py --model-file ./checkpoints/unidac.pt --model-name UniDAC \
#                        --config-file <CFG> --base-path datasets
```

Add `--vis` to `scripts/test.py` to dump visualizations; `--out-dir` controls the destination (default `show_dirs/`). `--base-path` is the dataset root; per-dataset subdir comes from `config["data"]["data_root"]`.

**Demo** (no dataset needed; uses bundled images in `demo/input`, writes to `demo/output`):

```bash
python demo/demo_unidac.py --model-file checkpoints/unidac.pt
python demo/demo_video_frame0.py        # single-frame variant
python demo/grid_search_cam_params.py   # sweeps cam params when intrinsics are unknown
```

There is **no training script, no test suite, and no linter** in this repo. The README mentions `demo.sh` and `bash demo.sh` but no such file exists — run the Python demo scripts directly.

## Architecture

### Pipeline (the non-obvious flow)

For any input camera, inference is:

1. **Camera → ERP patch.** `unidac/utils/erp_geometry.py::cam_to_erp_patch_fast` warps the raw image (perspective, fisheye, MEI, OPENCV_FISHEYE, …) plus its depth into an equirectangular crop sized from `crop_wFoV` and `cano_sz` (canonical size, e.g. `[1400, 1400]`). True ERP inputs (Matterport3D, GibsonV2) skip this step.
2. **Resize to model input.** `unidac/dataloaders/dataset.py::resize_for_input` resizes to `fwd_sz` (e.g. `[704,704]` or `[512,1024]`) and returns `pred_scale_factor` plus latitude/longitude grids.
3. **Forward pass** (`unidac/models/unidac.py::UniDAC.forward`):
   - `pixel_encoder` is a DINOv3 ViT (`unidac/models/backbones/`); when `rope_lat_weight` is true it receives `lat_grid` for latitude-aware RoPE.
   - `pixel_decoder` is a DPT head producing relative depth from multi-scale features (`output_idx` in config picks which encoder blocks).
   - `scale_head` (`ScaleMapEstimation` in `scale_est.py` / `scale_est_dinov2.py`) takes the encoder's CLS token + normalized relative depth and predicts a per-pixel scale map; metric depth = `scale_map * (rel_depth / median(rel_depth))`.
4. **Output scaling.** The returned depth is multiplied by `pred_scale_factor` from step 2 before visualization.

The pair `(rel_loss, metric_loss)` is built from `unidac/optimization/losses.py` (SILog etc.) and is only used in training paths; for eval, only the prediction tensor matters.

### Configs

`configs/test/*.json` and `configs/train/*.json` are the single source of truth for input size (`fwd_sz`), canonical ERP size (`cano_sz`), backbone choice, weights paths, and which dataset class to instantiate (`config["data"]["val_dataset"]` is `eval`'d by name from `unidac.dataloaders`). Changing a model variant means editing/adding a config, not changing `scripts/test.py`.

The `--model-name` arg of `scripts/test.py` is `eval()`'d (`UniDAC`, historically `IDiscERP`/`IDisc`/etc.); `erp_mode` is inferred from whether `"ERP"` is in that name.

### Dataloaders

Each dataset has its own module in `unidac/dataloaders/` (e.g. `kitti360_erp.py`, `scannetpp_erp.py`, `m3d.py`). `*_online.py` variants do the cam→ERP warp on the fly during training; non-`online` variants assume pre-warped data. The val dataset class name in the config must match a symbol exported by `unidac/dataloaders/__init__.py`.

## Working tips specific to this repo

- The repo is `pip install -e .`-able via `pyproject.toml` (package `unidac`), but `eval.sh` deliberately uses `PYTHONPATH` instead — both work, don't fight it.
- When adding a new camera type, the work is almost entirely in `erp_geometry.py` (forward/back projection) and a new dataloader; the model itself is camera-agnostic once input is ERP.
- Visualization helpers live in `unidac/utils/visualization.py` (`save_val_imgs_v2` used by both demo and eval).
- Splits and fisheye ray grids are checked into `splits/`; demo samples reference these (e.g. `splits/kitti360/grid_fisheye_02.npy`).
- Training code is listed as "not yet released" in the README — there is no train entrypoint in `scripts/`.
