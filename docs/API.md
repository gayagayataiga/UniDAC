# UniDAC Python API

`unidac.api.UniDACPipeline` is a stateful, programmatic entrypoint for running UniDAC on single images or video files. Construct it once (loads the model onto GPU), then call `predict_frame` or `predict_video` repeatedly.

This document covers: environment setup, the smallest working example, every constructor argument, return-value shapes, and the camera-parameter presets.

---

## 1. Environment

```bash
git clone <repo>
cd UniDAC

conda create -n unidac python=3.10.18 -y
conda activate unidac

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Make the `unidac` package importable. Either:
export PYTHONPATH="$PWD:$PYTHONPATH"
# or (one-time):
pip install -e .
```

Checkpoints (download once, place under `checkpoints/`):

- `checkpoints/unidac.pt` — UniDAC weights from [huggingface.co/girish1511/UniDAC](https://huggingface.co/girish1511/UniDAC).
- `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` — DINOv3 backbone weights referenced by every config in `configs/test/`.

Sanity check:

```bash
python -c "from unidac.api import UniDACPipeline; print('ok')"
```

---

## 2. Minimal example

### Single image

```python
import cv2
from unidac.api import UniDACPipeline

pipe = UniDACPipeline()                       # model loaded onto CUDA (or CPU fallback)
bgr  = cv2.imread("my_image.jpg")
depth = pipe.predict_frame(bgr)               # np.ndarray, dtype=float32, shape=(H_out, W_out)
print(depth.min(), depth.max(), "meters")
```

`depth` is metric depth in **meters**. Its spatial size is the model's `fwd_sz` from the chosen config (e.g. `(512, 704)` for Preset A), **not** the input image resolution.

### Full video

```python
result = pipe.predict_video("clip.mp4", stride=1)
# result.depth: np.ndarray (T, H_out, W_out) float32 meters
# result.fps:   float (output fps; equals source_fps / stride)
# result.input_frame_count: int (frames in the source)
# result.stride: int (the stride you passed in)
```

`stride=1` processes every frame; `stride=2` every other; etc. Skipped frames are not in `result.depth`.

### Saving the depth map

The API only returns arrays — visualization and disk I/O are your responsibility. For a quick magma-colored mp4 + raw `uint16` mm npz, see `demo/demo_video.py::process_video` for the canonical pattern (or import `depth_to_bgr` from that module).

---

## 3. Constructor reference

```python
UniDACPipeline(
    preset: str = "A",
    model_file: str = "checkpoints/unidac.pt",
    device: Optional[str] = None,             # e.g. "cuda:1", "cpu". None = auto.
    cam_overrides: Optional[dict] = None,     # partial override of preset's cam params
    crop_wFoV: Optional[float] = None,        # override the preset's ERP crop width (degrees)
    config_file: Optional[str] = None,        # override the preset's UniDAC config (json)
)
```

`cam_overrides` keys (any subset, merged on top of the preset):

| key | meaning |
|---|---|
| `fl_x_ref`, `fl_y_ref` | focal lengths in pixels at the reference width `REF_W_NATIVE = 5312`. Actual `fl_*` used at inference = `ref * (W_input / 5312)`. |
| `k1`, `k2`, `k3`, `k4` | OpenCV fisheye distortion coefficients. |
| `camera_model` | one of the strings supported by `unidac.utils.erp_geometry.cam_to_erp_patch_fast` (`OPENCV_FISHEYE`, `OPENCV`, `MEI`, …). |

`fl_x_ref` and `fl_y_ref` use a fixed reference width of 5312 px so the same numeric values stay valid across resolutions — the pipeline rescales them per input frame internally.

---

## 4. Method reference

### `predict_frame(frame_bgr) -> np.ndarray`

- **Input**: `frame_bgr` — `np.uint8`, shape `(H, W, 3)`, BGR channel order (OpenCV native).
- **Output**: `np.float32`, shape `(H_out, W_out)`, metric depth in meters. Spatial size equals the config's `fwd_sz` (Preset A → `(512, 704)`).

Stateless w.r.t. the pipeline — safe to call concurrently with the same model only if you serialize on the GPU.

### `predict_video(video_path, stride=1) -> VideoResult`

- **Input**: path to any video OpenCV can decode (mp4, MOV, …).
- **Output**: dataclass `VideoResult` with `depth: (T, H_out, W_out) float32`, `fps: float`, `input_frame_count: int`, `stride: int`.

Memory: the full depth stack stays in RAM. A 1500-frame video at `(512, 704)` is `1500 * 512 * 704 * 4 B ≈ 2 GB`. For longer clips, call `predict_frame` in a loop and stream the result to disk yourself.

---

## 5. Presets

Both presets target a GoPro Hero 11 + Max Lens Mod, but differ on the assumed lens/sensor mapping.

| field | Preset A (default) | Preset B (legacy) |
|---|---|---|
| recommended for | **Max Lens + HyperView (8:7 → 16:9 stretched)** | older Max Lens isotropic assumption (incorrect for HyperView) |
| `fl_x_ref` | 1820 | 1909 |
| `fl_y_ref` | 1275 (anisotropic) | 1909 (isotropic) |
| `k1` | 0.0 | -0.20 |
| `crop_wFoV` | 150 | 120 |
| `fwd_sz` (from config) | (512, 704) | (704, 704) |
| config | `configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json` | `configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json` |
| tuned on | `demo/sweeps/sweep_GX010086_lowfly/` (GX010086 frame 0) | `demo/tools/grid_search_cam_params.py` on 640×360 episode_000204 |

```python
pipe = UniDACPipeline(preset="A")    # default
pipe = UniDACPipeline(preset="B")    # legacy

# Custom — start from A, tweak two values:
pipe = UniDACPipeline(
    preset="A",
    cam_overrides={"fl_y_ref": 1200.0, "k1": -0.02},
    crop_wFoV=145,
)
```

The numeric values themselves and the sweep trail that led to them are in `docs/VIDEO_INFERENCE.md` and `docs/PARAMS.md`.

---

## 6. What the API does *not* do

- **Calibration**: it does not derive `fl_x/fl_y/k1` from your footage. If the presets are wrong for your camera, calibrate (OpenCV chessboard) or run `demo/tools/sweep_GX010012_frame0.py` against your own video.
- **Depth visualization / I/O**: returns NumPy arrays only. Encode/save yourself (`demo/demo_video.py` is the reference).
- **Backprojection to the input camera**: the returned depth lives in the ERP-cropped frame, not the original fisheye pixel grid. Geometric re-alignment is on the roadmap (see `docs/VIDEO_INFERENCE.md` "Plan B").
- **Batched / fp16 inference**: each frame goes through the model independently in fp32. For high-throughput needs, wrap the model directly.

---

## 7. Troubleshooting

| symptom | likely cause |
|---|---|
| `ModuleNotFoundError: unidac` | `PYTHONPATH` not set, or `pip install -e .` not run, or wrong conda env active |
| `ModuleNotFoundError: timm` | conda env `unidac` not activated |
| `FileNotFoundError: checkpoints/...` | checkpoint files not downloaded into `checkpoints/` |
| ERP RGB looks cropped vs. input frame | `fl_x_ref` too small for your lens — try larger, or run the sweep scripts under `demo/` |
| Four-corner black in ERP | `fl_y_ref` too large for vertical FOV — lower it (Preset A's 1275 worked for HyperView) |
| Depth values look uniform / flat | check that the scene actually has depth variation in the ERP RGB; if so, `depth_max` in your visualization is too large (try 3.0 m for indoor close-range) |
