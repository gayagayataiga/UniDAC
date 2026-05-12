# demo/output/ — inference results index

Each subdirectory holds depth outputs for one (video set, preset) pair. Filenames are uniform:

- `<name>_depth.mp4` — magma-colored depth video (visualization only, depth_max baked in)
- `<name>_depth_raw.npz` — raw depth, `uint16` millimeters: `depth.shape == (T, H, W)`, also stores `fps`, `scale (=1000)`
- `_run.log` — inference console log (tqdm + warnings)

## Subdirectories

| dir | preset | source videos | notes |
|---|---|---|---|
| [`old32_presetB/`](old32_presetB/) | B (legacy, isotropic Max Lens) | `/home/gayagaya/video/*.MP4` (32 clips, 5.3K, 86,694 frames) | original batch — left in place for archive but **incorrect for HyperView**; superseded by `old32_presetA/` once that run completes |
| [`new_presetA/`](new_presetA/) | **A (current default, HyperView anisotropic)** | `/home/gayagaya/video/new/{GX010085, GX010086}.MP4` | also contains `_input_frame0.jpg`, `_preset_{A,B}.jpg` subplots, `_qc_grid.jpg`, and `*_depth_dmax3.mp4` (re-encoded at depth_max=3) |
| [`new_presetB/`](new_presetB/) | B | same 2 new videos | comparison archive only |
| [`old32_presetA/`](old32_presetA/) | A | `/home/gayagaya/video/*.MP4` (31 clips, 5.3K) | **in-progress on dl41** via 4-GPU shards (`shards/s0..s3` → `s0..s3`); see [`docs/VIDEO_INFERENCE.md`](../../docs/VIDEO_INFERENCE.md) |

## Loading depth from npz

```python
import numpy as np
d = np.load("demo/output/new_presetA/GX010086_depth_raw.npz")
depth_m = d["depth"].astype(np.float32) / d["scale"]   # (T, H, W) float32 meters
fps = float(d["fps"])
```

## Preset parameters

See [`docs/PARAMS.md`](../../docs/PARAMS.md) for the side-by-side A vs B table with subplot images, or [`docs/INFERENCE_MANIFEST.json`](../../docs/INFERENCE_MANIFEST.json) for the machine-readable catalog of every output here.
