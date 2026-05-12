# docs/ — index

| file | purpose |
|---|---|
| [API.md](API.md) | Public Python API (`unidac.api.UniDACPipeline`) — env setup, examples, preset reference, troubleshooting |
| [PARAMS.md](PARAMS.md) | Camera parameter presets (A vs B) with side-by-side ERP+depth subplots for each new video |
| [VIDEO_INFERENCE.md](VIDEO_INFERENCE.md) | Pipeline build log: HyperView discovery, parameter sweep history, Preset A finalization, remaining TODOs |
| [INFERENCE_MANIFEST.json](INFERENCE_MANIFEST.json) | Machine-readable catalog: every (video, preset, output mp4/npz, frame count) tuple |
| [DATA.md](DATA.md) | Upstream UniDAC dataset preparation (not modified) |
| pano_teaser.gif / pipeline.png | README assets |

## Quick links

- **Just want to call the model?** → [API.md §2 Minimal example](API.md#2-minimal-example)
- **What preset should I use?** → [PARAMS.md](PARAMS.md) (or [API.md §5 Presets](API.md#5-presets) for the short version)
- **Which outputs exist on disk?** → [INFERENCE_MANIFEST.json](INFERENCE_MANIFEST.json) (auto-generated, see `inferences[]`)
- **Why did we end up with Preset A?** → [VIDEO_INFERENCE.md §"HyperView 対応"](VIDEO_INFERENCE.md)
