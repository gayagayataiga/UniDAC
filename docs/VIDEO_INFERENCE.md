# Video inference setup (UniDAC + GoPro Hero 11 Black Max Lens)

## やったこと

### 1. 動画一括推論スクリプトを新規作成
- ファイル: `demo/demo_video.py`
- 仕様: フォルダ内の `*.mp4`/`*.MP4` を全件読み込み、各動画につき1本の深度カラーマップ MP4 を `--out-dir` に書き出す
- `--save-raw` で **`<videoname>_depth_raw.npz` (uint16 mm) も保存**
  - 中身: `depth: (T, H, W) uint16`、`fps: float32`、`scale: float32 (=1000.0)`
  - 読込み: `np.load("..._depth_raw.npz")["depth"] / 1000.0` で float メートルに復元
- 主要 CLI: `--video-dir`, `--out-dir`, `--config-file`, `--depth-max`, `--stride`, `--limit`, `--skip-existing`, `--save-raw`

### 2. カメラパラメータを Max Lens Mode 用に同梱
- `demo/demo_video.py` 内に `build_sample_for_frame(image_bgr)` を定義
- 旧 Wide モード前提の単一フレームヘルパー（後に削除済 `demo_video_frame0.py`、commit 8a3d872 以前を参照）は流用せず別実装
- 採用値（`demo/grid_search_cam_params.py` の探索結果から選定）:
  - `k1=-0.20, k2=k3=k4=0`
  - `fl = 230 × (W / 640)` ← 640px幅で校正、入力解像度に応じてスケール（5.3K で fl≈1909）
  - `crop_wFoV=120`
  - `fwd_sz=(704, 704)`
  - `camera_model=OPENCV_FISHEYE`

### 3. デフォルト config を kitti360 (outdoor) に変更
- `demo/demo_video.py` のデフォルト `--config-file` を `configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json` に
- 理由: scannetpp (indoor) では grid search 全パターンで深度マップが一様潰れだった。kitti360 (outdoor, 広角魚眼向け) で明確な near/far 構造が出た
- デフォルト `--depth-max` を `3.0` に変更（実出力レンジ 0.25〜2.2m に整合）

### 4. grid search スクリプトを引数化
- `demo/grid_search_cam_params.py` の `CONFIG_FILE` ハードコードを `--config-file` で上書き可能に
- `fwd_sz` も config から自動取得
- 検証出力: `demo/sweeps/sweep_GX010086_lowfly/` （初期 `grid_ep204*` 出力は不要として削除済、commit 58dd941 以前を参照）

### 5. CLAUDE.md を新規作成
- リポジトリ構成・eval/demoコマンド・パイプラインの非自明な流れを記載

### 6. 推論実行
- スモーク 1: 旧パラメータで通った（後に誤と判明）
- スモーク 2: 5.3K + Max Lens + kitti360 で通った、深度に幾何構造あり
- **失敗した本実行 1**: 古いコードを読み込んだプロセス (1番) と新しいコードを読み込んだプロセス (2番) を同時に走らせてしまい、生成 mp4 が混在 → **全削除してやり直し**（下記「失敗と修正」参照）
- **クリーンな本実行（進行中）**: GPU 0 で `--stride 1 --save-raw --skip-existing` をバックグラウンド起動。約1.5〜2h想定。32本 / 86,694フレーム
  - ログ: `demo/output/old32_presetB/_run.log`
  - 出力: `demo/output/old32_presetB/<name>_depth.mp4` + `<name>_depth_raw.npz`

## 失敗と修正

最初に「進んで」と言われて起動したバックグラウンド run を **kill せずに放置したまま** `demo_video.py` を Max Lens 用に書き換えてしまい、Python は起動時のコードをキャッシュするため、放置されたプロセスは古いコード（scannetpp config + Wide params + depth_max=20）で動き続けた。後から正しいコードでもう1本起動したが、`--skip-existing` で先行プロセスが書いた誤推論 mp4 をスキップしてしまい、結果として `demo/output/old32_presetB/*_depth.mp4` のほぼ全てが古いパラメータ由来になっていた。

修正: 両プロセスを kill → 既存 mp4 と `_run.log` を全削除 → 新コード単一プロセスで `--save-raw` 付きで再実行。

**今後の教訓**: バックグラウンド推論ジョブを動かしたままスクリプトを編集しない。編集する場合は先にプロセスを kill する。

## 2026-05-12〜13 追加: HyperView 対応と Preset A 確定

### 経緯
最初の Preset B（旧パラメ）で 32 動画を推論 → 「歪んでる」「画像の一部しか使われてない」と判明。原因は2つ:
1. `k1=-0.20` が強すぎる: OPENCV_FISHEYE の `theta_d = theta*(1+k1*theta^2)` 効果でレンズ周辺角度が画像端に届かない（中央部しかサンプルされない）
2. **GoPro Hero 11 + Max Lens + HyperView は 8:7 ネイティブを 16:9 に横ストレッチするため `fl_x ≠ fl_y`** が必要だった（旧は isotropic 想定）

GX010012 / GX010086 frame 0 で `(fl_x, fl_y, k1, crop_wFoV)` を 4 段階に分けてスイープ:
1. `sweep_GX010012/` 128 通り (config × k1 × REF_FL × crop)
2. `sweep_GX010012_wide_crop/` 16 通り (crop 拡張)
3. `sweep_GX010012_wide2/` 16 通り (低 fl)
4. `sweep_GX010012_lowk1/` 12 通り (k1 ≈ 0)
5. `sweep_GX010012_highfl/` 24 通り (高 fl, k1≈0) → 横画角は埋まるが上下黒
6. `sweep_GX010086_hyperview/` 27 通り (`fl_x ≠ fl_y` 導入)
7. **`sweep_GX010086_lowfly/` 36 通り → 最終確定**

スイープ中の幾何解析メモ:
- ERP の θ_x=80° に対応する画像 u_offset = `fl_x × 1.40`。画像端 (W/2) と一致させるには `fl_x ≈ W/(2×1.40)` ≈ 966 @ W=2704 → REF_FL_x ≈ 1898（W=5312 換算）
- 同様に v_offset 制約から `fl_y ≤ 649 @ W=2704` → **REF_FL_y ≤ 1275**（ERP 四隅が image 内に収まる条件）

### 確定パラメータ (Preset A) — `demo/demo_video.py` の現在デフォルト
- camera_model: `OPENCV_FISHEYE`
- `fl_x = 1820 × (W/5312)`, `fl_y = 1275 × (W/5312)` ← 非対称
- `k1 = 0`, `k2 = k3 = k4 = 0`
- `crop_wFoV = 150`
- `fwd_sz = (512, 704)`（scannetpp config の値、横長アスペクト 1.375 が HyperView の FOV 比 ~1.35 に一致）
- config: `configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json`
- 可視化 `depth_max = 3.0`（実 depth p99 ≈ 2m）

旧パラメータは `Preset B` として保持。詳細比較は **`docs/PARAMS.md`**（subplot 画像埋め込み付き）。

### 新動画
- 新撮影: `/home/gayagaya/video/new/{GX010085.MP4, GX010086.MP4}` (2704×1520 @29.97fps、HyperView Max Lens)
- Preset A 推論結果: `demo/output/new_presetA/{GX010085, GX010086}_depth.mp4` + `_depth_raw.npz` + `_depth_dmax3.mp4`（再エンコ、depth_max=3）
- Preset B アーカイブ: `demo/output/new_presetB/`（比較用、両方 integrity OK）

### 公開 API
- `unidac/api.py::UniDACPipeline` を追加（Preset A デフォルト、`predict_frame(bgr)` / `predict_video(path)` / `cam_overrides=` 部分上書き対応）
- 使い方: `README.md` の "Python API" 節 + 詳細 `docs/API.md`

## やらなければいけないこと（残タスク）

### 優先度: 高
- [x] ~~クリーン本実行の完了確認（旧 32 動画 Preset B）~~ 完了
- [x] ~~出力の目視確認 → 歪み判明 → パラメータ刷新~~
- [x] ~~Preset A 確定~~（`sweep_GX010086_lowfly/sc_k1+0.00_flx1820_fly1275_crop150.jpg`）
- [x] ~~新動画 Preset A 推論 + 整合性チェック~~（`demo/output/new_presetA/`、mp4=npz, zero=0%）
- [x] ~~Preset B 新動画推論 + 整合性~~（`demo/output/new_presetB/`、mp4=npz, zero=5.3%）
- [ ] **旧 31 動画を Preset A で再推論**（4-GPU 並列計画あり、起動待ち）
  - **マシン: dl41**（RTX A6000 ×8、推論時に GPU 0–3 を使う）
  - 動画実体は 31 本（`/home/gayagaya/video/*.MP4`）、計 86,288 フレーム。doc 上の「32 本/86,694」は実物と微差あり
  - 単一 GPU: 3.9 fps → 約 6.2h。4 並列: 各 GPU 2–4 fps（CPU/IO 競合で ~25% 減）→ **推定 2.0–2.5h**
  - 既存の中途半端な出力 `demo/output/old32_presetA/`（旧 21〜22h 推定時の単一 GPU 起動の名残）は事前に rename or rm
  - 撮影モード（HyperView か Wide か Linear か）を確認できると Preset 選定の判断材料になるが、当面は Preset A 全適用で走らせる方針
  - シャード（LPT 貪欲法、各 ~21,500 フレーム、不均衡 0.7%）:
    - `s0` (GPU0, 7 本): GX030039, GX010014, GX010053, GX010007, GX010008, GX010042, GX010001
    - `s1` (GPU1, 7 本): GX020039, GX010009, GX010017, GX010011, GX010032, GX010037, GX010052
    - `s2` (GPU2, 7 本): GX010039, GX010010, GX010033, GX010005, GX010040, GX010004, GX010038
    - `s3` (GPU3, 10 本): GX040039, GX010003, GX010015, GX010034, GX010012, GX010013, GX010006, GX010016, GX010043, GX010041
  - 起動手順（シャード dir はシンボリックリンクで作成）:
    ```bash
    # 1. シャード dir 作成
    OUT=demo/output/old32_presetA
    mkdir -p $OUT/shards/s{0,1,2,3}
    for v in GX030039 GX010014 GX010053 GX010007 GX010008 GX010042 GX010001; do
      ln -sf /home/gayagaya/video/$v.MP4 $OUT/shards/s0/; done
    for v in GX020039 GX010009 GX010017 GX010011 GX010032 GX010037 GX010052; do
      ln -sf /home/gayagaya/video/$v.MP4 $OUT/shards/s1/; done
    for v in GX010039 GX010010 GX010033 GX010005 GX010040 GX010004 GX010038; do
      ln -sf /home/gayagaya/video/$v.MP4 $OUT/shards/s2/; done
    for v in GX040039 GX010003 GX010015 GX010034 GX010012 GX010013 GX010006 GX010016 GX010043 GX010041; do
      ln -sf /home/gayagaya/video/$v.MP4 $OUT/shards/s3/; done

    # 2. 4 GPU 並列起動（nohup でログアウト耐性つき）
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate unidac
    export PYTHONPATH=$PWD
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i nohup python demo/demo_video.py \
        --video-dir $OUT/shards/s$i --out-dir $OUT/s$i \
        --stride 1 --save-raw --skip-existing \
        > $OUT/s$i.log 2>&1 &
    done
    ```
  - Dry-run 結果（2026-05-13, dl41）:
    - 単一 GPU: GX010052 (79f) を 25.7s → 3.07 fps（モデルロード除けば ~3.9 fps）
    - 4 並列（GX010052, GX010001, GX010038, GX010032 を別 GPU）: wall 190s、各 GPU 1.86–3.85 fps、出力 mp4 と `_depth_raw.npz` 正常生成
    - environment OK（unidac env, timm import OK, dl41 GPU 0–4,6 空き）

### 優先度: 中
- [ ] **シーン依存の `depth_max` 自動調整**（`--auto-depth-max`）
  - 現状 visualizer は手で `8.0` → `3.0` に振り直し
  - 各 npz の p95-99 から自動決定
- [ ] `cam_overrides` を JSON ファイル経由で渡せる CLI フラグ（`--cam-params *.json`）

### 優先度: 低（最小構成の範囲外、必要になったら）
- [ ] **出力解像度を入力に揃える**（Plan A）: `cv2.resize(bgr, (src_w, src_h))` を1行
- [ ] **RGB と幾何的に一致する深度マップ**（Plan B）: ERP→cam 逆warp、`unidac/utils/erp_geometry.py::erp_patch_to_cam_fast` 要調査
- [ ] **ERP warp の前計算で高速化**: 同じ (W, H, cam_params) なら毎フレーム同じ warp。`cv2.remap` テーブル化
- [ ] **fp16 / batching**: `torch.autocast` 化、複数フレームの GPU バッチ
- [ ] **再開時のフレーム単位レジューム**: 現状 `--skip-existing` は動画単位スキップのみ
- [ ] k2, k4 を含めた歪み係数の追い込み（チェッカーボード校正のほうが王道）

## 参考

- 旧動画 (32 本、5.3K): `/home/gayagaya/video/*.MP4` (5312×2988, 計 86,694 フレーム)
- 新動画 (HyperView 設定で撮り直し): `/home/gayagaya/video/new/{GX010085, GX010086}.MP4` (2704×1520)
- リサイズ版: `/home/gayagaya/video/resized/{GX010013.MP4, episode_000204.mp4}` (640×360)
- Preset A 出力: `demo/output/new_presetA/`（入力 frame 0 jpg, Preset A/B subplot, mp4 ×2, depth_raw.npz ×2, depth_dmax3.mp4 ×2） — Preset 比較表は `docs/PARAMS.md`
- Preset B 出力（旧 32 動画）: `demo/output/old32_presetB/`
- Preset B 出力（新 2 動画、アーカイブ）: `demo/output/new_presetB/`
- sweep スクリプト: `demo/sweep_GX010012_frame0.py`、各 sweep 出力は `demo/sweeps/{sweep_*,grid_*}/`
- 公開 API: `unidac/api.py`、`docs/API.md`
- npz 読込みサンプル:
  ```python
  import numpy as np
  data = np.load("demo/output/new_presetA/GX010086_depth_raw.npz")
  depth_m = data["depth"].astype(np.float32) / data["scale"]  # (T, H, W) meters
  fps = float(data["fps"])
  ```
