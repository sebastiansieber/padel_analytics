# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Computer-vision pipeline that turns a single fixed-camera padel match recording into analytics: per-frame player/ball/court-keypoint detections, a 2D top-down court projection drawn onto the output video, and a CSV of player positions, velocities and accelerations in meters.

## Commands

```bash
# Environment (README uses conda; a plain venv works too)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# PyTorch must be installed separately for your platform: https://pytorch.org/get-started/locally/

# Run the full inference pipeline (all settings come from config.py)
python main.py

# Streamlit dashboard over the collected data
streamlit run app.py

# Interactive court keypoint picker (writes FIXED_COURT_KEYPOINTS_LOAD_PATH)
python select_court_keypoints.py
```

There is no test suite, linter config, or CI workflow in this repo. `torch`/`torchvision` are deliberately not in `requirements.txt` (platform-specific install, see README); everything else imported is listed there.

`supervision` must stay pinned to 0.25.1: 0.24 removed the `frame_resolution_wh` argument of `PolygonZone` and 0.26+ removed `sv.ByteTrack`, which `PlayerTracker` depends on.

Model weights are not in the repo (`weights/` is gitignored). The Google Drive folder in README.md holds four of the five files; `ball_detection/TrackNet_best.pt` is missing there and must be taken from the official TrackNetV3 checkpoints (link in github.com/qaz812345/TrackNetV3 README; `seq_len=8`, `bg_mode=concat`, trained on badminton, so padel ball recall is mediocre). `gdown` works for both Drive links. Default batch sizes assume ~8GB VRAM; `Tracker.DEVICE` picks CUDA, then Apple MPS, then CPU. A 15 s 1080p/50 fps clip (749 frames) takes ~4 min end to end on an Apple Silicon Mac.

### Trying a new video

1. Cut a short clip (ffmpeg can range-seek a remote mp4 directly, no full download needed) into `videos/` (gitignored).
2. In `config.py`, point `INPUT_VIDEO_PATH` at it and set every `*_LOAD_PATH` to `None` so stale `cache/*.json` from a previous video is not reused.
3. Court keypoints: `main.py` pops an OpenCV window for 12 clicks, but this only works when all 12 points are inside the frame. For overhead club cameras (e.g. Playsight) the near baseline is usually below the frame edge, so run `python select_court_keypoints.py` instead: it pads the frame so off-frame points can be clicked, lets you skip points you cannot see, completes them from the court geometry via a homography and reports the per-point reprojection error (a few px is good). It writes the JSON list of 12 `[x, y]` pixel pairs in k1..k12 order (off-frame coordinates such as negative x or y > height are fine) to `FIXED_COURT_KEYPOINTS_LOAD_PATH`. The bundled court-keypoint YOLO model was trained on broadcast angles and does not detect this view.

## Architecture

### Configuration is a module of globals

`config.py` is `from config import *`-ed by `main.py` and `app.py`. Every path, batch size and load/save toggle lives there. There is no CLI argument parsing.

### Tracker abstraction (`trackers/tracker.py`)

Every detector subclasses `Tracker` and produces per-frame `Object` instances (`Players`, `Ball`, `Keypoints`, `PlayersKeypoints`). Key contract points:

- `__str__` must return a unique id (`"players_tracker"`, `"ball_tracker"`, …); the runner keys trackers by it.
- `object()` returns the `Object` subclass; `Object.from_json`/`serialize` are what make the JSON cache work.
- A tracker implements **either** `predict_frames` (consumes the whole frame generator, e.g. YOLO/ByteTrack) **or** `predict_sample` (gets `batch_size` frames at a time, e.g. TrackNet). The unused one must raise `NoPredictFrames` / `NoPredictSample`; `predict_and_update` catches the exception to pick the path.
- `video_info_post_init(video_info)` is called by the runner before inference for anything that depends on fps/resolution (e.g. ByteTrack).
- `load_path` / `save_path` give each tracker a JSON prediction cache. If `load_path` yields predictions, the runner **skips inference entirely** for that tracker, with no check that the cached frame count matches the video. Delete or `None` out the `./cache/*.json` entries in `config.py` when switching videos or you will get stale results or `IndexError` in `draw_and_collect_data`.

### Runner (`trackers/runner.py`)

`TrackingRunner.run()` does two passes over the video:
1. For each tracker in order: move to device, run over a fresh frame generator, move back to CPU, save predictions. Trackers are independent; nothing is fused during inference.
2. `draw_and_collect_data()`: re-reads the video, draws every tracker's prediction for frame *i* onto the frame, then hands `Players`/`Ball`/`Keypoints` to `ProjectedCourt` to draw the 2D mini-court overlay and push player positions into `DataAnalytics`. Note the deliberate `data_analytics.step(1)` after every frame and the trailing `frames[:-1]` trim.

### Court keypoints and homography (`analytics/projected_court.py`)

The pipeline assumes a fixed camera. `main.py` pops an OpenCV window asking for 12 court keypoints clicked in the order of the ASCII diagram at the top of `main.py` (also reproduced in `ProjectedCourtKeypoints`). That order matters: k1/k2 and k11/k12 are used to build the `PolygonZone` that filters person detections to on-court players, and the full 12 are the source points for `cv2.findHomography` against `ProjectedCourt`'s synthetic mini-court. Clicked keypoints can be persisted via `FIXED_COURT_KEYPOINTS_SAVE_PATH` and reloaded via `FIXED_COURT_KEYPOINTS_LOAD_PATH` to skip the UI.

When `KeypointsTracker` is given `fixed_keypoints_detection`, it returns the same `Keypoints` for every frame and the homography is computed once; otherwise it is recomputed per frame from the model (12, 18 or 22 keypoints supported).

Pixel-to-meter conversion is anchored on `BASE_LINE = 10` m (court width) from `constants/court_dimensions.py`; positions in `DataAnalytics` are in meters relative to the court center (k6–k7 midpoint).

### Data output (`analytics/data_analytics.py`)

`DataAnalytics` accumulates one `DataPoint` per frame with up to four `PlayerPosition`s (ids 1–4 only; others are dropped). `into_dataframe(fps)` produces the CSV columns `player{n}_x/_y` plus derived distance, velocity and acceleration per player.

### Ball tracker (`trackers/ball_tracker/`)

Vendored TrackNetV3-style code: `TrackNet` for heatmap detection on frame sequences plus `InpaintNet` to fill trajectory gaps, with a background-median image estimated from up to `BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM` frames. This is the only tracker using `predict_sample` batching.

## Known gaps

- `estimate_velocity.py` (`BallVelocityEstimator`, `ImpactType`, used by `app.py`) was deleted upstream in commit 293645c and restored here from upstream PR #27; keep it when merging upstream.
- `DataAnalytics` only keeps ByteTrack ids 1-4. When a player is occluded and re-identified, the new id (5, 6, ...) is silently dropped from the CSV, so per-player coverage falls below the frame count.
- The pose model (`players_keypoints_detection`) and court-keypoint model were trained on broadcast footage; on overhead club cameras pose is detected for only one or two players per frame.
- `ui.py` is an older standalone keypoint-selection script superseded by the UI built into `main.py` and by `select_court_keypoints.py`.
