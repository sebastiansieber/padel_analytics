"""
Interactive court keypoint selection with room to click outside the frame and
with geometry-based completion of points you cannot see.

The first frame of INPUT_VIDEO_PATH is shown with a grey border around it so
that keypoints lying outside the image (e.g. the near baseline on overhead club
cameras) can still be clicked. Click the keypoints in this order, pressing
`n` to skip any point you cannot see precisely (net floor line hidden under the
net, dark wall junctions, off-frame corners, ...):

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2

k1/k2 = near baseline, k11/k12 = far baseline, k6/k7 = net line on the floor.

As soon as 4 or more well-spread points are set, the remaining points are
computed from the official court dimensions via a homography and the full court
grid is drawn so you can judge the fit. The status line reports the
reprojection error of your clicked points (a few px is good; 15+ px means one
of the clicked points is off). Painted-line intersections (k3, k4, k5, k8, k9,
k10) are the most reliable; the far baseline corners (k11, k12) help if the
floor-to-wall junction is visible.

Once points exist you can drag any of them with the left mouse button; the
grid and fit error update live. Dragging a blue computed point pins it as a
clicked point; right-clicking a clicked point releases it to computed again.

Keys:  n = skip point   u = undo   r = reset   Enter/s = save   q/Esc = quit

If the output file already exists its 12 points are loaded as the starting
state so they can simply be dragged; pass --fresh to start empty.

Usage:
    python select_court_keypoints.py [--pad 200] [--max-width 1800 --max-height 1050 | --scale 0.75] [--out ...] [--fresh]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from config import INPUT_VIDEO_PATH, FIXED_COURT_KEYPOINTS_LOAD_PATH

LINES = [(1, 2), (3, 5), (6, 7), (8, 10), (11, 12), (1, 11), (2, 12), (4, 9)]

# Court coordinates in meters (x across the court, y along it), k1..k12
WORLD = np.array(
    [
        (-5, 0), (5, 0),
        (-5, 3), (0, 3), (5, 3),
        (-5, 10), (5, 10),
        (-5, 17), (0, 17), (5, 17),
        (-5, 20), (5, 20),
    ],
    np.float32,
)


def complete_points(
    points: list[Optional[tuple[float, float]]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fill in missing keypoints from the clicked ones via a court homography.

    Returns (all 12 points, reprojection error of the clicked points) or
    (None, None) when there are not enough well-spread clicked points yet.
    """
    idx = [i for i, p in enumerate(points) if p is not None]
    if len(idx) < 4:
        return None, None
    src = WORLD[idx]
    dst = np.array([points[i] for i in idx], np.float32)
    # need points spanning both court axes, otherwise the homography is degenerate
    if np.ptp(src[:, 0]) < 1 or np.ptp(src[:, 1]) < 1:
        return None, None
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return None, None
    all_pts = cv2.perspectiveTransform(WORLD.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(all_pts[idx] - dst, axis=1)
    return all_pts, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=INPUT_VIDEO_PATH)
    parser.add_argument("--pad", type=int, default=200, help="border in source pixels added around the frame")
    parser.add_argument("--scale", type=float, default=None,
                        help="display scale of the padded frame (default: fit --max-width x --max-height)")
    parser.add_argument("--max-width", type=int, default=1800, help="display width budget in pixels")
    parser.add_argument("--max-height", type=int, default=1050, help="display height budget in pixels")
    parser.add_argument(
        "--out",
        default=FIXED_COURT_KEYPOINTS_LOAD_PATH or "cache/fixed_keypoints_detection.json",
    )
    parser.add_argument("--fresh", action="store_true", help="start empty instead of loading the existing keypoints file")
    args = parser.parse_args()

    frame = next(sv.get_video_frames_generator(args.video, start=0, end=1))
    h, w = frame.shape[:2]
    pad = args.pad
    if args.scale is None:
        args.scale = min(args.max_width / (w + 2 * pad), args.max_height / (h + 2 * pad), 1.0)
    print(f"display scale {args.scale:.2f}, window {int((w + 2 * pad) * args.scale)}x{int((h + 2 * pad) * args.scale)}")
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 90, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = frame
    cv2.rectangle(canvas, (pad, pad), (pad + w - 1, pad + h - 1), (0, 255, 255), 2)

    # one slot per keypoint: a clicked (x, y) in source pixels, or None if skipped
    points: list[Optional[tuple[float, float]]] = []
    if not args.fresh and Path(args.out).is_file():
        with open(args.out) as f:
            loaded = json.load(f)
        if len(loaded) == 12:
            points = [(float(x), float(y)) for x, y in loaded]
            print(f"loaded 12 keypoints from {args.out}; drag to adjust, or run with --fresh to start over")

    def to_canvas(p) -> tuple[int, int]:
        return int(p[0] + pad), int(p[1] + pad)

    def render() -> np.ndarray:
        img = canvas.copy()
        full, err = complete_points(points)
        if full is not None:
            # thin orange grid: what the court geometry implies from all clicked points
            for a, b in LINES:
                cv2.line(img, to_canvas(full[a - 1]), to_canvas(full[b - 1]), (0, 140, 255), 1)
            # effective position of every keypoint: the click where there is one, else the computed point
            eff = [points[i] if i < len(points) and points[i] is not None else full[i] for i in range(12)]
            # solid red grid through the actual points, so dragging a point moves its lines
            for a, b in LINES:
                cv2.line(img, to_canvas(eff[a - 1]), to_canvas(eff[b - 1]), (0, 0, 255), 2)
            for i, p in enumerate(full):
                if i >= len(points) or points[i] is None:
                    cv2.circle(img, to_canvas(p), 7, (255, 200, 0), 2)
                    cv2.putText(img, f"k{i + 1}", (to_canvas(p)[0] + 10, to_canvas(p)[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        for i, p in enumerate(points):
            if p is None:
                continue
            c = to_canvas(p)
            cv2.circle(img, c, 8, (0, 255, 255), -1)
            cv2.putText(img, f"k{i + 1}", (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if len(points) < 12:
            msg = f"click k{len(points) + 1} of 12 (n = skip)"
        elif full is None:
            msg = "not enough clicked points to complete the court"
        else:
            msg = "all 12 set: Enter/s to save"
        if err is not None:
            msg += f"   fit error: mean {err.mean():.1f}px, max {err.max():.1f}px"
        cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(img, "red = through your points, orange = court geometry fit. drag to move, right-click to unpin  (n skip, u undo, r reset, s save, q quit)",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

    drag = {"i": None}
    grab_radius = 14  # display pixels

    def nearest_point(x: float, y: float) -> Optional[int]:
        """Index of the clicked or computed point under the cursor, if any."""
        full, _ = complete_points(points)
        best, best_d = None, grab_radius
        for i in range(12):
            if i < len(points) and points[i] is not None:
                px, py = points[i]
            elif full is not None:
                px, py = full[i]
            else:
                continue
            d = np.hypot((px + pad) * args.scale - x, (py + pad) * args.scale - y)
            if d < best_d:
                best, best_d = i, d
        return best

    def on_mouse(event, x, y, flags, params):
        sx, sy = x / args.scale - pad, y / args.scale - pad
        if event == cv2.EVENT_LBUTTONDOWN:
            i = nearest_point(x, y)
            if i is not None:
                # grabbing a computed point pins it; pad the list so the index exists
                while len(points) <= i:
                    points.append(None)
                points[i] = (sx, sy)
                drag["i"] = i
            elif len(points) < 12:
                points.append((sx, sy))
            cv2.imshow(win, render())
        elif event == cv2.EVENT_MOUSEMOVE and drag["i"] is not None:
            points[drag["i"]] = (sx, sy)
            cv2.imshow(win, render())
        elif event == cv2.EVENT_LBUTTONUP:
            drag["i"] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = nearest_point(x, y)
            if i is not None and i < len(points) and points[i] is not None:
                points[i] = None
                cv2.imshow(win, render())

    win = "select court keypoints"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, render())

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            print("aborted, nothing saved")
            break
        if key == ord("n") and len(points) < 12:
            points.append(None)
            cv2.imshow(win, render())
        if key == ord("u") and points:
            points.pop()
            cv2.imshow(win, render())
        if key == ord("r"):
            points.clear()
            cv2.imshow(win, render())
        if key in (13, ord("s")):
            if len(points) != 12:
                print(f"need 12 entries (clicked or skipped), have {len(points)}")
                continue
            full, err = complete_points(points)
            if full is None:
                print("need at least 4 clicked points spread over the court")
                continue
            # keep clicked points verbatim, fill skipped ones from the homography
            final = [
                list(points[i]) if points[i] is not None else [float(full[i][0]), float(full[i][1])]
                for i in range(12)
            ]
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(final, f)
            preview = out.with_suffix(".png")
            cv2.imwrite(str(preview), render())
            print(f"saved 12 keypoints to {out} (preview: {preview})")
            print(f"fit error of clicked points: mean {err.mean():.1f}px, max {err.max():.1f}px")
            for i, (x, y) in enumerate(final):
                tag = "clicked" if points[i] is not None else "computed"
                print(f"k{i + 1}: ({x:.0f}, {y:.0f})  {tag}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
