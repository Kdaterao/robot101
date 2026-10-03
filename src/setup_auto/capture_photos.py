"""Live OpenCV camera capture. Saves a JPEG every N seconds.

    uv run python src/setup_auto/capture_photos.py
    uv run python src/setup_auto/capture_photos.py --interval 2 --camera 1
Press 1 for an extra photo. Press q to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _save(frame, out_dir: Path) -> Path | None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = out_dir / f"IMG_{stamp}.jpg"
    if cv2.imwrite(str(path), frame):
        print(f"Saved {path}")
        return path
    print(f"Failed to write {path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Take pictures from an OpenCV camera every N seconds"
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--interval",
        "-n",
        type=float,
        default=2.0,
        help="Seconds between automatic photos (default: 2)",
    )
    parser.add_argument(
        "--out",
        default="captured_images",
        help="Folder to save JPEGs (created if missing)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        print("--interval must be > 0")
        sys.exit(1)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)

    print(f"Camera {args.camera} open. Saving to {out_dir}")
    print(f"Photo every {args.interval:g}s. Press 1 for an extra shot. Press q to quit.")

    window = "Capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved = 0
    flash_until = 0
    next_shot = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed.")
                break

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            now = time.perf_counter()
            if now >= next_shot:
                if _save(frame, out_dir) is not None:
                    saved += 1
                    flash_until = cv2.getTickCount() + int(0.2 * cv2.getTickFrequency())
                next_shot = now + args.interval

            remaining = max(0.0, next_shot - time.perf_counter())
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"every {args.interval:g}s   next {remaining:0.1f}s   "
                f"1 = extra   q = quit   saved: {saved}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if cv2.getTickCount() < flash_until:
                cv2.rectangle(
                    overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1),
                    (0, 255, 0), 8,
                )
            cv2.imshow(window, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                if _save(frame, out_dir) is not None:
                    saved += 1
                    flash_until = cv2.getTickCount() + int(0.2 * cv2.getTickFrequency())
                next_shot = time.perf_counter() + args.interval
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. {saved} photo(s) in {out_dir}")


if __name__ == "__main__":
    main()
