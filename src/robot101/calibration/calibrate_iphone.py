
from robot101.paths import REPO_ROOT

import cv2
import numpy as np
import os
import json
import subprocess
import tempfile


# ============================================================
# SETTINGS
# ============================================================

# ChArUco board dimensions
SQUARES_X = 10
SQUARES_Y = 7

# Physical dimensions of the board.
# These don't need to correspond to your Mac screen size exactly,
# but keep the ratio consistent.
SQUARE_LENGTH = 0.04       # meters
MARKER_LENGTH = 0.03       # meters

# Folder where calibration images will be saved
IMAGE_DIR_NAME = "calibration_images"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
HEIC_EXTS = {".heic", ".heif"}

# Camera image resolution.
# CHANGE THIS to the resolution of the photos you take.
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# ============================================================
# CREATE CHArUCO BOARD
# ============================================================

def create_board():

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )

    # Generate a printable/displayable board
    img = board.generateImage(
        (2000, 1400),
        marginSize=50
    )

    cv2.imwrite("charuco_board.png", img)

    print()
    print("Created:")
    print("    charuco_board.png")
    print()
    print("Open this image on your Mac and display it FULLSCREEN.")
    print()
    print("IMPORTANT:")
    print("Do NOT resize/stretch the image.")
    print("Keep the aspect ratio exactly the same.")
    print()


# ============================================================
# FIND / READ CALIBRATION IMAGES
# ============================================================

def find_image_dir():
    """Prefer the folder that actually contains photos."""

    candidates = [
        os.path.join(SCRIPT_DIR, IMAGE_DIR_NAME),
        os.path.join(os.getcwd(), IMAGE_DIR_NAME),
        os.path.join(REPO_ROOT, IMAGE_DIR_NAME),
    ]

    seen = set()
    unique = []
    for path in candidates:
        resolved = os.path.abspath(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    for path in unique:
        if os.path.isdir(path) and collect_images(path):
            return path

    return unique[0]


def collect_images(image_dir):
    """Return one image path per shot, preferring JPEG/PNG over HEIC."""

    if not os.path.isdir(image_dir):
        return []

    by_stem = {}

    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        stem = os.path.splitext(name)[0]
        path = os.path.join(image_dir, name)
        previous = by_stem.get(stem)

        if previous is None:
            by_stem[stem] = path
            continue

        previous_ext = os.path.splitext(previous)[1].lower()
        if previous_ext in HEIC_EXTS and ext not in HEIC_EXTS:
            by_stem[stem] = path

    return [by_stem[stem] for stem in sorted(by_stem)]


def read_image(path):
    """Load an image OpenCV can use, converting HEIC via macOS sips."""

    ext = os.path.splitext(path)[1].lower()

    if ext not in HEIC_EXTS:
        return cv2.imread(path)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "sips",
                "-s", "format", "jpeg",
                path,
                "--out", tmp_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Could not convert HEIC: {path}")
            if result.stderr:
                print(result.stderr.strip())
            return None

        return cv2.imread(tmp_path)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# DETECT CHArUCO BOARD
# ============================================================

def detect_charuco(image, detector):

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)

    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None

    return charuco_corners, charuco_ids


# ============================================================
# CALIBRATE
# ============================================================

def calibrate():

    image_dir = find_image_dir()
    images = collect_images(image_dir)

    print()
    print(f"Looking in: {image_dir}")

    if len(images) < 10:
        print()
        print("ERROR:")
        print("You need at least ~10 images.")
        print("20–30 images is recommended.")
        print()
        print("Put iPhone photos (.HEIC, .jpeg, .jpg, or .png) into:")
        print(f"    {image_dir}")
        print()
        return

    print()
    print(f"Found {len(images)} calibration images.")
    print()

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )

    detector = cv2.aruco.CharucoDetector(board)

    all_object_points = []
    all_image_points = []

    image_size = None

    # --------------------------------------------------------
    # DETECT BOARD IN EVERY IMAGE
    # --------------------------------------------------------

    for i, filename in enumerate(images):

        image = read_image(filename)

        if image is None:
            print(f"Could not read {filename}")
            continue

        image_size = (
            image.shape[1],
            image.shape[0]
        )

        corners, ids = detect_charuco(image, detector)

        if corners is None:
            print(
                f"[{i+1}/{len(images)}] "
                f"FAILED: {os.path.basename(filename)}"
            )
            continue

        object_points, image_points = board.matchImagePoints(
            corners,
            ids
        )

        if object_points is None or len(object_points) < 6:
            print(
                f"[{i+1}/{len(images)}] "
                f"FAILED: {os.path.basename(filename)}"
            )
            continue

        print(
            f"[{i+1}/{len(images)}] "
            f"GOOD: {os.path.basename(filename)} "
            f"({len(ids)} corners)"
        )

        all_object_points.append(object_points)
        all_image_points.append(image_points)

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if len(all_object_points) < 10:

        print()
        print("ERROR:")
        print(
            f"Only {len(all_object_points)} valid images "
            "were detected."
        )
        print("Take more pictures.")
        return

    print()
    print(
        f"Using {len(all_object_points)} valid images "
        "for calibration."
    )
    print()

    # --------------------------------------------------------
    # CALIBRATE CAMERA
    # --------------------------------------------------------

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            all_object_points,
            all_image_points,
            image_size,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("=" * 60)
    print("CAMERA CALIBRATION RESULTS")
    print("=" * 60)

    print()
    print("Camera matrix:")
    print(camera_matrix)

    print()
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())

    print()
    print("AprilTag parameters:")
    print()
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    print()
    print(f"RMS reprojection error: {rms}")

    print()
    print("=" * 60)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    np.save(
        "camera_matrix.npy",
        camera_matrix
    )

    np.save(
        "dist_coeffs.npy",
        dist_coeffs
    )

    calibration = {
        "image_width": image_size[0],
        "image_height": image_size[1],

        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),

        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),

        "rms_error": float(rms)
    }

    with open(
        "calibration.json",
        "w"
    ) as f:

        json.dump(
            calibration,
            f,
            indent=4
        )

    print()
    print("Saved:")
    print("    camera_matrix.npy")
    print("    dist_coeffs.npy")
    print("    calibration.json")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_dir = find_image_dir()
    os.makedirs(
        image_dir,
        exist_ok=True
    )

    create_board()

    print()
    print("After displaying charuco_board.png on your Mac:")
    print()
    print("1. Take 20–30 photos with your iPhone.")
    print("2. Move the phone around between photos.")
    print("3. Put the board near different parts of the image.")
    print("4. Tilt the phone in different directions.")
    print("5. Use different distances.")
    print()
    print("Put the photos into:")
    print(f"    {image_dir}")
    print()
    print("Then run this script again.")
    print()

    input(
        "Press ENTER when your calibration images are ready..."
    )

    calibrate()

