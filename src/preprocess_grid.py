import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from pathlib import Path

RAW_ROOT = Path("data/raw/grid")
PROC_ROOT = Path("data/processed/grid_mouth")

SPEAKERS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
OUTPUT_SIZE = 64  # 64x64 mouth crops

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Mouth-related landmark indices in FaceMesh (approx)
MOUTH_LANDMARKS = [
    61,  146, 91,  181, 84,  17,  314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14,  87,  178, 88,  95,  185, 40,  39,
    37,  0,   267, 269, 270, 409, 415, 310,
    311, 312, 13,  82,  81,  42,  183, 78
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_mouth_box(landmarks, image_width, image_height, margin=0.2):
    xs = []
    ys = []
    for idx in MOUTH_LANDMARKS:
        lm = landmarks.landmark[idx]
        xs.append(lm.x * image_width)
        ys.append(lm.y * image_height)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add margin around mouth box
    w = x_max - x_min
    h = y_max - y_min
    x_min -= margin * w
    x_max += margin * w
    y_min -= margin * h
    y_max += margin * h

    # Filter out tiny boxes before clipping
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 10 or box_h < 10:
        return None  # skip tiny boxes

    # Clip to image bounds
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_width - 1, int(x_max))
    y_max = min(image_height - 1, int(y_max))

    return x_min, y_min, x_max, y_max


def process_video(video_path: Path, out_dir: Path) -> bool:
    """
    Extract mouth crops from a single GRID video and save them as 64x64 PNG frames.

    Returns True if at least one frame was saved, False otherwise.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return False

    saved_any = False
    last_box = None
    frame_idx = 0

    # We only create the directory when we actually save the first frame
    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        box = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            box = get_mouth_box(face_landmarks, w, h)
            if box is not None:
                last_box = box

        # If this frame has no valid box, but we had one before, reuse it.
        if box is None:
            if last_box is None:
                # No mouth box ever found yet → skip this frame
                frame_idx += 1
                continue
            box = last_box

        x_min, y_min, x_max, y_max = box
        mouth = frame[y_min:y_max, x_min:x_max]

        # Safety check: skip if the crop is empty or invalid
        if mouth.size == 0 or (x_max <= x_min) or (y_max <= y_min):
            frame_idx += 1
            continue

        mouth = cv2.resize(mouth, (OUTPUT_SIZE, OUTPUT_SIZE))

        # Only now ensure the directory exists
        if not saved_any:
            ensure_dir(out_dir)

        out_path = out_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), mouth)
        saved_any = True

        frame_idx += 1

    cap.release()
    return saved_any


def main():
    for spk in SPEAKERS:
        spk_video_dir = RAW_ROOT / spk / "video"
        if not spk_video_dir.exists():
            print(f"[WARN] Video directory does not exist: {spk_video_dir}")
            continue

        video_files = sorted(spk_video_dir.glob("*.mpg"))
        print(f"Processing {spk}: {len(video_files)} videos")

        for vid_path in tqdm(video_files):
            vid_name = vid_path.stem  # e.g., "s1_1"
            out_dir = PROC_ROOT / spk / vid_name
            if out_dir.exists() and any(out_dir.iterdir()):
                # already processed
                continue
            process_video(vid_path, out_dir)

if __name__ == "__main__":
    main()
