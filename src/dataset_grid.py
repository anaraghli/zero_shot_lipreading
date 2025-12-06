import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

# Paths
RAW_ROOT = Path("data/raw/grid")
PROC_ROOT = Path("data/processed/grid_mouth")
SPLITS_DIR = Path("splits")

# Should match your preprocessing size
OUTPUT_SIZE = 64

# We’ll use char-level CTC. 0 is reserved for the CTC "blank".
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 '"


class CharVocab:
    def __init__(self, chars: str = VOCAB_CHARS):
        self.chars = chars
        # 0 = CTC blank, so characters start from 1
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(chars)}

    def text_to_indices(self, text: str) -> List[int]:
        text = text.lower()
        indices = []
        for ch in text:
            if ch in self.char2idx:
                indices.append(self.char2idx[ch])
            else:
                # Skip unknown chars; you could also add an <unk> token
                continue
        return indices

    def indices_to_text(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            if idx == 0:
                # CTC blank
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)


def load_speaker_list(split: str) -> List[str]:
    """
    split: 'train' or 'test'
    Reads splits/train_speakers.txt or splits/test_speakers.txt
    """
    if split == "train":
        path = SPLITS_DIR / "train_speakers.txt"
    elif split == "test":
        path = SPLITS_DIR / "test_speakers.txt"
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(path, "r") as f:
        speakers = [line.strip() for line in f if line.strip()]
    return speakers


def parse_align_to_text(align_path: Path) -> str:
    """
    Parse a GRID align file into a text sentence.

    Typical GRID align line format:
      <start> <end> <word>

    We:
      - read all lines
      - take the 3rd column as the word
      - drop 'sil'
      - join words with spaces
    """
    words = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            w = parts[2]
            if w.lower() == "sil":
                continue
            words.append(w)
    sentence = " ".join(words)
    return sentence


class GridLipReadingDataset(Dataset):
    def __init__(
        self,
        split: str,
        vocab: Optional[CharVocab] = None,
        max_frames: Optional[int] = None,
    ):
        """
        split: 'train' or 'test'
        vocab: CharVocab instance (if None, a default one is created)
        max_frames: if not None, truncate sequences to at most this many frames
        """
        super().__init__()
        self.split = split
        self.vocab = vocab or CharVocab()
        self.max_frames = max_frames

        self.samples: List[Dict] = []
        self._build_index()

    def _build_index(self):
        """
        Build an index of samples by matching processed video directories
        and alignment files by their common stem (e.g. 'bbaf2n').
        """
        speakers = load_speaker_list(self.split)

        for spk in speakers:
            spk_proc_dir = PROC_ROOT / spk
            spk_raw_dir = RAW_ROOT / spk

            if not spk_proc_dir.exists():
                print(f"[WARN] Processed dir for {spk} not found: {spk_proc_dir}")
                continue

            align_dir = spk_raw_dir / "align"
            if not align_dir.exists():
                print(f"[WARN] Align dir for {spk} not found: {align_dir}")
                continue

            # Map: video_stem -> vid_dir  (e.g. 'bbaf2n' -> data/processed/.../bbaf2n)
            vid_dirs: Dict[str, Path] = {}
            for d in sorted(spk_proc_dir.iterdir()):
                if not d.is_dir():
                    continue
                stem = d.name  # e.g. "bbaf2n"
                vid_dirs[stem] = d

            if not vid_dirs:
                print(f"[WARN] No processed videos found for speaker {spk}")
                continue

            # Map: align_stem -> align_path  (e.g. 'bbaf2n' -> data/raw/.../bbaf2n.align)
            align_files: Dict[str, Path] = {}
            for p in sorted(align_dir.glob("*.align")):
                align_files[p.stem] = p
            for p in sorted(align_dir.glob("*.txt")):
                align_files[p.stem] = p

            if not align_files:
                print(f"[WARN] No align files found for speaker {spk}")
                continue

            # Intersection of stems present in both
            common_stems = sorted(set(vid_dirs.keys()) & set(align_files.keys()))
            if not common_stems:
                print(
                    f"[WARN] No matching stems between processed videos and align files "
                    f"for {spk}"
                )
                continue

            for stem in common_stems:
                vid_dir = vid_dirs[stem]
                align_path = align_files[stem]

                frame_paths = sorted(vid_dir.glob("frame_*.png"))
                if len(frame_paths) == 0:
                    continue

                text = parse_align_to_text(align_path)
                label_indices = self.vocab.text_to_indices(text)
                if len(label_indices) == 0:
                    continue

                self.samples.append(
                    {
                        "speaker": spk,
                        "video_id": stem,
                        "frames": frame_paths,
                        "text": text,
                        "label": label_indices,
                    }
                )

        print(f"[INFO] Built {self.split} dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        """
        Load a sequence of frames and return a tensor of shape (T, C, H, W).
        """
        if self.max_frames is not None and len(frame_paths) > self.max_frames:
            frame_paths = frame_paths[: self.max_frames]

        imgs = []
        for p in frame_paths:
            img = cv2.imread(str(p))  # BGR
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # normalize to [0,1]
            # H, W, C -> C, H, W
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)

        if not imgs:
            # Fallback: dummy frame if all failed
            imgs = [np.zeros((3, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)]

        video = np.stack(imgs, axis=0)  # (T, C, H, W)
        return torch.from_numpy(video)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video = self._load_frames(sample["frames"])
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "video": video,               # (T, C, H, W)
            "label": label,               # (L,)
            "text": sample["text"],
            "speaker": sample["speaker"],
            "video_id": sample["video_id"],
        }
