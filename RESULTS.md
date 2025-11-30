# Zero-shot Lip Reading Results (GRID, s1–s6 → s7–s8)

## Setup

- Dataset: GRID Corpus
- Speakers:
  - Train: s1–s6
  - Test (zero-shot speakers): s7–s8
- Input: 64×64 mouth crops (Mediapipe FaceMesh ROI)
- Max frames per clip: 40
- Loss: CTC, char-level
- Evaluation: greedy CTC decoding, character error rate (CER)

## Models

### 1. Baseline – CNN + BiGRU + CTC

- Frame encoder: 4-layer CNN
- Temporal encoder: 2-layer BiGRU (hidden size 256)
- Output: char-level logits + CTC

**Test CER (s7–s8, epoch 5):** `0.513`

### 2. Transformer – CNN + Transformer Encoder + CTC

- Frame encoder: same CNN as baseline
- Temporal encoder: 4-layer Transformer encoder (d_model=256, nhead=4)
- Output: char-level logits + CTC

**Test CER (s7–s8, epoch 5):** `0.455`

- Absolute improvement: ~**5.8** percentage points
- Relative CER reduction: ~**11%**
