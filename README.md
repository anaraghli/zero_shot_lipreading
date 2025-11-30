# Zero-shot Lip Reading with CNN–BiGRU and Transformers

This repository contains a small research-style project on **zero-shot lip reading** using the GRID corpus.  
The goal is to train models on a subset of speakers and evaluate how well they **generalize to unseen speakers**.

- Input: sequences of 64×64 mouth crops extracted from video
- Output: character-level transcription using CTC
- Zero-shot setting: train on speakers **s1–s6**, test on unseen speakers **s7–s8**

---

## 1. Project Overview

### Pipeline

1. **Preprocessing**
   - Use MediaPipe FaceMesh to detect facial landmarks.
   - Extract a **mouth region of interest (ROI)** for each frame in GRID videos.
   - Resize crops to **64×64** and save as frame sequences.

2. **Dataset & Splits**
   - `GridLipReadingDataset` pairs processed frame sequences with GRID alignment files.
   - Speaker-based split:
     - Train speakers: `s1, s2, s3, s4, s5, s6`
     - Test (zero-shot) speakers: `s7, s8`

3. **Models**
   - **Baseline:** CNN → BiGRU → CTC
   - **Transformer:** CNN → Transformer encoder → CTC

4. **Evaluation**
   - Greedy CTC decoding.
   - **Character Error Rate (CER)** on unseen speakers (`s7`, `s8`).

---

## 2. Setup

Clone and install dependencies:

```bash
git clone https://github.com/anaraghli/zero_shot_lipreading.git
cd zero_shot_lipreading

# (optional but recommended: create & activate a virtualenv / conda env)

pip install -r requirements.txt
```

The project uses:

- Python 3.11
- PyTorch (CUDA build recommended if you have a GPU)
- OpenCV
- MediaPipe
- tqdm
- NumPy

---

## 3. Data Layout

The GRID corpus must be obtained separately.  
Expected directory structure:

```text
data/
  raw/
    grid/
      s1/
        video/       # original .mpg videos, e.g. bbaf2n.mpg
        align/       # alignment files, e.g. bbaf2n.align
      s2/
      s3/
      s4/
      s5/
      s6/
      s7/
      s8/

  processed/
    grid_mouth/
      s1/
        bbaf2n/
          frame_0000.png
          frame_0001.png
          ...
      s2/
      ...
      s8/
```

Notes:

- The **raw videos** live under `data/raw/grid/sX/video`.
- The **alignment files** live under `data/raw/grid/sX/align`.
- The **processed mouth crops** are created by the preprocessing script and saved under `data/processed/grid_mouth`.

The dataset itself is **not** included in this repo and should not be committed (see `.gitignore`).

---

## 4. Preprocessing (Mouth ROI Extraction)

The preprocessing script reads the original GRID videos, runs MediaPipe FaceMesh, and crops a mouth ROI for each frame.

Run:

```bash
python src/preprocess_grid.py
```

What it does:

- Loops over speakers defined in `SPEAKERS` inside `preprocess_grid.py`.
- For each video in `data/raw/grid/sX/video/`:
  - Detects face landmarks with MediaPipe FaceMesh.
  - Computes a bounding box over mouth-related landmarks.
  - Crops and resizes frames to **64×64**.
  - Saves them as `frame_XXXX.png` under:
    - `data/processed/grid_mouth/sX/VIDEO_ID/`

To test preprocessing on a single speaker first, you can temporarily set:

```python
SPEAKERS = ["s1"]
```

inside `src/preprocess_grid.py`.

---

## 5. Dataset & Splits

Speaker splits are defined in:

- `splits/train_speakers.txt`  
  ```text
  s1
  s2
  s3
  s4
  s5
  s6
  ```

- `splits/test_speakers.txt`  
  ```text
  s7
  s8
  ```

The dataset implementation is in `src/dataset_grid.py`:

- `GridLipReadingDataset`:
  - Loads processed frame sequences from `data/processed/grid_mouth/sX`.
  - Matches videos with alignment files by **stem**, e.g. `bbaf2n`:
    - `data/processed/grid_mouth/s1/bbaf2n/…`
    - `data/raw/grid/s1/align/bbaf2n.align`
  - Parses alignment files into word-level transcripts.
  - Encodes them with a **character-level vocabulary** for CTC.

- Output per sample:
  - `video`: tensor of shape `(T, 3, 64, 64)`
  - `label`: tensor of character indices `(L,)`
  - `text`: original text string
  - `speaker`, `video_id`: metadata

You can quickly sanity-check the dataset with:

```bash
python src/test_dataset.py
```

---

## 6. Models

### 6.1 Baseline: CNN–BiGRU + CTC

Defined in `src/model_baseline.py`:

- Frame encoder: `ConvFeatureExtractor`
  - 4 convolutional layers + pooling → frame-level feature vectors.
- Temporal encoder: 2-layer **bidirectional GRU**.
- Output layer: linear projection to character logits.
- Loss: **CTCLoss** with a char-level vocabulary where index `0` is the CTC blank.

### 6.2 Transformer: CNN + Transformer Encoder + CTC

Defined in `src/model_transformer.py`:

- Same CNN frame encoder as the baseline.
- **PositionalEncoding** over the sequence of frame features.
- `nn.TransformerEncoder` with configurable depth, heads, and feedforward size.
- Linear projection to character logits.
- Trained with the same CTC setup for a fair comparison.

---

## 7. Training

### 7.1 Baseline Training

```bash
python src/train_baseline.py
```

This will:

- Build training dataset from `s1–s6`.
- Train the CNN–BiGRU model with CTC loss.
- Save checkpoints to:

```text
checkpoints/baseline_epoch_1.pt
checkpoints/baseline_epoch_2.pt
...
```

Key settings (see `train_baseline.py`):

- `BATCH_SIZE` (e.g. 8)
- `EPOCHS` (e.g. 5 to start)
- `LR = 1e-3`
- `MAX_FRAMES = 40` (truncate sequences for efficiency)
- Uses GPU if available (`cuda`), otherwise CPU.

### 7.2 Transformer Training

```bash
python src/train_transformer.py
```

This will:

- Use the same training dataset and CTC loss.
- Train the `LipReadingTransformerCTC` model.
- Save checkpoints to:

```text
checkpoints/transformer_epoch_1.pt
checkpoints/transformer_epoch_2.pt
...
```

Default settings (see `train_transformer.py`):

- `BATCH_SIZE` (e.g. 8)
- `EPOCHS` (e.g. 5 to start)
- `LR = 1e-4` (slightly lower LR for transformer)
- `MAX_FRAMES = 40`

---

## 8. Evaluation

Evaluation is done on **unseen speakers** (`s7`, `s8`) with **greedy CTC decoding** and **Character Error Rate (CER)**.

### 8.1 Baseline Evaluation

```bash
python src/eval_baseline.py
```

- Loads `GridLipReadingDataset(split="test")`.
- Loads a CNN–BiGRU checkpoint (configured via `CKPT_PATH` inside the script).
- Runs greedy CTC decoding.
- Computes average CER and prints sample predictions.

Example output:

```text
[RESULT] Average CER on test set (approx, using greedy decode): 0.513
```

### 8.2 Transformer Evaluation

```bash
python src/eval_transformer.py
```

- Loads the same test dataset.
- Loads a transformer checkpoint (`CKPT_PATH`).
- Runs greedy decoding and computes CER.

Example output:

```text
[RESULT] Transformer Average CER on test set (greedy decode): 0.455
```

---

## 9. Results (Zero-shot by Speaker)

**Setup**

- Train speakers: `s1–s6`
- Test speakers (unseen): `s7–s8`
- Metric: Character Error Rate (CER), lower is better.
- Decoding: greedy CTC.

**Results (epoch 5):**

| Model                            | Test CER (s7–s8) |
|----------------------------------|------------------|
| CNN–BiGRU + CTC (baseline)       | 0.513            |
| Transformer encoder + CTC        | 0.455            |

This corresponds roughly to:

- Baseline char accuracy ≈ **48.7%**
- Transformer char accuracy ≈ **54.5%**

The transformer model achieves about an **11% relative reduction in CER** compared to the baseline, indicating improved generalization to **unseen speakers** in this zero-shot setting.

---

## 10. Possible Extensions

Some natural extensions of this project:

- Study how the **number of training speakers** affects zero-shot performance.
- Explore different transformer depths / widths and attention head counts.
- Try **beam search** decoding instead of greedy CTC.
- Experiment with different mouth ROI strategies (e.g., larger crops, different resolutions).
- Add word-level metrics (WER) in addition to CER.
