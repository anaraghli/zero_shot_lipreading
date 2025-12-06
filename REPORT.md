# Zero-shot Lip Reading on GRID with CNN–BiGRU and Transformer Encoders

## 1. Introduction

Visual speech recognition (lip reading) aims to map silent video of a speaker’s mouth
to a textual transcription. A key challenge is generalising to **unseen speakers**,
rather than only performing well on the identities seen during training.

In this project, I build an end-to-end lip reading pipeline on the GRID corpus and
compare a CNN–BiGRU baseline with Transformer-based temporal encoders. I focus on a
**zero-shot by speaker** setting, where models are trained on one group of speakers
and evaluated on different, unseen speakers.

## 2. Dataset and Zero-shot Setup

I use the GRID audiovisual corpus, which contains short, fixed-grammar sentences
spoken by multiple speakers. Each sentence consists of a command, colour, preposition,
letter, digit, and an adverb (e.g. “bin blue at f two now”).

To study zero-shot speaker generalisation:

- **Train speakers:** s1–s6  
- **Test speakers (unseen):** s7–s8  

That is, models never see s7 and s8 during training and must generalise to these
speakers at test time.

## 3. Preprocessing and Input Representation

Starting from the raw GRID videos, I extract a sequence of mouth-region crops for
each utterance:

1. Use **MediaPipe FaceMesh** to detect facial landmarks for each frame.
2. Compute a bounding box around mouth-related landmarks.
3. Crop the mouth region and resize it to **64×64** RGB.
4. Normalise pixel values to [0, 1] and store frames as `(T, 3, 64, 64)` tensors.

To keep training efficient, I cap each sequence at **40 frames**. The result is a
dataset of ~5.8k mouth-only video clips for speakers s1–s6 and 2k clips for s7–s8.

## 4. Models

### 4.1 CNN–BiGRU Baseline

The baseline model follows a common lip reading architecture:

- A **frame-wise CNN** processes each 64×64 RGB mouth image into a feature vector.
- A **2-layer bidirectional GRU** models temporal dynamics over the sequence of
  frame features.
- A final linear layer maps hidden states to character logits.

I train the model with **CTC loss** at the character level, using a vocabulary of
26 letters, digits 0–9, space, and apostrophe, plus a CTC blank symbol. This avoids
requiring explicit frame-to-character alignments.

### 4.2 Transformer Encoder with CTC

To explore whether self-attention helps zero-shot generalisation, I replace the
recurrent encoder with a **Transformer encoder**:

- The same CNN front-end is used to extract per-frame features.
- Positional encodings are added to the sequence of frame features.
- A **TransformerEncoder** with multi-head self-attention models temporal structure.
- As with the baseline, a linear layer and CTC loss are applied at the character level.

I fix the transformer size to `d_model = 256`, `nhead = 4`, and vary only the number
of encoder layers:

- 2 layers
- 4 layers
- 6 layers

This allows a small ablation on how depth affects performance on unseen speakers.

## 5. Experimental Setup

All models are implemented in PyTorch and trained on an NVIDIA RTX 3080 GPU.

- Optimiser: **Adam**
- Learning rate: **1e-3** for the CNN–BiGRU baseline, **1e-4** for transformers
- Epochs: **5** for all models
- Batch size: **8**
- Maximum frames per clip: **40**

Training always uses speakers **s1–s6**, and evaluation is always performed on
**s7–s8** to measure zero-shot speaker generalisation. I use **greedy CTC decoding**
(i.e. argmax over characters at each timestep, followed by CTC collapsing) and report
**Character Error Rate (CER)** as the main metric.

## 6. Results

### 6.1 Baseline vs Transformer

The table below compares the CNN–BiGRU baseline with the best transformer model:

| Model                       | Test CER on s7–s8 |
|-----------------------------|-------------------|
| CNN–BiGRU baseline          | 0.513             |
| Transformer (4 layers)      | 0.466             |

The 4-layer transformer improves CER from **0.513** to **0.466**, which corresponds
to roughly a **9% relative reduction** in character error rate on unseen speakers.
This suggests that self-attention over frame features helps the model capture
longer-range temporal dependencies in mouth motion.

### 6.2 Effect of Transformer Depth

I also vary the depth of the transformer encoder while keeping all other
hyperparameters fixed:

| Model                         | Test CER on s7–s8 |
|------------------------------|--------------------|
| CNN–BiGRU baseline           | 0.513              |
| Transformer, 2 layers        | 0.530              |
| Transformer, 4 layers        | 0.466              |
| Transformer, 6 layers        | 0.529              |

A 2-layer transformer underperforms the baseline, while a 4-layer transformer
performs best. Increasing depth further to 6 layers does not yield improvements
and in fact performs similarly or slightly worse than the baseline.

These results indicate that transformer depth has a **sweet spot** in this setting:
too shallow, and the model cannot fully exploit self-attention; too deep, and the
model may overfit or become harder to optimise given the limited number of training
speakers and short training schedule.

## 7. Discussion and Limitations

This project demonstrates that replacing a BiGRU with a transformer encoder can
improve zero-shot lip reading performance on unseen speakers, even in a relatively
small-scale setting. At the same time, the depth ablation shows that simply stacking
more layers does not monotonically improve generalisation.

There are several limitations:

- I only trained for 5 epochs; longer training or a learning rate schedule
  may change the relative performance.
- Evaluation uses **greedy CTC decoding**; beam search and language models could
  reduce CER further.
- The experiments use only one dataset (GRID) and a relatively small number of
  training speakers (six), so results may not directly transfer to larger,
  more diverse corpora.

## 8. Conclusion and Future Work

In summary, I built a complete lip reading pipeline on the GRID corpus, from
mouth-region preprocessing to sequence models and evaluation on unseen speakers.
A 4-layer transformer encoder reduced zero-shot CER from **0.513** (CNN–BiGRU
baseline) to **0.466**, while both shallower and deeper transformers underperformed.

Future work could explore training on more speakers, using stronger decoders
(e.g. beam search with language models), or pretraining the visual front-end
on larger video datasets before fine-tuning for lip reading.
