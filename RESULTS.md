# Zero-shot Lip Reading Results (GRID, s1–s6 → s7–s8)

## Setup

- Dataset: GRID corpus
- Train speakers: **s1–s6**
- Test speakers (zero-shot): **s7–s8**
- Input: 64×64 mouth crops (MediaPipe FaceMesh ROI)
- Max frames per clip: 40
- Loss: CTC, character-level
- Evaluation: greedy CTC decoding, Character Error Rate (CER)

---

## Baseline vs Transformer (4-layer)

| Model                      | Test CER (s7–s8) |
|---------------------------|------------------|
| CNN–BiGRU baseline        | 0.513            |
| Transformer, 4 layers     | 0.466            |

- Absolute improvement: 0.513 → 0.466 (Δ = 0.047)
- Relative CER reduction: ≈ **9%**

The 4-layer Transformer encoder improves zero-shot performance on unseen speakers compared to the CNN–BiGRU baseline.

---

## Effect of Transformer Depth on Zero-shot CER

All transformer models share the same CNN front-end and are trained for 5 epochs
on speakers s1–s6 (zero-shot evaluation on s7–s8).

| Model                         | Test CER (s7–s8) |
|------------------------------|------------------|
| CNN–BiGRU baseline           | 0.513            |
| Transformer, 2 layers        | 0.530            |
| Transformer, 4 layers        | 0.466            |
| Transformer, 6 layers        | 0.529            |

### Observations

- A **4-layer** transformer encoder improves zero-shot performance compared to the
  CNN–BiGRU baseline, reducing CER from 0.513 to 0.466 (≈9% relative CER reduction).
- A **2-layer** transformer underperforms both the baseline and the 4-layer model
  (CER 0.530), suggesting that too shallow an encoder cannot fully capture the temporal
  dynamics of lip motion.
- Increasing depth further to **6 layers** does **not** improve performance (CER 0.529),
  and in this configuration performs similarly or worse than the baseline. This hints at
  diminishing returns and possible overfitting given the limited number of training
  speakers and epochs.
