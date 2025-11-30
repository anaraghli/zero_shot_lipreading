import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_grid import GridLipReadingDataset, CharVocab
from model_baseline import LipReadingGRUCTC, BLANK_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BATCH_SIZE = 4
MAX_FRAMES = 40

# Change this if your best checkpoint is a different epoch
CKPT_PATH = "checkpoints/baseline_epoch_5.pt"


def collate_fn(batch):
    """
    Same as in train_baseline.py – pads variable-length sequences for CTC.
    """
    batch = sorted(batch, key=lambda x: x["video"].shape[0], reverse=True)

    videos = [b["video"] for b in batch]
    labels = [b["label"] for b in batch]

    lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    T_max = lengths.max().item()
    B = len(videos)
    C, H, W = videos[0].shape[1:]

    padded_videos = torch.zeros(B, T_max, C, H, W, dtype=torch.float32)
    for i, v in enumerate(videos):
        t = v.shape[0]
        padded_videos[i, :t] = v

    concat_labels = torch.cat(labels, dim=0)

    return padded_videos, lengths, concat_labels, target_lengths, batch


def greedy_decode_ctc(logits, lengths, vocab: CharVocab):
    """
    logits: (T_max, B, C)
    lengths: (B,) lengths of each input sequence (before padding)
    Returns: list of decoded strings (length B)
    """
    # (T_max, B, C) -> argmax over classes -> (T_max, B)
    pred_indices = logits.argmax(dim=-1)  # (T_max, B)
    pred_indices = pred_indices.cpu()

    decoded_texts = []

    for b in range(pred_indices.shape[1]):
        seq = pred_indices[:, b][: lengths[b]].tolist()

        # CTC collapsing: remove repeats and blanks
        collapsed = []
        prev = BLANK_IDX
        for idx in seq:
            if idx != BLANK_IDX and idx != prev:
                collapsed.append(idx)
            prev = idx

        text = vocab.indices_to_text(collapsed)
        decoded_texts.append(text)

    return decoded_texts


def cer(ref: str, hyp: str) -> float:
    """
    Character Error Rate = edit_distance(ref, hyp) / len(ref)
    Simple Levenshtein distance.
    """
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return float(m > 0)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m] / max(1, n)


def main():
    vocab = CharVocab()

    print("[INFO] Loading test dataset (speakers from test_speakers.txt)")
    test_ds = GridLipReadingDataset(split="test", vocab=vocab, max_frames=MAX_FRAMES)
    print("Test dataset size:", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    print(f"[INFO] Loading checkpoint from {CKPT_PATH}")
    model = LipReadingGRUCTC().to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    total_cer = 0.0
    total_count = 0
    printed = 0
    max_print = 20  # how many examples to print

    with torch.no_grad():
        for videos, lengths, targets, target_lengths, batch in test_loader:
            videos = videos.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(videos, lengths)  # (T_max, B, C)
            log_probs = F.log_softmax(logits, dim=-1)

            # Greedy decode
            pred_texts = greedy_decode_ctc(log_probs, lengths, vocab)

            # Ground truth texts from batch list
            gt_texts = [b["text"] for b in batch]

            # Compute CER
            for gt, pred in zip(gt_texts, pred_texts):
                total_cer += cer(gt, pred)
                total_count += 1

            # Print some examples
            for gt, pred, b in zip(gt_texts, pred_texts, batch):
                if printed >= max_print:
                    break
                print("-" * 60)
                print(f"Speaker: {b['speaker']}  Video ID: {b['video_id']}")
                print(f"GT : {gt}")
                print(f"PRD: {pred}")
                printed += 1

            if printed >= max_print:
                break

    avg_cer = total_cer / max(1, total_count)
    print("\n[RESULT] Average CER on test set (approx, using greedy decode): "
          f"{avg_cer:.3f}")


if __name__ == "__main__":
    main()
