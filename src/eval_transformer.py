import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_grid import GridLipReadingDataset, CharVocab
from model_transformer import LipReadingTransformerCTC
from model_baseline import BLANK_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BATCH_SIZE = 4
MAX_FRAMES = 40


def collate_fn(batch):
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
    pred_indices = logits.argmax(dim=-1)  # (T_max, B)
    pred_indices = pred_indices.cpu()

    decoded_texts = []

    for b in range(pred_indices.shape[1]):
        seq = pred_indices[:, b][: lengths[b]].tolist()

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
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m] / max(1, n)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate transformer lip-reading model (GRID)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to transformer checkpoint .pt file",
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=20,
        help="Number of example predictions to print",
    )
    # model hyperparams – must match training config
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()
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

    print(f"[INFO] Loading transformer checkpoint from {args.ckpt}")

    # IMPORTANT: construct model with SAME hyperparameters as during training
    model = LipReadingTransformerCTC(
        cnn_out_dim=args.d_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(DEVICE)

    state_dict = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    total_cer = 0.0
    total_count = 0
    printed = 0

    with torch.no_grad():
        for videos, lengths, targets, target_lengths, batch in test_loader:
            videos = videos.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(videos, lengths)  # (T_max, B, C)
            log_probs = F.log_softmax(logits, dim=-1)

            pred_texts = greedy_decode_ctc(log_probs, lengths, vocab)
            gt_texts = [b["text"] for b in batch]

            for gt, pred in zip(gt_texts, pred_texts):
                total_cer += cer(gt, pred)
                total_count += 1

            for gt, pred, b in zip(gt_texts, pred_texts, batch):
                if printed >= args.max_print:
                    break
                print("-" * 60)
                print(f"Speaker: {b['speaker']}  Video ID: {b['video_id']}")
                print(f"GT : {gt}")
                print(f"PRD: {pred}")
                printed += 1

            if printed >= args.max_print:
                break

    avg_cer = total_cer / max(1, total_count)
    print(
        "\n[RESULT] Transformer Average CER on test set (greedy decode): "
        f"{avg_cer:.3f}"
    )


if __name__ == "__main__":
    main()
