import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_grid import GridLipReadingDataset, CharVocab
from model_transformer import LipReadingTransformerCTC
from model_baseline import BLANK_IDX  # reuse same blank index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


def collate_fn(batch):
    """
    Pads variable-length video and label sequences for CTC training.
    """
    # sort by video length (descending) for efficiency
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

    return padded_videos, lengths, concat_labels, target_lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformer-based lip reading model (GRID)"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=40)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag for checkpoint filenames, e.g. 'L4_d256'",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    vocab = CharVocab()
    train_ds = GridLipReadingDataset(
        split="train",
        vocab=vocab,
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    model = LipReadingTransformerCTC(
        cnn_out_dim=args.d_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(DEVICE)

    criterion = nn.CTCLoss(
        blank=BLANK_IDX,
        zero_infinity=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting transformer training on", DEVICE)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for i, (videos, lengths, targets, target_lengths) in enumerate(train_loader):
            videos = videos.to(DEVICE)
            lengths = lengths.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(videos, lengths)  # (T_max, B, num_classes)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            loss = criterion(
                log_probs,
                targets,
                lengths,
                target_lengths,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                avg_loss = total_loss / 50
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{i+1}] "
                    f"Loss: {avg_loss:.4f}"
                )
                total_loss = 0.0

        # save checkpoint each epoch, include depth/tag in filename
        tag = args.tag if args.tag else f"L{args.num_layers}_d{args.d_model}"
        ckpt_path = f"checkpoints/transformer_{tag}_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved transformer checkpoint: {ckpt_path}")

    print("Transformer training complete.")


if __name__ == "__main__":
    main()
