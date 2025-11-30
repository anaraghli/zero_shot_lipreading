import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_grid import GridLipReadingDataset, CharVocab
from model_transformer import LipReadingTransformerCTC
from model_baseline import BLANK_IDX  # reuse same vocab / blank index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BATCH_SIZE = 8          # you can try 8 or 16 on 3080
EPOCHS = 5              # start with 5, you can increase later
LR = 1e-4               # a bit lower LR for transformer
MAX_FRAMES = 40         # keep consistent with baseline


def collate_fn(batch):
    """
    Pads variable-length video and label sequences for CTC training.
    Same as in train_baseline.py, but we duplicate it here for clarity.
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


def main():
    vocab = CharVocab()
    train_ds = GridLipReadingDataset(split="train", vocab=vocab, max_frames=MAX_FRAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    model = LipReadingTransformerCTC().to(DEVICE)

    criterion = nn.CTCLoss(
        blank=BLANK_IDX,
        zero_infinity=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting transformer training on", DEVICE)
    model.train()

    for epoch in range(1, EPOCHS + 1):
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
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i+1}] Loss: {avg_loss:.4f}")
                total_loss = 0.0

        # save checkpoint each epoch
        ckpt_path = f"checkpoints/transformer_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved transformer checkpoint: {ckpt_path}")

    print("Transformer training complete.")


if __name__ == "__main__":
    main()
