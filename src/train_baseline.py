import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_grid import GridLipReadingDataset, CharVocab
from model_baseline import LipReadingGRUCTC, NUM_CLASSES, BLANK_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
MAX_FRAMES = 40  # to keep things manageable


def collate_fn(batch):
    """
    Pads variable-length video and label sequences for CTC training.
    """
    # sort by video length (descending) for pack_padded_sequence efficiency
    batch = sorted(batch, key=lambda x: x["video"].shape[0], reverse=True)

    videos = [b["video"] for b in batch]     # [(T_i, C, H, W), ...]
    labels = [b["label"] for b in batch]     # [(L_i,), ...]

    lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    T_max = lengths.max().item()
    B = len(videos)
    C, H, W = videos[0].shape[1:]

    # pad videos
    padded_videos = torch.zeros(B, T_max, C, H, W, dtype=torch.float32)
    for i, v in enumerate(videos):
        t = v.shape[0]
        padded_videos[i, :t] = v

    # concatenate labels into 1D for CTC
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

    model = LipReadingGRUCTC().to(DEVICE)
    criterion = nn.CTCLoss(
        blank=BLANK_IDX,
        zero_infinity=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training on", DEVICE)
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

        # optionally save checkpoint each epoch
        torch.save(model.state_dict(), f"checkpoints/baseline_epoch_{epoch}.pt")
        print(f"[INFO] Saved checkpoint for epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()
