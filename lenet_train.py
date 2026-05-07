import argparse
import csv
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lenet_model import LeNet5

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NORM_MEAN = 0.2860
NORM_STD = 0.3530


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    num_workers: int
    seed: int


def build_dataloaders(cfg: TrainConfig):
    train_tfm = transforms.Compose(
        [
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # common Fashion-MNIST stats
        ]
    )
    train_ds = datasets.FashionMNIST(root=cfg.data_dir, train=True, download=True, transform=train_tfm)
    test_ds = datasets.FashionMNIST(root=cfg.data_dir, train=False, download=True, transform=eval_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total += y.numel()
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_history_csv(history, csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(history)


def detect_overfitting_epoch(history):
    """
    Heuristic:
    overfitting starts when train loss keeps decreasing while val loss increases,
    and the train/val accuracy gap is noticeably positive.
    """
    if len(history) < 3:
        return None

    for i in range(1, len(history)):
        prev_h = history[i - 1]
        cur_h = history[i]
        train_loss_down = cur_h["train_loss"] < prev_h["train_loss"]
        val_loss_up = cur_h["val_loss"] > prev_h["val_loss"]
        acc_gap = cur_h["train_acc"] - cur_h["val_acc"]
        if train_loss_down and val_loss_up and acc_gap >= 0.02:
            return cur_h["epoch"]
    return None


def smooth_series(values, window: int = 7):
    if window <= 1 or len(values) < 3:
        return values[:]
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        left = max(0, i - half)
        right = min(len(values), i + half + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


@torch.no_grad()
def save_prediction_preview(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    out_path: str,
    epoch: int,
    seed: int,
) -> bool:
    """
    每轮从测试集随机抽 8 张做可视化。若固定用 DataLoader 的第一个 batch 且 shuffle=False，
    则每轮预览图会完全相同。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    model.eval()
    n = len(test_dataset)
    if n == 0:
        return False

    g = torch.Generator()
    g.manual_seed(seed + epoch * 1_000_003)
    k = min(8, n)
    perm = torch.randperm(n, generator=g)[:k]
    imgs = []
    labels = []
    for idx in perm:
        img, lab = test_dataset[int(idx)]
        imgs.append(img)
        labels.append(int(lab) if not isinstance(lab, torch.Tensor) else int(lab.item()))
    images = torch.stack(imgs, dim=0).to(device, non_blocking=True)

    logits = model(images)
    preds = torch.argmax(logits, dim=1).cpu()
    images_cpu = images.detach().cpu()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= images_cpu.size(0):
            axes[i].axis("off")
            continue
        img = images_cpu[i, 0] * NORM_STD + NORM_MEAN
        img = img.clamp(0.0, 1.0)
        gt = labels[i]
        pd = int(preds[i].item())
        ok = "OK" if gt == pd else "ERR"

        axes[i].imshow(img.numpy(), cmap="gray")
        axes[i].set_title(f"P:{CLASS_NAMES[pd]}\nT:{CLASS_NAMES[gt]} [{ok}]", fontsize=8)
        axes[i].axis("off")

    fig.suptitle(f"Prediction Preview (epoch {epoch})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_history_plot(history, plot_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    s_train_loss = smooth_series(train_loss, window=7)
    s_val_loss = smooth_series(val_loss, window=7)
    s_train_acc = smooth_series(train_acc, window=7)
    s_val_acc = smooth_series(val_acc, window=7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, "r.-", alpha=0.25, label="Train loss (raw)")
    axes[0].plot(epochs, val_loss, "b.-", alpha=0.25, label="Val loss (raw)")
    axes[0].plot(epochs, s_train_loss, "r-", linewidth=2.0, label="Train loss (smooth)")
    axes[0].plot(epochs, s_val_loss, "b-", linewidth=2.0, label="Val loss (smooth)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, "r.-", alpha=0.25, label="Train acc (raw)")
    axes[1].plot(epochs, val_acc, "b.-", alpha=0.25, label="Val acc (raw)")
    axes[1].plot(epochs, s_train_acc, "r-", linewidth=2.0, label="Train acc (smooth)")
    axes[1].plot(epochs, s_val_acc, "b-", linewidth=2.0, label="Val acc (smooth)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("acc")
    axes[1].legend()

    overfit_epoch = detect_overfitting_epoch(history)
    if overfit_epoch is not None:
        for ax in axes:
            ax.axvline(overfit_epoch, color="gray", linestyle="--", linewidth=1.2)
        y_anchor = max(val_loss) if val_loss else 0.0
        axes[0].annotate(
            f"开始过拟合：第 {overfit_epoch} 轮",
            xy=(overfit_epoch, y_anchor),
            xytext=(overfit_epoch + 2, y_anchor + 0.03),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9,
            color="gray",
        )

    fig.suptitle("LeNet 训练/验证曲线")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True


def train(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_loader, test_loader = build_dataloaders(cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "best_lenet.pth")
    history_csv_path = os.path.join(cfg.out_dir, "train_history.csv")
    history_plot_path = os.path.join(cfg.out_dir, "train_curve.png")
    preview_dir = os.path.join(cfg.out_dir, "preview_images")
    os.makedirs(preview_dir, exist_ok=True)
    best_acc = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        preview_path = os.path.join(preview_dir, f"epoch_{epoch:03d}.png")
        preview_saved = save_prediction_preview(
            model, test_loader.dataset, device, preview_path, epoch, cfg.seed
        )

        if preview_saved:
            print(f"preview_image={preview_path}")
        else:
            print("preview_image=skipped (matplotlib not available)")

        print(
            f"epoch={epoch:03d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_acc,
                    "epoch": epoch,
                    "cfg": cfg.__dict__,
                },
                best_path,
            )
        scheduler.step()

    save_history_csv(history, history_csv_path)
    plot_saved = save_history_plot(history, history_plot_path)

    print(f"best_val_acc={best_acc:.4f} saved_to={best_path}")
    print(f"history_csv={history_csv_path}")
    if plot_saved:
        print(f"history_plot={history_plot_path}")
    else:
        print("history_plot=skipped (matplotlib not available)")
    return best_path


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train LeNet on Fashion-MNIST (PyTorch).")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data_pytorch"), help="dataset cache dir")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "save_pytorch"), help="model output dir")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=">0 会抬高训练 CE 下限；默认 0 以压低训练损失",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return TrainConfig(
        data_dir=a.data_dir,
        out_dir=a.out_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        label_smoothing=a.label_smoothing,
        num_workers=a.num_workers,
        seed=a.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

