import argparse
import os
import torch
from torchvision import datasets, transforms
from lenet_model import LeNet5

CLASSES = [
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

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained LeNet checkpoint on Fashion-MNIST.")
    p.add_argument(
        "--ckpt",
        default=os.path.join(os.path.dirname(__file__), "save_pytorch", "best_lenet.pth"),
        help="path to checkpoint produced by lenet_train.py",
    )
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data_pytorch"), help="dataset cache dir")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = LeNet5().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    test_ds = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=tfm)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / max(total, 1)
    print(f"test_acc={acc:.4f} ({correct}/{total})")
    cfg_saved = ckpt.get("cfg")
    if "val_acc" in ckpt:
        print(
            f"ckpt_best_val_acc={ckpt['val_acc']:.4f} "
            f"ckpt_best_at_epoch={ckpt.get('epoch')} (验证集最佳时的轮次)"
        )
    elif "test_acc" in ckpt:
        print(f"ckpt_best_acc={ckpt['test_acc']:.4f} ckpt_best_at_epoch={ckpt.get('epoch')}")


if __name__ == "__main__":
    main()

