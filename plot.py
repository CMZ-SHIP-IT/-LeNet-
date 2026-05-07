import argparse
import os
import random

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 64 Fashion-MNIST samples.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data_pytorch"),
        help="Fashion-MNIST cache directory",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "save_pytorch", "fashion_mnist_64_samples.png"),
        help="Output image path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    ds = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    indices = random.sample(range(len(ds)), 64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle("Fashion-MNIST 64 Samples (with Labels)", fontsize=14)

    for i, idx in enumerate(indices):
        img, label = ds[idx]
        ax = axes[i // 8, i % 8]
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.set_title(CLASS_NAMES[int(label)], fontsize=7)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=180)
    plt.show()
    print(f"saved_figure={args.out}")
    print("你可以根据图中图像-标签对应关系，人工检查是否存在明显标签噪声。")


if __name__ == "__main__":
    main()

