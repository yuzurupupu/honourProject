import os
import matplotlib.pyplot as plt


def save_slice(volume, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    z = volume.shape[0] // 2
    slice_img = volume[z]

    plt.figure(figsize=(5, 5))
    plt.imshow(slice_img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()