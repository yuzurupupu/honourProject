import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, root, modality="flair"):
        self.root = root
        self.modality = modality
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def _resize_3d(self, arr, target_shape=(64, 64, 64)):
        import scipy.ndimage
        zoom = [t / s for t, s in zip(target_shape, arr.shape)]
        return scipy.ndimage.zoom(arr, zoom=zoom, order=1)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx])

        modal_map = {
            "t1": 0,
            "t1ce": 1,
            "t2": 2,
            "flair": 3,
        }
        vol = vol[modal_map[self.modality]]   # -> [Z, Y, X]

        vol = self._resize_3d(vol, (64, 64, 64)).astype(np.float32)

        # 保证范围在 [-1, 1]
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
            vol = vol * 2.0 - 1.0

        vol = torch.from_numpy(vol).unsqueeze(0).float()  # [1,64,64,64]
        return vol