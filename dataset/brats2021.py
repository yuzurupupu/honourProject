import os
import torch
import numpy as np
from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(self, root):

        self.files = [
            os.path.join(root,f)
            for f in os.listdir(root)
            if f.endswith(".npy")
        ]

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):

        x = np.load(self.files[idx])

        x = torch.from_numpy(x).float()

        return x