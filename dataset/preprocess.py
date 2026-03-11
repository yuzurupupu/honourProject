import os
import json
import random
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool

# -------------------------- 配置 --------------------------

RAW_DATA_DIR = "C:/BRaTS2021/archive/BraTS2021_Training_Data"
PROCESSED_DATA_DIR = "C:/BRaTS2021/archive/processed_brats2021"

TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPE = (128, 160, 160)

TRAIN_NUM = 120
TEST_NUM = 30

SEED = 42

MODALITIES = ["t1", "t1ce", "t2", "flair"]


# -------------------------- IO --------------------------

def load_nii(path):

    img = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()

    return array, spacing


# -------------------------- brain crop --------------------------

def brain_crop(volume):

    coords = np.where(volume > 0)

    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()

    return volume[zmin:zmax, ymin:ymax, xmin:xmax]


# -------------------------- resample --------------------------

def resample_image(volume, original_spacing):

    itk_image = sitk.GetImageFromArray(volume)
    itk_image.SetSpacing(original_spacing)

    original_size = itk_image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / TARGET_SPACING[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)

    resampled = resampler.Execute(itk_image)

    return sitk.GetArrayFromImage(resampled)


# -------------------------- normalization --------------------------

def normalize_intensity(volume):

    mask = volume > 0

    vals = volume[mask]

    lower, upper = np.percentile(vals, [1, 99])

    volume = np.clip(volume, lower, upper)

    volume = (volume - lower) / (upper - lower + 1e-8)

    volume[~mask] = 0

    # GAN推荐 [-1,1]
    volume = volume * 2 - 1

    return volume.astype(np.float32)


# -------------------------- center crop --------------------------

def center_crop_or_pad(arr):

    tz, ty, tx = TARGET_SHAPE
    z, y, x = arr.shape

    pad_z = max(0, tz - z)
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)

    arr = np.pad(
        arr,
        (
            (pad_z//2, pad_z - pad_z//2),
            (pad_y//2, pad_y - pad_y//2),
            (pad_x//2, pad_x - pad_x//2)
        ),
        mode="constant"
    )

    z, y, x = arr.shape

    sz = (z - tz)//2
    sy = (y - ty)//2
    sx = (x - tx)//2

    return arr[sz:sz+tz, sy:sy+ty, sx:sx+tx]


# -------------------------- 单病例处理 --------------------------

def process_case(case_id):

    case_dir = os.path.join(RAW_DATA_DIR, case_id)

    volumes = []

    for mod in MODALITIES:

        path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")

        if not os.path.exists(path):
            print("missing", path)
            return None

        arr, spacing = load_nii(path)

        arr = brain_crop(arr)

        arr = resample_image(arr, spacing)

        arr = normalize_intensity(arr)

        arr = center_crop_or_pad(arr)

        volumes.append(arr)

    volumes = np.stack(volumes, axis=0)

    return volumes


# -------------------------- 数据划分 --------------------------

def split_dataset():

    case_ids = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if d.startswith("BraTS2021")
    ])

    random.seed(SEED)
    random.shuffle(case_ids)

    train_ids = case_ids[:TRAIN_NUM]
    test_ids = case_ids[TRAIN_NUM:TRAIN_NUM + TEST_NUM]

    record = {
        "seed": SEED,
        "train_ids": train_ids,
        "test_ids": test_ids
    }

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    with open(os.path.join(PROCESSED_DATA_DIR, "split.json"), "w") as f:

        json.dump(record, f, indent=4)

    return train_ids, test_ids


# -------------------------- 主流程 --------------------------

def preprocess_split(case_ids, split):

    save_dir = os.path.join(PROCESSED_DATA_DIR, split)

    os.makedirs(save_dir, exist_ok=True)

    for case in tqdm(case_ids):

        vol = process_case(case)

        if vol is None:
            continue

        np.save(
            os.path.join(save_dir, case + ".npy"),
            vol
        )


def preprocess():

    train_ids, test_ids = split_dataset()

    preprocess_split(train_ids, "train")

    preprocess_split(test_ids, "test")


# --------------------------

if __name__ == "__main__":

    preprocess()