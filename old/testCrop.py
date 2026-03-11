import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def verify_crop_160_128(raw_data_dir):
    # 1. 随机找一个病例
    case_id = [d for d in os.listdir(raw_data_dir) if d.startswith("BraTS2021_")][0]
    file_path = os.path.join(raw_data_dir, case_id, f"{case_id}_flair.nii.gz")

    # 2. 加载并获取数组 (z, y, x)
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)
    z, y, x = arr.shape  # 原始通常是 (155, 240, 240)

    # 3. 计算中心裁剪坐标 (目标: 128, 160, 160)
    tz, ty, tx = 128, 160, 160
    sz = max((z - tz) // 2, 0)
    sy = max((y - ty) // 2, 0)
    sx = max((x - tx) // 2, 0)

    cropped = arr[sz:sz + tz, sy:sy + ty, sx:sx + tx]

    # 4. 可视化三个轴向的切面
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    mid_z, mid_y, mid_x = tz // 2, ty // 2, tx // 2

    # 原始图像展示
    axes[0, 0].imshow(arr[z // 2], cmap='gray');
    axes[0, 0].set_title(f"Original Axial (Z)")
    axes[0, 1].imshow(arr[:, y // 2, :], cmap='gray');
    axes[0, 1].set_title("Original Coronal (Y)")
    axes[0, 2].imshow(arr[:, :, x // 2], cmap='gray');
    axes[0, 2].set_title("Original Sagittal (X)")

    # 裁剪结果展示
    axes[1, 0].imshow(cropped[mid_z], cmap='gray');
    axes[1, 0].set_title(f"Cropped 160x160 (Z)")
    axes[1, 1].imshow(cropped[:, mid_y, :], cmap='gray');
    axes[1, 1].set_title("Cropped 160x128 (Y)")
    axes[1, 2].imshow(cropped[:, :, mid_x], cmap='gray');
    axes[1, 2].set_title("Cropped 160x128 (X)")

    plt.suptitle(f"Case: {case_id} | Crop Size: {cropped.shape}")
    plt.show()


verify_crop_160_128("C:/BRaTS2021/archive/BraTS2021_Training_Data")
