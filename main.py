import json
import os
import random
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
RAW_DATA_DIR = "C:/BRaTS2021/archive/BraTS2021_Training_Data"
PROCESSED_DATA_DIR = "C:/BRaTS2021/archive/processed_brats2021"
SPLIT_RECORD_PATH = os.path.join(PROCESSED_DATA_DIR, "dataset_split_record.json")
TARGET_RESOLUTION = (1.0, 1.0, 1.0)  # 目标分辨率：1mm³
TRAIN_NUM = 120  # 小样本训练集数量
TEST_NUM = 30  # 测试集数量
SEED = 42  # 随机种子


# -------------------------- 工具函数 --------------------------
def load_nii_image(file_path):
    """加载nii.gz格式的3D MRI图像"""
    itk_image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(itk_image)  # (z, y, x)
    spacing = itk_image.GetSpacing()  # (x, y, z)，注意顺序！
    origin = itk_image.GetOrigin()
    direction = itk_image.GetDirection()
    return image_array, spacing, origin, direction


def resample_image(image_array, original_spacing, target_spacing, interpolator=sitk.sitkLinear):
    """重采样到目标分辨率"""
    itk_image = sitk.GetImageFromArray(image_array)
    itk_image.SetSpacing(original_spacing[::-1])  # 匹配(z,y,x)和(x,y,z)的顺序

    # 计算重采样参数
    original_size = itk_image.GetSize()
    target_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetInterpolator(interpolator)

    resampled_image = resampler.Execute(itk_image)
    return sitk.GetArrayFromImage(resampled_image)

#Zscore改成min-max归一化
def normalize_intensity(image_array):
    mask = image_array > 0
    if not np.any(mask): return image_array

    vals = image_array[mask]
    # 鲁棒归一化：去除 1% 的极端值
    lower, upper = np.percentile(vals, [1, 99])
    image_array = np.clip(image_array, lower, upper)

    # 映射到 [0, 1]
    normalized = (image_array - lower) / (upper - lower + 1e-8)
    normalized[~mask] = 0
    return normalized

def center_crop_to_fixed(arr, target_shape=(128, 160, 160)):
    z, y, x = arr.shape
    tz, ty, tx = target_shape
    sz, sy, sx = (z-tz)//2, (y-ty)//2, (x-tx)//2
    # 确保不越界并处理边际
    sz, sy, sx = max(0, sz), max(0, sy), max(0, sx)
    return arr[sz:sz+tz, sy:sy+ty, sx:sx+tx]

def save_dataset_split(record_path, train_ids, test_ids, seed, train_num, test_num):
    """保存数据集划分结果"""
    split_record = {
        "seed": seed,
        "train_num": train_num,
        "test_num": test_num,
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
        "create_time": os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip()  # 记录生成时间
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    # 保存为JSON文件（易读、易解析）
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(split_record, f, indent=4)
    print(f"数据集划分结果已保存至：{record_path}")

# -------------------------- 主预处理流程 --------------------------
def preprocess_brats():
    # 创建保存目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, "test"), exist_ok=True)

    # 获取所有病例ID并打乱
    case_ids = sorted([d for d in os.listdir(RAW_DATA_DIR) if d.startswith("BraTS2021_")])
    random.seed(SEED)
    random.shuffle(case_ids)

    # 划分训练/测试集
    train_case_ids = case_ids[:TRAIN_NUM]
    test_case_ids = case_ids[TRAIN_NUM:TRAIN_NUM + TEST_NUM]
    save_dataset_split(
        record_path=SPLIT_RECORD_PATH,
        train_ids=train_case_ids,
        test_ids=test_case_ids,
        seed=SEED,
        train_num=TRAIN_NUM,
        test_num=TEST_NUM
    )
    print(f"训练集病例数：{len(train_case_ids)}, 测试集病例数：{len(test_case_ids)}")

    # 批量预处理
    for split, case_list in [("train", train_case_ids), ("test", test_case_ids)]:
        save_dir = os.path.join(PROCESSED_DATA_DIR, split)
        os.makedirs(save_dir, exist_ok=True)

        for case_id in tqdm(case_list, desc=f"预处理{split}集"):
            case_dir = os.path.join(RAW_DATA_DIR, case_id)

            # 加载4种模态
            modalities = ["t1", "t1ce", "t2", "flair"]
            modal_arrays = []
            modal_spacings = []
            for mod in modalities:
                mod_path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
                if not os.path.exists(mod_path):
                    print(f"警告：{case_id} 缺少 {mod} 模态，跳过该病例")
                    break
                arr, spacing, _, _ = load_nii_image(mod_path)
                modal_arrays.append(arr)
                modal_spacings.append(spacing)
            if len(modal_arrays) != 4:
                continue  # 跳过模态不全的病例

            # 加载掩码
            seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
            if not os.path.exists(seg_path):
                print(f"警告：{case_id} 缺少掩码，跳过该病例")
                continue
            seg_arr, seg_spacing, _, _ = load_nii_image(seg_path)

            # 统一重采样（用第一个模态的spacing作为基准，保证所有数据一致）
            target_spacing = TARGET_RESOLUTION
            base_spacing = modal_spacings[0]  # 统一基准spacing
            resampled_modals = []
            for arr in modal_arrays:
                resampled = resample_image(arr, base_spacing, target_spacing)
                resampled_modals.append(resampled)
            # 掩码重采样用最近邻插值
            resampled_seg = resample_image(seg_arr, base_spacing, target_spacing, interpolator=sitk.sitkNearestNeighbor)

            # 强度归一化
            normalized_modals = [normalize_intensity(arr) for arr in resampled_modals]
            cropped_modals = [center_crop_to_fixed(arr) for arr in normalized_modals]
            cropped_seg = center_crop_to_fixed(resampled_seg)

            # 合并4种模态为一个数组（shape: (4, z, y, x)）
            combined_modals = np.stack(cropped_modals, axis=0)

            # 保存为npy格式
            save_path = os.path.join(save_dir, f"{case_id}.npy")
            np.save(save_path, {
                "modalities": combined_modals,
                "segmentation": cropped_seg
            })

if __name__ == "__main__":
    preprocess_brats()