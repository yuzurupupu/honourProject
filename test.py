import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------- 配置参数 --------------------------
PROCESSED_DATA_DIR = "C:/BRaTS2021/archive/processed_brats2021/train"
GENERATED_DATA_DIR = "C:/BRaTS2021/archive/vae_gan_generated"  # VAE/GAN生成的样本路径
MODEL_SAVE_DIR = "C:/BRaTS2021/archive/unet_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # 3D U-Net显存占用高，强制批次1
LEARNING_RATE = 1e-4
EPOCHS = 50  # 分割模型训练无需太多轮次
SEED = 42
DICE_THRESHOLD = 0.5  # 肿瘤分割Dice系数阈值

# 显存优化
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------- 数据集定义（含原始/增强样本加载） --------------------------
class BraTSDataset(Dataset):
    def __init__(self, data_dir, generated_dir=None, use_aug=False):
        """
        Args:
            data_dir: 原始样本路径
            generated_dir: 生成样本路径
            use_aug: 是否使用增强样本（True=原始+生成，False=仅原始）
        """
        # 加载原始样本
        self.data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.use_aug = use_aug

        # 加载生成样本（增强模式）
        if use_aug and generated_dir and os.path.exists(generated_dir):
            generated_paths = [os.path.join(generated_dir, f) for f in os.listdir(generated_dir) if f.endswith(".npy")]
            self.data_paths += generated_paths  # 合并原始+生成样本

        if len(self.data_paths) == 0:
            raise ValueError("未找到有效样本，请检查路径！")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx], allow_pickle=True).item()
        modalities = torch.from_numpy(data["modalities"]).float()  # (4, 128, 160, 160)
        seg_mask = torch.from_numpy(data["seg_mask"]).float()  # 肿瘤分割掩码 (1, 128, 160, 160)

        modalities = torch.clamp(modalities, 0.0, 1.0)
        seg_mask = torch.clamp(seg_mask, 0.0, 1.0)  # 掩码二值化（0=背景，1=肿瘤）
        return modalities, seg_mask


# -------------------------- 3D U-Net核心模块 --------------------------
class DoubleConv3D(nn.Module):
    """3D U-Net基本单元：两次3D卷积+BN+ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """下采样：MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """上采样：转置卷积 + 拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接（保证尺寸匹配）
        diff = x2.size()[2:] - x1.size()[2:]
        x1 = nn.functional.pad(x1, [diff[2] // 2, diff[2] - diff[2] // 2,
                                    diff[1] // 2, diff[1] - diff[1] // 2,
                                    diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """输出层：1×1×1卷积映射到分割通道"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))  # 二分类用sigmoid


# -------------------------- 3D U-Net完整模型 --------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv3D(in_channels, 32)  # 轻量化：64→32
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        self.up1 = Up3D(256, 128)
        self.up2 = Up3D(128, 64)
        self.up3 = Up3D(64, 32)
        self.outc = OutConv3D(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# -------------------------- 评估指标：Dice系数（医学分割金标准） --------------------------
def dice_coeff(pred, target, threshold=DICE_THRESHOLD):
    """计算Dice系数（越接近1越好）"""
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0  # 无肿瘤时Dice=1
    return (2. * intersection) / union


# -------------------------- 训练函数（含对比实验） --------------------------
def train_unet(use_aug=False):
    """
    训练U-Net：
    use_aug=False → 仅原始样本（对照组）
    use_aug=True → 原始+生成样本（实验组）
    """
    # 加载数据集
    dataset = BraTSDataset(PROCESSED_DATA_DIR, GENERATED_DATA_DIR, use_aug)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"训练集样本数：{len(dataset)}（{'原始+增强' if use_aug else '仅原始'}）")

    # 初始化模型、优化器、损失函数
    model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()  # 二分类交叉熵（适配分割掩码）

    # 记录训练过程
    train_loss = []
    train_dice = []
    best_dice = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} (Aug: {use_aug})")
        for modalities, seg_mask in pbar:
            modalities = modalities.to(DEVICE)
            seg_mask = seg_mask.to(DEVICE)

            # 前向传播
            pred_mask = model(modalities)
            loss = criterion(pred_mask, seg_mask)
            dice = dice_coeff(pred_mask, seg_mask)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计指标
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            pbar.set_postfix({"Loss": loss.item(), "Dice": dice.item()})

            # 显存清理
            torch.cuda.empty_cache()

        # 计算平均指标
        avg_loss = epoch_loss / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        train_loss.append(avg_loss)
        train_dice.append(avg_dice)

        print(f"Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 平均Dice：{avg_dice:.4f}")

        # 保存最优模型
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "use_aug": use_aug
            }, os.path.join(MODEL_SAVE_DIR, f"best_unet_{'aug' if use_aug else 'raw'}.pth"))
            print(f"保存最优模型，Dice：{best_dice:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"U-Net训练损失 (Aug: {use_aug})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_dice, label="Train Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title(f"U-Net训练Dice (Aug: {use_aug})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f"unet_train_curve_{'aug' if use_aug else 'raw'}.png"))
    plt.show()

    return best_dice


# -------------------------- 对比实验主函数 --------------------------
if __name__ == "__main__":
    # 1. 训练对照组：仅原始样本
    print("===== 训练对照组：仅原始样本 =====")
    dice_raw = train_unet(use_aug=False)

    # 2. 训练实验组：原始+增强样本
    print("\n===== 训练实验组：原始+增强样本 =====")
    dice_aug = train_unet(use_aug=True)

    # 3. 输出对比结果
    print("\n===== 数据增强效果对比 =====")
    print(f"仅原始样本 U-Net Dice：{dice_raw:.4f}")
    print(f"原始+增强样本 U-Net Dice：{dice_aug:.4f}")
    print(f"Dice提升：{(dice_aug - dice_raw):.4f} ({(dice_aug - dice_raw) / dice_raw * 100:.2f}%)")

    # 保存对比结果
    with open(os.path.join(MODEL_SAVE_DIR, "augmentation_result.txt"), "w") as f:
        f.write(f"仅原始样本 Dice：{dice_raw:.4f}\n")
        f.write(f"原始+增强样本 Dice：{dice_aug:.4f}\n")
        f.write(f"提升幅度：{(dice_aug - dice_raw) / dice_raw * 100:.2f}%\n")