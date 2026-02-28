import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
PROCESSED_DATA_DIR = "C:/BRaTS2021/archive/processed_brats2021/train"
MODEL_SAVE_DIR = "C:/BRaTS2021/archive/vae_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 训练参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先用GPU
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 300
LATENT_DIM = 256
KL_TARGET_WEIGHT = 0.001
SEED = 42

# 固定随机种子（保证结果可复现）
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# -------------------------- 数据集定义 --------------------------
class BraTSDataset(Dataset):
    """BraTS预处理数据集加载类"""

    def __init__(self, data_dir):
        self.data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 加载npy文件
        data = np.load(self.data_paths[idx], allow_pickle=True).item()
        modalities = data["modalities"]  # (4, z, y, x)

        # 转换为torch张量（float32，适配GPU）
        modalities_tensor = torch.from_numpy(modalities).float()

        # 对数据进行padding（保证维度能被下采样/上采样整除）
        # 3D VAE要求输入维度是2^n的倍数，这里统一padding到(4, 144, 192, 176)
        pad_z = 144 - modalities_tensor.shape[1]
        pad_y = 192 - modalities_tensor.shape[2]
        pad_x = 176 - modalities_tensor.shape[3]
        padding = (
            pad_x // 2, pad_x - pad_x // 2,  # x轴padding
            pad_y // 2, pad_y - pad_y // 2,  # y轴padding
            pad_z // 2, pad_z - pad_z // 2  # z轴padding
        )
        modalities_tensor = nn.functional.pad(modalities_tensor, padding, mode="constant", value=0)

        return modalities_tensor


# -------------------------- 3D VAE模型定义 --------------------------
class ResBlock3D(nn.Module):
    """3D残差块：防止梯度消失，保留空间细节"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels)
        )

    def forward(self, x):
        return nn.functional.relu(x + self.conv(x))


class Encoder3D(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 4 -> 32 -> 64 -> 128 -> 256
        self.init_conv = nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.res1 = ResBlock3D(64)
        self.down2 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.res2 = ResBlock3D(128)
        self.down3 = nn.Sequential(nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.res3 = ResBlock3D(256)

        self.gap = nn.AdaptiveAvgPool3d((4, 4, 4))  # 将不同输入的尺寸统一，减小FC层压力
        self.fc_mu = nn.Linear(256 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder3D(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        # 256 -> 128 -> 64 -> 32
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128), nn.ReLU()
        )
        self.res1 = ResBlock3D(128)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU()
        )
        self.res2 = ResBlock3D(64)
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU()
        )
        # 最终层调整到接近原始输入尺寸 (144, 192, 176)
        # 注意：这里需要根据你的Padding后的尺寸微调或使用Interpolation
        self.final_conv = nn.Conv3d(32, 4, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4, 4)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))  # 如果需要更多层，按此模式添加

        # 动态插值到目标尺寸，解决ConvTranspose尺寸不匹配问题
        x = nn.functional.interpolate(x, size=(144, 192, 176), mode='trilinear', align_corners=False)
        return torch.sigmoid(self.final_conv(x))  # 输出[0, 1]

class VAE3D(nn.Module):
    """完整的3D VAE模型"""

    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从正态分布采样z = mu + sigma*epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# -------------------------- 损失函数定义 --------------------------
def vae_loss(recon_x, x, mu, logvar, kl_weight):
    """
    VAE损失：ELBO = 重建损失 + KL散度
    :param recon_x: 重建图像
    :param x: 原始图像
    :param mu: 潜在空间均值
    :param logvar: 潜在空间对数方差
    :param kl_weight: KL散度权重
    :return: 总损失、重建损失、KL损失
    """
    # 重建损失：MSE（匹配输入和重建的像素值）
    recon_loss = nn.MSELoss()(recon_x, x)

    # KL散度：衡量潜在分布与标准正态分布的差异
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (x.size(0) * mu.size(1))  # 按批次大小归一化

    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# -------------------------- 训练流程 --------------------------
def train_vae():
    # 1. 加载数据集
    dataset = BraTSDataset(PROCESSED_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Windows下num_workers=0
    print(f"训练集样本数：{len(dataset)}，批次大小：{BATCH_SIZE}，总批次数：{len(dataloader)}")

    # 2. 初始化模型、优化器
    model = VAE3D(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 早停策略（避免过拟合）
    best_recon_loss = float("inf")
    patience = 10  # 连续10轮无提升则停止
    patience_counter = 0

    # 4. 训练记录
    train_losses = []
    recon_losses = []
    kl_losses = []

    # 5. 开始训练
    model.train()
    for epoch in range(EPOCHS):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)

            # 前向传播
            recon_batch, mu, logvar = model(batch)
            total_loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, KL_WEIGHT)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            # 更新进度条
            pbar.set_postfix({
                "total_loss": f"{total_loss.item():.4f}",
                "recon_loss": f"{recon_loss.item():.4f}",
                "kl_loss": f"{kl_loss.item():.4f}"
            })

        # 计算本轮平均损失
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)

        train_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        print(
            f"Epoch {epoch + 1} | 平均总损失：{avg_total_loss:.4f} | 平均重建损失：{avg_recon_loss:.4f} | 平均KL损失：{avg_kl_loss:.4f}")

        # 早停判断
        if avg_recon_loss < best_recon_loss:
            best_recon_loss = avg_recon_loss
            patience_counter = 0
            # 保存最优模型
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_recon_loss": best_recon_loss
            }, os.path.join(MODEL_SAVE_DIR, "best_vae3d.pth"))
            print(f"保存最优模型（重建损失：{best_recon_loss:.4f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"连续{patience}轮重建损失无提升，触发早停")
                break

    # 6. 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss Curve")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label="Recon Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Curve")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label="KL Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Loss Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "loss_curves.png"))
    plt.show()

    return model


# -------------------------- 生成样本验证 --------------------------
def generate_samples(model_path):
    """加载训练好的模型，生成合成MRI样本并可视化"""
    # 1. 加载模型
    model = VAE3D(LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. 加载一个真实样本作为参考
    dataset = BraTSDataset(PROCESSED_DATA_DIR)
    real_sample = dataset[0].unsqueeze(0).to(DEVICE)  # (1, 4, 144, 192, 176)

    # 3. 生成重建样本
    with torch.no_grad():
        recon_sample, _, _ = model(real_sample)
        # 随机生成新样本（从标准正态分布采样z）
        random_z = torch.randn(1, LATENT_DIM).to(DEVICE)
        gen_sample = model.decoder(random_z)

    # 4. 转换为numpy数组，可视化（取第80层）
    slice_idx = 80
    real_np = real_sample.cpu().numpy()[0]  # (4, 144, 192, 176)
    recon_np = recon_sample.cpu().numpy()[0]
    gen_np = gen_sample.cpu().numpy()[0]

    # 可视化4种模态的真实/重建/生成样本
    modalities = ["T1", "T1CE", "T2", "FLAIR"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i, mod in enumerate(modalities):
        # 真实样本
        axes[0, i].imshow(real_np[i, slice_idx], cmap="gray")
        axes[0, i].set_title(f"Real - {mod}")
        axes[0, i].axis("off")

        # 重建样本
        axes[1, i].imshow(recon_np[i, slice_idx], cmap="gray")
        axes[1, i].set_title(f"Reconstructed - {mod}")
        axes[1, i].axis("off")

        # 生成样本
        axes[2, i].imshow(gen_np[i, slice_idx], cmap="gray")
        axes[2, i].set_title(f"Generated - {mod}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "generated_samples.png"))
    plt.show()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 训练VAE模型
    trained_model = train_vae()

    # 生成样本验证
    generate_samples(os.path.join(MODEL_SAVE_DIR, "best_vae3d.pth"))