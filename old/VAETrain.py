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
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 300
LATENT_DIM = 256
KL_TARGET_WEIGHT = 0.1  # 修正1：提高KL目标权重（量级对齐后）
SEED = 42

# 固定随机种子（保证结果可复现）
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True  # 新增：固定cudnn，避免数值波动
    torch.backends.cudnn.benchmark = False


# -------------------------- 数据集定义 --------------------------
class BraTSDataset(Dataset):
    """BraTS预处理数据集加载类"""

    def __init__(self, data_dir):
        self.data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        # 新增：检查数据集是否为空
        if len(self.data_paths) == 0:
            raise ValueError(f"未找到.npy文件，请检查路径：{data_dir}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 加载npy文件
        data = np.load(self.data_paths[idx], allow_pickle=True).item()
        modalities = data["modalities"]  # 预期形状: (4, 128, 160, 160)

        # 转换为torch张量（预处理已归一化，此处无需重复操作）
        modalities_tensor = torch.from_numpy(modalities).float()
        # 新增：防止归一化后仍有极端值（比如nan/inf）
        modalities_tensor = torch.clamp(modalities_tensor, 0.0, 1.0)
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
        self.down4 = nn.Sequential(nn.Conv3d(256, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.res4 = ResBlock3D(256)

        self.fc_mu = nn.Linear(256 * 8 * 10 * 10, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 10 * 10, latent_dim)

        # 新增：约束logvar范围，防止exp溢出
        self.logvar_clamp = lambda x: torch.clamp(x, min=-10.0, max=10.0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        x = self.res4(self.down4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.logvar_clamp(self.fc_logvar(x))  # 约束logvar范围
        return mu, logvar


class Decoder3D(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 10 * 10)

        # 256 -> 128 -> 64 -> 32
        self.up4 = nn.Sequential(nn.ConvTranspose3d(256, 256, 3, stride=2, padding=1, output_padding=1),nn.BatchNorm3d(256), nn.ReLU())
        self.res4 = ResBlock3D(256)  # 新增残差块
        # 第3次上采样：256→128，维度16×20×20→32×40×40
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128), nn.ReLU()
        )
        self.res3 = ResBlock3D(128)
        # 第2次上采样：128→64，维度32×40×40→64×80×80
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU()
        )
        self.res2 = ResBlock3D(64)
        # 第1次上采样：64→32，维度64×80×80→128×160×160
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU()
        )
        self.res1 = ResBlock3D(32)

        # 最终层：32→4，恢复4个模态，尺寸已完美匹配128×160×160
        self.final_conv = nn.Conv3d(32, 4, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 8, 10, 10)
        x = self.res4(self.up4(x))
        x = self.res3(self.up3(x))
        x = self.res2(self.up2(x))
        x = self.res1(self.up1(x))
        return torch.sigmoid(self.final_conv(x))


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
    修正版VAE损失：
    1. 重建损失：按总像素数求平均（量级≈0.01~0.1）
    2. KL损失：按每个样本的latent_dim求平均，再按批次平均（量级≈0.1~1.0）
    3. 新增数值稳定处理
    """
    # 1. 重建损失：对所有像素取平均（等价于sum/x.numel()）
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)

    # 2. KL损失：数值稳定版 + 按维度平均
    # 步骤1：逐元素计算KL项，防止整体溢出
    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    # 步骤2：限制KL元素范围，避免极端值
    kl_element = torch.clamp(kl_element, min=-1e6, max=1e6)
    # 步骤3：按每个样本的latent_dim求平均，再按批次平均
    kl_per_sample = -0.5 * torch.mean(kl_element, dim=1)
    kl_loss = torch.mean(kl_per_sample)

    # 3. 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# -------------------------- 训练流程 --------------------------
def train_vae():
    # 1. 加载数据集
    dataset = BraTSDataset(PROCESSED_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"训练集样本数：{len(dataset)}，批次大小：{BATCH_SIZE}，总批次数：{len(dataloader)}")

    # 2. 初始化模型、优化器
    model = VAE3D(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    grad_clip_norm = 1.0

    # 3. 续训关键：定义检查点路径（保存最新训练状态）
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "latest_ckpt.pth")
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_vae3d.pth")
    start_epoch = 0  # 初始开始轮次
    best_recon_loss = float("inf")  # 初始最佳损失
    train_losses, recon_losses, kl_losses = [], [], []  # 初始损失记录

    # 4. 加载检查点（如果存在，自动续训）
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]  # 恢复上次中断的轮次
        best_recon_loss = checkpoint["best_recon_loss"]  # 恢复最佳损失
        train_losses = checkpoint["train_losses"]  # 恢复损失记录
        recon_losses = checkpoint["recon_losses"]
        kl_losses = checkpoint["kl_losses"]
        print(f"✅ 找到检查点，从第 {start_epoch + 1} 轮继续训练！")
    else:
        print("🔍 未找到检查点，从头开始训练")

    # 5. 早停策略（基于恢复的best_recon_loss）
    patience = 15
    patience_counter = 0

    # 6. 开始训练（从start_epoch开始，而非0）
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        # KL权重暖启动（轮次从恢复后开始计算，不影响）
        kl_warmup_epochs = 50
        current_kl_weight = KL_TARGET_WEIGHT * min(1.0, (epoch + 1) / kl_warmup_epochs)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            # 前向传播
            recon_batch, mu, logvar = model(batch)
            total_loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, current_kl_weight)
            # 反向传播+梯度裁剪
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            # 更新进度条
            pbar.set_postfix({
                "total_loss": f"{total_loss.item():.4f}",
                "recon_loss": f"{recon_loss.item():.4f}",
                "kl_loss": f"{kl_loss.item():.4f}",
                "kl_w": f"{current_kl_weight:.4f}"
            })

        # 计算本轮平均损失
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        train_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        print(
            f"Epoch {epoch + 1} | 平均总损失：{avg_total_loss:.4f} | 平均重建损失：{avg_recon_loss:.4f} | 平均KL损失：{avg_kl_loss:.4f}"
        )

        # 7. 保存最新检查点（每轮都存，覆盖式，中断后用这个续训）
        torch.save({
            "epoch": epoch + 1,  # 保存当前完成的轮次
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_recon_loss": best_recon_loss,
            "train_losses": train_losses,
            "recon_losses": recon_losses,
            "kl_losses": kl_losses
        }, checkpoint_path)
        print(f"💾 已保存最新检查点到：{checkpoint_path}")

        # 8. 保存最佳模型（按重建损失，只存最优的）
        if avg_recon_loss < best_recon_loss:
            best_recon_loss = avg_recon_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_recon_loss": best_recon_loss
            }, best_model_path)
            print(f"🏆 保存最优模型（重建损失：{best_recon_loss:.4f}）到：{best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ 连续{patience}轮重建损失无提升，触发早停")
                break

    # 9. 绘制损失曲线（基于恢复+新增的损失记录）
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
    real_sample = dataset[0].unsqueeze(0).to(DEVICE)  # (1, 4, 128, 160, 160)

    # 3. 生成重建样本
    with torch.no_grad():
        recon_sample, _, _ = model(real_sample)
        # 随机生成新样本（从标准正态分布采样z）
        random_z = torch.randn(1, LATENT_DIM).to(DEVICE)
        gen_sample = model.decoder(random_z)

    # 4. 转换为numpy数组，可视化（取第64层）
    slice_idx = 64
    real_np = real_sample.cpu().numpy()[0]  # (4, 128, 160, 160)
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