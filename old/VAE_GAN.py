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
MODEL_SAVE_DIR = "C:/BRaTS2021/archive/vae_gan_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
EPOCHS = 100
LATENT_DIM = 128
LAMBDA1 = 10.0  # WGAN-GP梯度惩罚权重
LAMBDA2 = 10.0  # L1重建损失权重
SEED = 42

# 硬件保护
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# -------------------------- 数据集定义 --------------------------
class BraTSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        if len(self.data_paths) == 0:
            raise ValueError(f"未找到.npy文件，请检查路径：{data_dir}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx], allow_pickle=True).item()
        modalities = data["modalities"]  # (4, 128, 160, 160)
        modalities_tensor = torch.from_numpy(modalities).float()
        modalities_tensor = torch.clamp(modalities_tensor, 0.0, 1.0)
        return modalities_tensor


# -------------------------- 残差块 --------------------------
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels // 8),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels // 8),
            nn.BatchNorm3d(channels)
        )

    def forward(self, x):
        return nn.functional.relu(x + self.conv(x))


# -------------------------- 原有VAE模块 --------------------------
class Encoder3D(nn.Module):  # 复用，作为α-GAN的编码器
    def __init__(self, latent_dim):
        super().__init__()
        self.init_conv = nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2))
        self.res1 = ResBlock3D(64)
        self.down2 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2))
        self.res2 = ResBlock3D(128)
        self.down3 = nn.Sequential(nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2))
        self.res3 = ResBlock3D(256)
        self.down4 = nn.Sequential(nn.Conv3d(256, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2))
        self.res4 = ResBlock3D(256)
        self.fc_z = nn.Linear(256 * 8 * 10 * 10, latent_dim)  # 输出潜在向量z_e
        self.logvar_clamp = lambda x: torch.clamp(x, min=-10.0, max=10.0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        x = self.res4(self.down4(x))
        x = x.view(x.size(0), -1)
        z_e = self.fc_z(x)
        return z_e


class Generator3D(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 10 * 10)
        # Resize-Upscale + 3×3×3卷积（避免棋盘格伪影）
        self.block1 = nn.Sequential(  # 8×10×10 → 16×20×20
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock3D(256)
        self.block2 = nn.Sequential(  # 16×20×20 → 32×40×40
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResBlock3D(128)
        self.block3 = nn.Sequential(  # 32×40×40 → 64×80×80
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.res3 = ResBlock3D(64)
        self.block4 = nn.Sequential(  # 64×80×80 → 128×160×160
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.res4 = ResBlock3D(32)
        # 最终层：无BN，用Sigmoid
        self.final_conv = nn.Conv3d(32, 4, 3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 8, 10, 10)
        x = self.res1(self.block1(x))
        x = self.res2(self.block2(x))
        x = self.res3(self.block3(x))
        x = self.res4(self.block4(x))
        return torch.sigmoid(self.final_conv(x))


# -------------------------- GAN新增模块 --------------------------
class Discriminator3D(nn.Module):  # 5层3D卷积，区分真实/生成图像
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入：(4, 128, 160, 160)
            nn.Conv3d(4, 64, 4, stride=2, padding=1, bias=False),  # → (64,64,80,80)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),  # → (128,32,40,40)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),  # → (256,16,20,20)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, stride=2, padding=1, bias=False),  # → (512,8,10,10)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, 4, stride=1, padding=0),  # → (1,5,7,7)，最终输出单值
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # 展平为(batch_size, 1)


class CodeDiscriminator(nn.Module):  # 3层全连接，区分z_e（编码器）和z_r（随机）
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.model(z).view(-1, 1)


# -------------------------- 损失函数（严格遵循论文WGAN-GP） --------------------------
def gradient_penalty(critic, real, fake, device):
    """WGAN-GP梯度惩罚项，约束1-Lipschitz条件"""
    alpha = torch.randn((real.size(0), 1, 1, 1, 1), device=device)
    interpolated = alpha * real + (1 - alpha) * fake  # 插值：real和fake之间的样本
    interpolated.requires_grad_(True)
    critic_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)


def discriminator_loss(D, G, E, x_real, z_r, device):
    """D损失 = WGAN损失 + 梯度惩罚"""
    z_e = E(x_real)
    x_rec = G(z_e)  # 重建图像（fake1）
    x_gen = G(z_r)  # 生成图像（fake2）

    # WGAN损失：E[D(fake)] - E[D(real)]
    d_rec = D(x_rec)
    d_gen = D(x_gen)
    d_real = D(x_real)
    wgan_loss = torch.mean(d_rec) + torch.mean(d_gen) - 2 * torch.mean(d_real)

    # 梯度惩罚（对D的输入插值）
    gp = gradient_penalty(D, x_real.data, x_rec.data, device) + gradient_penalty(D, x_real.data, x_gen.data, device)
    return wgan_loss + LAMBDA1 * gp


def generator_loss(D, G, E, x_real, z_r, device):
    """G损失 = -WGAN损失 + L1重建损失"""
    z_e = E(x_real)
    x_rec = G(z_e)
    x_gen = G(z_r)

    d_rec = D(x_rec)
    d_gen = D(x_gen)
    wgan_loss = - (torch.mean(d_rec) + torch.mean(d_gen))
    l1_loss = torch.mean(torch.abs(x_real - x_rec))  # L1重建损失（比MSE更保边缘）
    return wgan_loss + LAMBDA2 * l1_loss


def code_discriminator_loss(C, E, z_r, x_real, device):
    """Code Discriminator损失 = WGAN损失 + 梯度惩罚"""
    z_e = E(x_real)  # fake z
    # WGAN损失：E[C(fake z)] - E[C(real z)]
    c_fake = C(z_e)
    c_real = C(z_r)
    wgan_loss = torch.mean(c_fake) - torch.mean(c_real)
    # 梯度惩罚（对C的输入插值）
    gp = gradient_penalty(C, z_r.data, z_e.data, device)
    return wgan_loss + LAMBDA1 * gp


def encoder_loss(C, E, z_r, x_real, device):
    """Encoder损失 = -Code Discriminator的WGAN损失"""
    z_e = E(x_real)
    c_fake = C(z_e)
    return -torch.mean(c_fake)


# -------------------------- 训练流程 --------------------------
def train_vae_gan():
    # 1. 加载数据集
    dataset = BraTSDataset(PROCESSED_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"训练集样本数：{len(dataset)}，批次大小：{BATCH_SIZE}，总批次数：{len(dataloader)}")

    # 2. 初始化模型、优化器
    E = Encoder3D(LATENT_DIM).to(DEVICE)  # 编码器
    G = Generator3D(LATENT_DIM).to(DEVICE)  # 生成器（原Decoder）
    D = Discriminator3D().to(DEVICE)  # 图像鉴别器
    C = CodeDiscriminator(LATENT_DIM).to(DEVICE)  # 代码鉴别器

    optim_E = optim.Adam(E.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_C = optim.Adam(C.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # 3. 续训配置
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "latest_ckpt.pth")
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_vae_gan.pth")
    start_epoch = 0
    best_loss = float("inf")
    losses = {"D": [], "G": [], "C": [], "E": []}

    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        E.load_state_dict(checkpoint["E_state_dict"])
        G.load_state_dict(checkpoint["G_state_dict"])
        D.load_state_dict(checkpoint["D_state_dict"])
        C.load_state_dict(checkpoint["C_state_dict"])
        optim_E.load_state_dict(checkpoint["optim_E"])
        optim_G.load_state_dict(checkpoint["optim_G"])
        optim_D.load_state_dict(checkpoint["optim_D"])
        optim_C.load_state_dict(checkpoint["optim_C"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        losses = checkpoint["losses"]
        print(f"✅ 从第 {start_epoch + 1} 轮继续训练！")
    else:
        print("🔍 未找到检查点，从头开始训练")

    # 4. 训练循环（论文顺序：E-G → D → C，G更新2次）
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_loss_C = 0.0
        epoch_loss_E = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            x_real = batch.to(DEVICE)
            z_r = torch.randn(x_real.size(0), LATENT_DIM).to(DEVICE)  # 随机潜在向量（真实z）

            # -------------------------- 1. 优化E和G（生成器侧） --------------------------
            E.train()
            G.train()
            D.eval()  # 固定D
            C.eval()  # 固定C

            # 第一次更新E和G
            loss_G = generator_loss(D, G, E, x_real, z_r, DEVICE)
            loss_E = encoder_loss(C, E, z_r, x_real, DEVICE)
            loss_EG = loss_G + loss_E

            optim_E.zero_grad()
            optim_G.zero_grad()
            loss_EG.backward()
            optim_E.step()
            optim_G.step()

            # 第二次更新G（G优化速度慢，更新2次）
            loss_G2 = generator_loss(D, G, E, x_real, z_r, DEVICE)
            optim_G.zero_grad()
            loss_G2.backward()
            optim_G.step()

            # -------------------------- 2. 优化D（鉴别器侧） --------------------------
            D.train()
            E.eval()
            G.eval()

            loss_D = discriminator_loss(D, G, E, x_real, z_r, DEVICE)
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # -------------------------- 3. 优化C（代码鉴别器侧） --------------------------
            C.train()
            E.eval()

            loss_C = code_discriminator_loss(C, E, z_r, x_real, DEVICE)
            optim_C.zero_grad()
            loss_C.backward()
            optim_C.step()

            # 累计损失
            epoch_loss_D += loss_D.item()
            epoch_loss_G += (loss_G.item() + loss_G2.item()) / 2  # 平均2次G损失
            epoch_loss_C += loss_C.item()
            epoch_loss_E += loss_E.item()

            # 更新进度条
            pbar.set_postfix({
                "D_loss": f"{loss_D.item():.4f}",
                "G_loss": f"{(loss_G.item() + loss_G2.item()) / 2:.4f}",
                "C_loss": f"{loss_C.item():.4f}",
                "E_loss": f"{loss_E.item():.4f}"
            })

        # 计算平均损失
        avg_D = epoch_loss_D / len(dataloader)
        avg_G = epoch_loss_G / len(dataloader)
        avg_C = epoch_loss_C / len(dataloader)
        avg_E = epoch_loss_E / len(dataloader)
        total_avg = (avg_D + avg_G + avg_C + avg_E) / 4

        losses["D"].append(avg_D)
        losses["G"].append(avg_G)
        losses["C"].append(avg_C)
        losses["E"].append(avg_E)

        print(
            f"Epoch {epoch + 1} | D_loss: {avg_D:.4f} | G_loss: {avg_G:.4f} | C_loss: {avg_C:.4f} | E_loss: {avg_E:.4f}"
        )

        # 保存检查点
        torch.save({
            "epoch": epoch + 1,
            "E_state_dict": E.state_dict(),
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "C_state_dict": C.state_dict(),
            "optim_E": optim_E.state_dict(),
            "optim_G": optim_G.state_dict(),
            "optim_D": optim_D.state_dict(),
            "optim_C": optim_C.state_dict(),
            "best_loss": best_loss,
            "losses": losses
        }, checkpoint_path)

        # 保存最优模型（按总损失）
        if total_avg < best_loss:
            best_loss = total_avg
            torch.save({
                "epoch": epoch + 1,
                "E_state_dict": E.state_dict(),
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "C_state_dict": C.state_dict(),
                "best_total_loss": best_loss
            }, best_model_path)
            print(f"🏆 保存最优模型（总损失：{best_loss:.4f}）")

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    for key, val in losses.items():
        plt.plot(val, label=f"{key} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE-GAN Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "vae_gan_loss_curves.png"))
    plt.show()

    return E, G, D, C


# -------------------------- 生成样本验证 --------------------------
def generate_samples_gan(model_path):
    """加载VAE-GAN模型，生成高清样本"""
    E = Encoder3D(LATENT_DIM).to(DEVICE)
    G = Generator3D(LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    E.load_state_dict(checkpoint["E_state_dict"])
    G.load_state_dict(checkpoint["G_state_dict"])
    E.eval()
    G.eval()

    # 加载真实样本
    dataset = BraTSDataset(PROCESSED_DATA_DIR)
    real_sample = dataset[0].unsqueeze(0).to(DEVICE)  # (1,4,128,160,160)

    # 生成样本
    with torch.no_grad():
        z_e = E(real_sample)
        recon_sample = G(z_e)  # 重建样本
        random_z = torch.randn(1, LATENT_DIM).to(DEVICE)
        gen_sample = G(random_z)  # 生成样本

    # 可视化
    slice_idx = 64
    real_np = real_sample.cpu().numpy()[0]
    recon_np = recon_sample.cpu().numpy()[0]
    gen_np = gen_sample.cpu().numpy()[0]

    modalities = ["T1", "T1CE", "T2", "FLAIR"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, mod in enumerate(modalities):
        axes[0, i].imshow(real_np[i, slice_idx], cmap="gray")
        axes[0, i].set_title(f"Real - {mod}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_np[i, slice_idx], cmap="gray")
        axes[1, i].set_title(f"Reconstructed (VAE-GAN) - {mod}")
        axes[1, i].axis("off")

        axes[2, i].imshow(gen_np[i, slice_idx], cmap="gray")
        axes[2, i].set_title(f"Generated (VAE-GAN) - {mod}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "vae_gan_generated_samples.png"))
    plt.show()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 训练VAE-GAN模型
    E, G, D, C = train_vae_gan()
    # 生成样本验证
    generate_samples_gan(os.path.join(MODEL_SAVE_DIR, "best_vae_gan.pth"))