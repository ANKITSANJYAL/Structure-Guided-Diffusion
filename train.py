import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from diffusers import DDPMScheduler

from src.models.unet import GeometryAwareUNet
from src.conditioning.clip_encoder import CLIPTextEncoder
from src.geometry.dino_encoder import DINOEncoder

# -----------------------
# ⚙️ Config
# -----------------------
EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
LR = 1e-4
SAVE_PATH = "checkpoints/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 🖼️ Dataset (placeholder)
# -----------------------

class KaggleFlowersDataset(torch.utils.data.Dataset):
    def __init__(self, root="/kaggle/input/flowers-recognition", img_size=224):
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.dataset = ImageFolder(root=root, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.dataset.classes[label]  # label as string for CLIP
        return img, class_name


# -----------------------
# 🚀 Initialize components
# -----------------------
clip_encoder = CLIPTextEncoder().to(DEVICE)
dino_encoder = DINOEncoder().to(DEVICE)
unet = GeometryAwareUNet().to(DEVICE)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

# -----------------------
# 📦 Load data
# -----------------------
dataset = SimpleFlowers(root="./data", split="train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -----------------------
# 🏋️ Training Loop
# -----------------------
for epoch in range(EPOCHS):
    unet.train()
    pbar = tqdm(dataloader)
    total_loss = 0

    for images, class_labels in pbar:
        images = images.to(DEVICE)
        bsz = images.size(0)

        # Step 1: Encode conditions
        clip_emb = clip_encoder.encode(class_labels).to(DEVICE)           # (B, 1, 768)
        dino_tokens = dino_encoder(images).to(DEVICE)                     # (B, T, 768)

        # Step 2: Add noise
        noise = torch.randn_like(images).to(DEVICE)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
        x_t = noise_scheduler.add_noise(images, noise, timesteps)

        # Step 3: Predict noise
        noise_pred = unet(x_t, timesteps, clip_emb, dino_tokens)

        # Step 4: Loss and backward
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    # Save checkpoint
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(SAVE_PATH, f"unet_epoch{epoch+1}.pth"))
