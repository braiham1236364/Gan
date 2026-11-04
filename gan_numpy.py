import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Dispositivo en uso: {device}")

CATEGORIA = "face"        # Categoría oficial de Quick, Draw! para caritas felices
NUM_MUESTRAS = 25000      # Dibujos reales usados (filtrando los más limpios)
latent_dim = 100          # Tamaño del vector de ruido z
batch_size = 64
lr = 0.0002               # Learning rate
num_epochs = 80           # Épocas de entrenamiento real

torch.manual_seed(42)
np.random.seed(42)

class QuickDrawDataset(Dataset):
    def __init__(self, categoria, n):
        archivo = f"{categoria}.npy"
        if not os.path.exists(archivo):
            url = (f"https://storage.googleapis.com/quickdraw_dataset/full/"
                   f"numpy_bitmap/{categoria}.npy")
            print(f"[*] Descargando '{categoria}' de Quick, Draw!...")
            urllib.request.urlretrieve(url, archivo)
        datos = np.load(archivo)[:60000].astype(np.float32) / 255.0   # (60000, 784) en [0, 1]
        
        # FILTRO DE CALIDAD: Descartamos el 15% más vacío y el 30% más cargado
        tinta = datos.mean(axis=1)
        bajo, alto = np.percentile(tinta, [15, 70])
        datos = datos[(tinta >= bajo) & (tinta <= alto)][:n]
        datos = datos * 2.0 - 1.0                                   # rango [-1, 1] para Tanh
        self.data = torch.tensor(datos).view(-1, 1, 28, 28)         # (n, canal, alto, ancho)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = QuickDrawDataset(CATEGORIA, NUM_MUESTRAS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"[+] Dataset cargado: {len(dataset)} imágenes | categoría: '{CATEGORIA}'")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # afina los trazos (28x28 -> 28x28)
            nn.Tanh(),                                            # píxeles en [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 7, 7)    # reordenar a forma de imagen
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        )
        self.out = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),   # una sola neurona de salida
            nn.Sigmoid(),                # 0 = falsa, 1 = real
        )

    def forward(self, img):
        x = self.conv(img)
        return self.out(x.view(x.size(0), -1))

def init_pesos(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(init_pesos)
discriminator.apply(init_pesos)

def describir_red(nombre, red):
    print(f"\n===== {nombre} =====")
    total = 0
    for nombre_param, p in red.named_parameters():
        total += p.numel()
        desv = p.std().item() if p.numel() > 1 else 0.0
        print(f"  {nombre_param:16s} forma={tuple(p.shape)!s:18s} "
              f"media={p.mean().item():+.4f}  desv={desv:.4f}  "
              f"min={p.min().item():+.3f}  max={p.max().item():+.3f}")
    print(f"  TOTAL de parámetros entrenables: {total:,}")

print("\n--- VALORES ANTES DE ENTRENAR ---")
describir_red("GENERADOR", generator)
describir_red("DISCRIMINADOR", discriminator)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

print(f"\n[--- Entrenando la GAN por {num_epochs} épocas ---]")

for epoch in range(num_epochs):
    sum_d, sum_g, sum_dr, sum_df, pasos = 0, 0, 0, 0, 0

    for real_imgs in dataloader:
        n = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_labels = torch.full((n, 1), 0.9, device=device)
        fake_labels = torch.zeros(n, 1, device=device)
        ones = torch.ones(n, 1, device=device)

        # ---- (A) Entrenar el DISCRIMINADOR ----
        optimizer_D.zero_grad()
        real_validity = discriminator(real_imgs)
        d_loss_real = criterion(real_validity, real_labels)

        z = torch.randn(n, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_validity = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(fake_validity, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---- (B) Entrenar el GENERADOR ----
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), ones)
        g_loss.backward()
        optimizer_G.step()

        sum_d += d_loss.item(); sum_g += g_loss.item()
        sum_dr += real_validity.mean().item(); sum_df += fake_validity.mean().item()
        pasos += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época [{epoch+1:3d}/{num_epochs}] | Loss D: {sum_d/pasos:.4f} | "
              f"Loss G: {sum_g/pasos:.4f} | D(real)={sum_dr/pasos:.2f} | D(falsa)={sum_df/pasos:.2f}")

print("\n[--- Entrenamiento concluido ---]")

print("\n--- VALORES DESPUÉS DE ENTRENAR ---")
describir_red("GENERADOR", generator)
describir_red("DISCRIMINADOR", discriminator)

generator.eval()
discriminator.eval()
with torch.no_grad():
    z = torch.randn(256, latent_dim, device=device)
    candidatas = generator(z)
    puntajes = discriminator(candidatas).flatten()
    mejores = puntajes.argsort(descending=True)[:16]
    generadas = candidatas[mejores].cpu().numpy()

fig, axs = plt.subplots(4, 4, figsize=(6, 6.6))
fig.suptitle(f"Caritas felices generadas por la GAN | categoría: '{CATEGORIA}'\n"
             "(las 16 mejores de 256, elegidas por el discriminador)", fontsize=11)
for k in range(16):
    ax = axs[k // 4, k % 4]
    ax.imshow(generadas[k, 0], cmap="gray_r", interpolation="bilinear", vmin=-1, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig("resultado_gan_carita.png", dpi=150)
plt.show()