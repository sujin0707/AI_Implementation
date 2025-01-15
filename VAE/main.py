import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import utils
import model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vae = model.VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=utils.learning_rate)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

def train():
    vae.train()
    for epoch in range(utils.epochs):
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = model.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{utils.epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}")

        if (epoch + 1) % 5 == 0:
            generate_images(num_images=64, epoch=epoch + 1)

def generate_images(num_images=64, epoch=1):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_images, utils.latent_dim).to(device)
        generated = vae.decode(z).cpu()
        generated = (generated + 1) / 2 

        rows = num_images // 8
        fig, axes = plt.subplots(rows, 8, figsize=(16, 2 * rows))
        for i, img in enumerate(generated):
            ax = axes[i // 8, i % 8]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f'generated_images_epoch_{epoch+1}.png')
        plt.close()
        print(f"Generated images saved.")

if __name__ == "__main__":
    train()
    generate_images()