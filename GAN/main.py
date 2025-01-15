import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import model
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "generated_images2"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

dataset = torchvision.datasets.CIFAR10(root="./data", download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=utils.batch_size, shuffle=True)

generator = model.Generator(utils.noise_dim, utils.img_channels).to(device)
discriminator = model.Discriminator(utils.img_channels).to(device)

criterion = nn.BCELoss()
optim_gen = optim.Adam(generator.parameters(), lr=utils.lr, betas=(0.5, 0.999))
optim_disc = optim.Adam(discriminator.parameters(), lr=utils.lr, betas=(0.5, 0.999))

for epoch in range(utils.epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.to(device)
        utils.batch_size = real.size(0)

        noise = torch.randn(utils.batch_size, utils.noise_dim, device=device)
        fake = generator(noise)
        disc_real = discriminator(real)
        disc_fake = discriminator(fake.detach())
        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + \
                    criterion(disc_fake, torch.zeros_like(disc_fake))

        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        output = discriminator(fake)
        loss_gen = criterion(output, torch.ones_like(output))

        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

    with torch.no_grad():
        noise = torch.randn(utils.batch_size, utils.noise_dim, device=device)
        fake_images = generator(noise)
        fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1) 
        save_image(fake_images, os.path.join(save_dir, f"epoch_{epoch+1}.png"), nrow=8)

    print(f"Epoch [{epoch+1}/{utils.epochs}] - Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")

print("Training complete! Generated images saved in", save_dir)