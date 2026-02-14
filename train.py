import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from generator import Generator
from discriminator import Discriminator

dataroot = "./data/football" 
batch_size = 64  
image_size = 32 
nc = 3 # число каналов
nz = 100  # размер латентного вектора
ngf = 64  # число карт признаков для генератора
ndf = 64  # число карт признаков для дискриминатора
num_epochs = 500  # количество эпох обучения
lr = 0.0001  # скорость обучения
beta1 = 0.5  # параметры Adam
beta2 = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Используемое устройство:", device)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),  # аугментация изображений
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = dset.ImageFolder(root=dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

os.makedirs("output", exist_ok=True)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):

        for _ in range(2):
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)

            label = torch.full((b_size,), 1.0, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

        netG.zero_grad()
        label.fill_(1.0) 
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}][Batch {i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")

    if (epoch + 1) % 1 == 0:
        print(f"Эпоха {epoch + 1}")
        noise_for_two = torch.randn(2, nz, 1, 1, device=device)
        with torch.no_grad():
            generated = netG(noise_for_two).detach().cpu()  # (2, nc, 32, 32)

        if generated.size(0) == 2:
            img1 = generated[0]  
            img2 = generated[1]  
            combined = torch.cat([img1, img2], dim=2)
            out_path = f"output/two_generated_epoch_{epoch + 1}.png"
            vutils.save_image(combined, out_path, normalize=True)
            print(f"Сохранено: {out_path}")
        else:
            print("Ошибка: не удалось сгенерировать 2 изображения.")
            
print("Обучение завершено.")
