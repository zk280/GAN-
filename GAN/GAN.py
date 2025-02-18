import GAN_utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from skimage.io import imread


class GAN():
    def __init__(self, num_epochs, batch_size, z_dim,train_path,jpg_size=28 * 28):
        self.train_path = train_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.jpg_size = jpg_size
        self.generator = GAN_utils.Generator(self.z_dim, self.jpg_size)
        self.discriminator = GAN_utils.Discriminator(self.jpg_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        optimizer_generator = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(self.num_epochs):
            for x in self.load_batch(self.train_path):
                x = torch.from_numpy(x).float().to(self.device)  # 转换为 PyTorch 张量
                x = x.view(-1, self.jpg_size)  # 展平图片

                # 训练判别器
                real_labels = torch.ones(x.size(0), 1).to(self.device)
                dis_real = self.discriminator(x)
                loss_real = criterion(dis_real, real_labels)
                D_x = dis_real.mean().item()

                # 生成伪图片
                noise = torch.randn(x.size(0), self.z_dim).to(self.device)
                fake = self.generator(noise).detach()  # 阻断反向传播
                fake_labels = torch.zeros(x.size(0), 1).to(self.device)
                dis_fake = self.discriminator(fake)
                loss_fake = criterion(dis_fake, fake_labels)

                errorD = loss_real + loss_fake
                errorD.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

                # 训练生成器
                noise = torch.randn(x.size(0), self.z_dim).to(self.device)
                fake = self.generator(noise)
                dis_fake = self.discriminator(fake)
                real_labels = torch.ones(x.size(0), 1).to(self.device)
                Gloss = criterion(dis_fake, real_labels)
                Gloss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()

                # 打印日志
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], ErrorD: {errorD.item():.4f}, Gloss: {Gloss.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {dis_fake.mean().item():.4f}")

            # 每轮保存模型
            torch.save(self.generator.state_dict(), f"generator_epoch_{epoch + 1}.pth")
            torch.save(self.discriminator.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")

    def load_batch(self,train_path):
        path = glob(train_path + '/*.jpg')
        batch_num = int(np.floor(len(path) / self.batch_size))

        # 获取图像的尺寸
        img_tem = imread(path[0])
        h, w = img_tem.shape
        # 初始化批次图像数组
        imsize = (self.batch_size, 1, h, w)
        imgs_A = np.zeros(imsize)
        # 处理每个批次
        for i in range(batch_num):
            batch = path[i * self.batch_size:(i + 1) * self.batch_size]
            for j, img_path in enumerate(batch):
                img = imread(img_path)
                img = img - np.min(img)
                img = img / np.max(img)
                img = img.astype('float32')
                img = img.reshape(1, h, w)
                imgs_A[j, :, :, :] = img

            # 生成并返回处理后的测试图像
            yield imgs_A





# 示例使用
g = GAN(num_epochs=10, batch_size=512, z_dim=64, train_path='C:/test for github/GAN/mnist_jpg')
g.execute()