import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import time
from torch import nn
import gan_network
from torch import save

batch_size=128
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch=100000
z_dim=100

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

train_dataset = datasets.ImageFolder(root='f:/misaka', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


g_net=gan_network.Generator(z_dim).to(device)
d_net=gan_network.Discriminator().to(device)
g_net.load_state_dict(torch.load('./gpara'))
d_net.load_state_dict(torch.load('./dpara'))
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))


bce_loss=nn.BCELoss()
fix_z=torch.randn(size=(1, z_dim, 1, 1)).to(device)

for iteration in range(epoch):
    for image, _ in train_loader:#没有标签
        # 1：数据准备
        image = image.to(device)
        real_label = torch.ones(size=(image.size(0), 1), requires_grad=False).to(device)
        fake_label = torch.zeros(size=(image.size(0), 1), requires_grad=False).to(device)
        z = torch.randn(size=(image.size(0), z_dim, 1, 1)).to(device)

        # #################################################
        # 2:训练生成器
        optimizer_G.zero_grad()
        # 2.1：生成伪造样本
        generated_image = g_net(z)
        # 2.2：计算判别器对伪造样本的输出的为真样本的概率值
        d_out_fake = d_net(generated_image)
        # 2.3：计算生成器伪造样本不被认为是真的损失
        g_loss = bce_loss(d_out_fake, real_label)
        # 2.4：更新生成器
        g_loss.backward()
        optimizer_G.step()

        # #################################################
        # 3：训练判别器
        optimizer_D.zero_grad()
        # 3.1：计算判别器对真实样本给出为真的概率
        d_out_real = d_net(image)
        # 3.2：计算判别器对真实样本的su's
        real_loss = bce_loss(d_out_real, real_label)
        # 3.3:计算判别器
        d_out_fake = d_net(generated_image.detach())
        fake_loss = bce_loss(d_out_fake, fake_label)
        d_loss = real_loss + fake_loss
        # 3.4:更新判别器参数
        d_loss.backward()
        optimizer_D.step()
        # 4:记录损失
        #record_iter.append(iteration)
        #record_g_loss.append(g_loss.item())
        #record_d_loss.append(d_loss.item())

        # #################################################
        # 5：打印损失，保存图片
        
    if iteration % 100 == 0:
        with torch.no_grad():
            g_net.eval()
            fixed_image = g_net(fix_z)
            g_net.train()
            print("[iter: {}], [G loss: {}], [D loss: {}]".format(iteration, g_loss.item(), d_loss.item()))
            image=fixed_image[0].squeeze().permute(1,2,0).cpu().numpy()
            image=image*255
            image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("f:/result/"+str(iteration)+".jpg", image)
            #save.save_image(image_tensor=fixed_image[0].squeeze(), out_name="results/"+str(iteration)+".jpg")
            torch.save(g_net.state_dict(),'./gpara')
            torch.save(d_net.state_dict(),'./dpara')