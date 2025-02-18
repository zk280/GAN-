import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self,in_features=28*28):

        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features,128),
                                 nn.LeakyReLU(0.1), #由于生成对抗网络的损失非常容易梯度消失，因此使用LeakyReLU
                                 nn.Linear(128,1),
                                 nn.Sigmoid()
                                 )
    def forward(self,data):
        return self.disc(data)

class Generator(nn.Module):
    def __init__(self,in_features,out_features=784):
        """
        in_features:生成器的in_features，一般输入z的维度z_dim，该值可自定义
        out_features:生成器的out_features，需要与真实数据的维度一致
        """
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(in_features,256)
                                #,nn.BatchNorm1d(256)
                                ,nn.LeakyReLU(0.1)
                                ,nn.Linear(256,out_features)
                                ,nn.Tanh() #用于归一化数据
                                )
    def forward(self,z):
        gz = self.gen(z)
        return gz