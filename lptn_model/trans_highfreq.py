
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high #Laplace的分解层次

        model = [nn.Conv2d(9, 64, 3, padding=1), #3个3通道的图像加起来就是9个通道的图像
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            print('pyr_original[-2].shape = ', pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])
            print('mask.shape = ', pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])
            #如果只有一个laplace层的话，下面这个上采样的过程就不需要了。
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            result_highfreq = torch.mul(pyr_original[-2-i], mask) + pyr_original[-2-i]
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            result_highfreq = self.trans_mask_block(result_highfreq)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low) #将生成的laplace顶层的低频的生成图也加入到高频转换的结果中

        return pyr_result


class Trans_high_singleImage(nn.Module):
    def __init__(self, num_residual_blocks, num_high=1):
        super(Trans_high_singleImage, self).__init__()

        self.num_high = num_high #Laplace的分解层次

        model = [nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        self.trans_mask_block = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 1))

    def forward(self, pyr_original_img, low_recon):

        mask = self.model(pyr_original_img)

        #print('pyr_original[-2].shape = ', pyr_original_img.shape[2], pyr_original_img.shape[3])
        #print('mask.shape = ', mask.shape[2], mask.shape[3])

        result_highfreq = torch.mul(pyr_original_img, mask) + pyr_original_img
        result_highfreq = self.trans_mask_block(result_highfreq)

        return result_highfreq

def test():
    trans_High_model = Trans_high(num_residual_blocks=3, num_high=3)

    intput_t = torch.randn(1, 3, 224, 224)
    intput_t_2 = torch.randn(1, 3, 112, 112)
    intput_t_4 = torch.randn(1, 3, 56, 56)
    intput_t_8 = torch.randn(1, 3, 28, 28)

    pyr_original = [intput_t, intput_t_2, intput_t_4, intput_t_8]
    x = torch.randn(1, 9, 56, 56)
    fake_low = intput_t_8

    output_t = trans_High_model(x, pyr_original, fake_low)
    print(type(output_t))

    print(len(output_t))
    for i in range(len(output_t)):
        print(output_t[i].shape)

def test_1layer():
    trans_High_model = Trans_high_singleImage(num_residual_blocks=3, num_high=1)

    intput_t = torch.randn(1, 1, 224, 224)

    output_t = trans_High_model(intput_t)
    print(type(output_t))


    print(output_t.shape)


class Trans_high_Fuse(nn.Module):
    def __init__(self, num_residual_blocks, num_high=1):
        super(Trans_high_Fuse, self).__init__()

        self.num_high = num_high #Laplace的分解层次

        model = [nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        self.trans_mask_block = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 1))
        self.count = 0

    def forward(self, pyr_original_img, origin_lap_mean):

        mask = self.model(pyr_original_img)

        #print('pyr_original[-2].shape = ', pyr_original_img.shape[2], pyr_original_img.shape[3])
        print('mask.shape = ', mask.shape[2], mask.shape[3])
        print('mask.shape = ', mask.shape)

        #将mask保存为png图片
        #step 1: 转换为numpy
        min_value = torch.min(mask)
        max_value = torch.max(mask)
        print(min_value, max_value)

        mask = ((mask - min_value)/(max_value - min_value))*255
        mask_numpy = mask.detach().cpu().numpy().astype(np.int8)
        #mask_numpy = (mask_numpy * 255).astype(np.int8)

        mask_numpy = np.squeeze(mask_numpy)
        #print(mask_numpy)
        print('mask numpy shape', mask_numpy.shape)
        #step 2: 将numpy保存为图片
        im = Image.fromarray(mask_numpy, mode='L')
        im.save("./data/mask/mask" + str(self.count)+'.png')
        self.count = self.count + 1

        result_highfreq = torch.mul(origin_lap_mean, mask) + origin_lap_mean
        result_highfreq = self.trans_mask_block(result_highfreq)

        return result_highfreq

def test_fuse():
    trans_High_model = Trans_high_Fuse(num_residual_blocks=3, num_high=1)

    intput_t = torch.randn(1, 3, 224, 224)
    origin_lap_mean = torch.randn(1, 1, 224, 224)
    output_t = trans_High_model(intput_t, origin_lap_mean)
    print(type(output_t))


    print(output_t.shape)

if __name__ == "__main__":
    #test()
    #test_1layer()
    test_fuse()