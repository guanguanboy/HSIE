from numpy import result_type
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_relu(inchannels, outchannels):
    layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
    return layer
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class Lap_Pyramid_Conv_Multi(nn.Module):
    def __init__(self, num_high=3, device=torch.device('cpu'), chal_num=1):
        super(Lap_Pyramid_Conv_Multi, self).__init__()

        self.device_para = device
        self.input_channel_num = chal_num
        self.num_high = num_high
        self.kernel = self.gauss_kernel()


    def gauss_kernel(self, channels=1):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.input_channel_num, 1, 1, 1)
        kernel = kernel.to(self.device_para)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

from model_rdn_mod import DenoiseRDN_CustomECA,DenoiseRDN_CustomWithoutECA
from lptn_model.trans_highfreq import Trans_high_singleImage,Trans_high_Fuse

class HSIRDNECA_LPTN(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_singleImage(num_residual_blocks=3, num_high=1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)

        x_spatial_feature_3 = self.spatial_feature_3(pyr_spatial[-1])
        x_spatial_feature_5 = self.spatial_feature_5(pyr_spatial[-1])
        x_spatial_feature_7 = self.spatial_feature_7(pyr_spatial[-1])

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        output = self.conv10(feature_rdn)

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_freq = self.trans_high(lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, output]

        residual = self.lap_pyramid.pyramid_recons(laplace_list)

        return residual


def test():
    net = HSIRDNECA_LPTN(24)
    #print(net)

    data = torch.randn(1, 24, 200, 200)
    data1 = torch.randn(1, 1, 200, 200)

    output = net(data1, data)
    print('output.shape=',output.shape)


class HSIRDNECA_LPTN_FUSE(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)

        return restored

class HSIRDNECA_LPTN_FUSE_CONV(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_ResBlock2(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_ResBlock2, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 3,rdb_count=1)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=1, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_OutDoor_Try(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_OutDoor_Try, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=2, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored)

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_Without_High(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Without_High, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)

        return low_residual + x_spatial

class HSIRDNECA_LPTN_FUSE_CONV_Without_High_MSFE(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Without_High_MSFE, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(40, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        #x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        #x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        #x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        #x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_all = torch.cat((x_spatial_feature_3, x_spectral_feature_3), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)

        return low_residual + x_spatial

class HSIRDNECA_LPTN_FUSE_CONV_Without_High_MSFE_ECA(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Without_High_MSFE_ECA, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(40, self.embed_dim))
        self.rdn = DenoiseRDN_CustomWithoutECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        #x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        #x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        #x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        #x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_all = torch.cat((x_spatial_feature_3, x_spectral_feature_3), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)

        return low_residual + x_spatial

class HSIRDNECA_LPTN_FUSE_CONV_Without_High_ECA(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Without_High_ECA, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomWithoutECA(channel = self.embed_dim, growth_rate=20, conv_number = 4,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=24)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)

        return low_residual + x_spatial

def test_lptn_fuse():
    net = HSIRDNECA_LPTN_FUSE_CONV(24).cuda()
    #print(net)

    data = torch.randn(1, 24, 200, 200).cuda()
    data1 = torch.randn(1, 1, 200, 200).cuda()

    output = net(data1, data)
    print('output.shape=',output.shape)

class HSIRDNECA_LPTN_FUSE_CONV_Ablation1(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Ablation1, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 6,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_Ablation2(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Ablation2, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 5,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_Ablation3(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Ablation3, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 3,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_Ablation4(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Ablation4, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 2,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

class HSIRDNECA_LPTN_FUSE_CONV_Ablation5(nn.Module):
    def __init__(self, k=24, num_high=1):
        super(HSIRDNECA_LPTN_FUSE_CONV_Ablation5, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu
        self.embed_dim = 60
        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, self.embed_dim))
        self.rdn = DenoiseRDN_CustomECA(channel = self.embed_dim, growth_rate=20, conv_number = 1,rdb_count=4)

        self.conv10 = nn.Conv2d(self.embed_dim, 1, kernel_size=3, stride=1, padding=1)

        self.lap_pyramid = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=1)
        self.lap_pyramid_cubic = Lap_Pyramid_Conv_Multi(num_high=1, device=torch.device('cuda'), chal_num=k)

        self.trans_high = Trans_high_Fuse(num_residual_blocks=3, num_high=1)
        self.conv_last = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        
        pyr_spatial = self.lap_pyramid.pyramid_decom(img=x_spatial)
        #print("Pyr_A:")
        #for i in range(len(pyr_spatial)):
        #    print(pyr_spatial[i].shape)

        pyr_spectral = self.lap_pyramid_cubic.pyramid_decom(x_spectral)
        #print("pyr_spectral:")
        #for i in range(len(pyr_spectral)):
        #    print(pyr_spectral[i].shape)
        low_spatial = pyr_spatial[-1]
        x_spatial_feature_3 = self.spatial_feature_3(low_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(low_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(low_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(pyr_spectral[-1])
        x_spectral_feature_5 = self.spectral_feature_5(pyr_spectral[-1])
        x_spectral_feature_7 = self.spectral_feature_7(pyr_spectral[-1])

        feature_all = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1)
        feature_all = self.relu(feature_all)

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        feature_rdn = self.rdn(f0)

        low_residual = self.conv10(feature_rdn)
        low_recon = low_residual + low_spatial

        real_low_up = nn.functional.interpolate(low_spatial, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(low_recon, size=(pyr_spatial[-2].shape[2], pyr_spatial[-2].shape[3]))

        lap_sum = torch.mean(pyr_spectral[0], dim=1,keepdim=True)
        #print('lap_sum.shape = ', lap_sum.shape)
        #以lap_sum作为输入，经过trans_high网络的处理
        high_with_low = torch.cat([lap_sum, real_low_up, fake_B_up], 1)

        high_freq = self.trans_high(high_with_low, lap_sum)
        #print('high_freq.shape = ', high_freq.shape)

        laplace_list = [high_freq, low_recon]

        restored = self.lap_pyramid.pyramid_recons(laplace_list)
        refined = self.conv_last(restored) + restored

        return refined

if __name__ == "__main__":
    #test()
    test_lptn_fuse()