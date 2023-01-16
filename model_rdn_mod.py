import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
'''
Residual Dense Network for Image Super-Resolution

Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, Yun Fu

arXiv:1802.08797 [cs.CV]

https://arxiv.org/abs/1802.08797
'''
from global_context_block import ContextBlock

class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)

class DenoiseBasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenoiseBasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)

class DenoiseBasicBlockGC(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenoiseBasicBlockGC,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
        #self.gc = ContextBlock(inplanes=output_dim, ratio=1)
    def forward(self,x):
        out = self.conv(x)
        #out = self.gc(out)
        out = self.relu(out)
        return torch.cat((x,out),1)


class DenoiseRDBGC(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBGC,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.gc = ContextBlock(inplanes=input_dim, ratio=1)
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlockGC(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.gc(out)
        return out+x

class DenoiseRDN_CustomGC(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomGC,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBGC(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )


    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf

from coordatt import CoordAtt
class DenoiseBasicBlockCoordAtt(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenoiseBasicBlockCoordAtt,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
        #self.coordatt = CoordAtt(inp=output_dim, oup=output_dim, reduction=2)
    def forward(self,x):
        out = self.conv(x)
        #out = self.coordatt(out)
        out = self.relu(out)
        return torch.cat((x,out),1)


class DenoiseRDBCoordAtt(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBCoordAtt,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.coordatt = CoordAtt(inp=input_dim, oup=input_dim, reduction=2)

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlockCoordAtt(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.coordatt(out)
        return out+x

class DenoiseRDN_CustomCoordAtt(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomCoordAtt,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBCoordAtt(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )


    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf

from selayer import SELayer

class DenoiseRDBSE(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBSE,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.selayer = SELayer(input_dim)

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.selayer(out)
        return out+x

class DenoiseRDN_CustomSE(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomSE,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBSE(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )


    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf

from cbam import CBAM

class DenoiseRDBCBAM(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBCBAM,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.cbamlayer = CBAM(gate_channels=input_dim, reduction_ratio=16)

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.cbamlayer(out)
        return out+x

class DenoiseRDN_CustomCBAM(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomCBAM,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBCBAM(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )

    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf

from eca_module import eca_layer

class DenoiseRDBECA(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBECA,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.ecalayer = eca_layer(channel=input_dim)

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.ecalayer(out)
        return out+x

import scipy.misc

def save_fm_to_png(featuremap, png_name):
    #print(featuremap.shape)
    feat_numpy = featuremap.data.cpu().numpy()
    feat_numpy = feat_numpy.squeeze(0)

    feat_mean = np.mean(feat_numpy, axis=0)
    feat_mean_rot = np.rot90(feat_mean,k=1, axes=(1,0))
    im = Image.fromarray(feat_mean_rot*255).convert('RGB')
    imsave_path = './data/feat_maps/' + png_name + '.png'
    im.save(imsave_path)
    imsave_path = './data/feat_maps/' + png_name + 'numpy.png'

    plt.imsave(imsave_path, feat_mean_rot,cmap="gray")
    #plt.imshow(feat_mean_rot)
    #plt.show()    
    imsave_path = './data/feat_maps/' + png_name + 'misc.png'

    #scipy.misc.imsave(imsave_path, feat_mean_rot)


class DenoiseRDN_CustomECA(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomECA,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBECA(nb_layers = conv_number,input_dim=channel,growth_rate=growth_rate)
            )

    def forward(self,x):
        #save_fm_to_png(x, 'pre_EAB')
        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)
            #save_fm_to_png(x, 'EAB' + str(i))


        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        #f_gf = self.GFF2(f_1x1)
        #save_fm_to_png(f_1x1, 'after_EAB')
        return f_1x1

class DenoiseRDBWithoutECA(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBWithoutECA,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.ecalayer = eca_layer(channel=input_dim)

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        #out = self.ecalayer(out)
        return out+x

class DenoiseRDN_CustomWithoutECA(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomWithoutECA,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBWithoutECA(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )

    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        #f_gf = self.GFF2(f_1x1)
        return f_1x1

from ecbam import ECBAMBlock,ChannelAttention

class DenoiseRDBECBAM(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDBECBAM,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
        self.ecbamlayer = ChannelAttention()

    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        out = self.ecbamlayer(out)
        return out+x

class DenoiseRDN_CustomECBAM(nn.Module):
    def __init__(self,channel,growth_rate, conv_number, rdb_count):
        super(DenoiseRDN_CustomECBAM,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.rdb_count = rdb_count

        self.GFF1 = nn.Conv2d(in_channels = channel*self.rdb_count,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )

        self.rdbModuleList = nn.ModuleList()
        for i in range(self.rdb_count):
            self.rdbModuleList.append(
                DenoiseRDBECBAM(nb_layers = conv_number,input_dim=60,growth_rate=growth_rate)
            )

    def forward(self,x):

        RDBs_out = []
        for i in range(self.rdb_count):
            x = self.rdbModuleList[i](x)
            RDBs_out.append(x)

        f_D = torch.cat(RDBs_out,1)

        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf

def rdn_test():
    input_t = torch.randn(5, 60, 20, 20)
    rdn = DenoiseRDN_CustomGC(channel = 60, growth_rate=20, conv_number = 4,rdb_count=3)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in rdn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in rdn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    output = rdn(input_t)
    print(output.shape)

def rdb_test():
    input_t = torch.randn(5, 60, 20, 20)
    rdn = DenoiseRDBGC(nb_layers = 4,input_dim=60,growth_rate=20)
    output = rdn(input_t)
    print(output.shape)

if __name__ == "__main__":
    #rdb_test()
    rdn_test()
