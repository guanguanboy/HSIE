from matplotlib.pyplot import axis, imshow

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset

from model_hsid_origin import HSID_origin
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper.helper_utils import init_params, get_summary_writer
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.modules.loss import _Loss
from hsidataset import HsiCubicTrainDataset
import numpy as np
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicTestDataset,HsiCubicLowlightTestDataset
import scipy.io as scio
from losses import EdgeLoss
from tvloss import TVLoss
#from warmup_scheduler import GradualWarmupScheduler
from dir_utils import *
from model_utils import *
import time
from utils import get_adjacent_spectral_bands
from model_rdn import HSIRDN,HSIRDNMOD,HSIRDNSE,HSIRDNECA,HSIRDNWithoutECA
import model_utils
import dir_utils
from hsi_lptn_model import HSIRDNECA_LPTN,HSIRDNECA_LPTN_FUSE

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.0004
K = 24
display_step = 20
display_band = 20
RESUME = False

#设置随机种子
seed = 200
torch.manual_seed(seed) #在CPU上设置随机种子
if DEVICE == 'cuda':
    torch.cuda.manual_seed(seed) #在当前GPU上设置随机种子
    torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def loss_fuction(x,y):
    MSEloss=sum_squared_error()
    loss1=MSEloss(x,y)

    return loss1

def loss_function_mse(x, y):
    MSELoss = nn.MSELoss()
    loss = MSELoss(x, y)
    return loss

recon_criterion = nn.L1Loss() 

def train_model_residual_lowlight_rdn():

    device = DEVICE
    print(device)
    #准备数据
    train_set = HsiCubicTrainDataset('./data/train_lowlight_patchsize64_k12/')
    #print('trainset32 training example:', len(train_set32))
    #train_set = HsiCubicTrainDataset('./data/train_lowlight/')

    #train_set_64 = HsiCubicTrainDataset('./data/train_lowlight_patchsize64/')

    #train_set_list = [train_set32, train_set_64]
    #train_set = ConcatDataset(train_set_list) #里面的样本大小必须是一致的，否则会连接失败
    print('total training example:', len(train_set))

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    #加载测试label数据
    mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label_hsi = scio.loadmat(mat_src_path)['label']

    #加载测试数据
    batch_size = 1
    #test_data_dir = './data/test_lowlight/cuk12/'
    test_data_dir = './data/test_lowlight/cuk12/'

    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))

    save_model_path = './checkpoints/hsirnd_indoor_lptn_fuse_train10_patchsize64_lr0004'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    #创建模型
    net = HSIRDNECA_LPTN_FUSE(K)
    init_params(net)
    device_ids = [0, 1]
    #net = nn.DataParallel(net).to(device)
    net = net.to(device)

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = MultiStepLR(hsid_optimizer, milestones=[200,400], gamma=0.5)

    #定义loss 函数
    #criterion = nn.MSELoss()
    best_psnr = 0
    best_ssim = 0
    best_sam = 0

    is_resume = RESUME
    #唤醒训练
    if is_resume:
        path_chk_rest    = dir_utils.get_last_path(save_model_path, 'model_latest.pth')
        model_utils.load_checkpoint(net,path_chk_rest)
        start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1
        model_utils.load_optim(hsid_optimizer, path_chk_rest)
        best_psnr = model_utils.load_best_psnr(path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    global tb_writer
    tb_writer = get_summary_writer(log_dir='logs')

    gen_epoch_loss_list = []

    cur_step = 0

    first_batch = next(iter(train_loader))

    best_epoch = 0
    best_iter = 0
    if not is_resume:
        start_epoch = 1
    num_epoch = 600

    mpsnr_list = []
    for epoch in range(start_epoch, num_epoch+1):
        epoch_start_time = time.time()
        scheduler.step()
        print('epoch = ', epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
        print(scheduler.get_lr())

        gen_epoch_loss = 0

        net.train()
        #for batch_idx, (noisy, label) in enumerate([first_batch] * 300):
        for batch_idx, (noisy, cubic, label) in enumerate(train_loader):
            #print('batch_idx=', batch_idx)
            noisy = noisy.to(device)
            label = label.to(device)
            cubic = cubic.to(device)

            hsid_optimizer.zero_grad()
            #denoised_img = net(noisy, cubic)
            #loss = loss_fuction(denoised_img, label)

            residual = net(noisy, cubic)
            alpha = 0.8
            loss = recon_criterion(residual, label)
            #loss = alpha*recon_criterion(residual, label-noisy) + (1-alpha)*loss_function_mse(residual, label-noisy)
            #loss = recon_criterion(residual, label-noisy)
            loss.backward() # calcu gradient
            hsid_optimizer.step() # update parameter

            gen_epoch_loss += loss.item()

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Batch_idx {batch_idx}: MSE loss: {loss.item()}")
                else:
                    print("Pretrained initial state")

            tb_writer.add_scalar("MSE loss", loss.item(), cur_step)

            #step ++,每一次循环，每一个batch的处理，叫做一个step
            cur_step += 1


        gen_epoch_loss_list.append(gen_epoch_loss)
        tb_writer.add_scalar("mse epoch loss", gen_epoch_loss, epoch)

        #scheduler.step()
        #print("Decaying learning rate to %g" % scheduler.get_last_lr()[0])
 
        torch.save({
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
        }, f"{save_model_path}/hsid_rdn_eca_l1_loss_600epoch_patchsize32_{epoch}.pth")

        #测试代码
        net.eval()
        psnr_list = []

        for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
            noisy_test = noisy_test.type(torch.FloatTensor)
            label_test = label_test.type(torch.FloatTensor)
            cubic_test = cubic_test.type(torch.FloatTensor)

            noisy_test = noisy_test.to(DEVICE)
            label_test = label_test.to(DEVICE)
            cubic_test = cubic_test.to(DEVICE)

            with torch.no_grad():

                residual = net(noisy_test, cubic_test)
                denoised_band = residual
                
                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,batch_idx] = denoised_band_numpy

                if batch_idx == 49:
                    residual_squeezed = torch.squeeze(residual, axis=0)
                    denoised_band_squeezed = torch.squeeze(denoised_band, axis=0) 
                    label_test_squeezed = torch.squeeze(label_test,axis=0)
                    noisy_test_squeezed = torch.squeeze(noisy_test,axis=0)
                    tb_writer.add_image(f"images/{epoch}_restored", denoised_band_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_residual", residual_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_label", label_test_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_noisy", noisy_test_squeezed, 1, dataformats='CHW')

            test_label_current_band = test_label_hsi[:,:,batch_idx]

            psnr = PSNR(denoised_band_numpy, test_label_current_band)
            psnr_list.append(psnr)
        
        mpsnr = np.mean(psnr_list)
        mpsnr_list.append(mpsnr)

        denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
        test_label_hsi_trans = test_label_hsi.transpose(2, 0, 1)
        mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
        sam = SAM(denoised_hsi_trans, test_label_hsi_trans)


        #计算pnsr和ssim
        print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(mpsnr, mssim, sam)) 
        tb_writer.add_scalars("validation metrics", {'average PSNR':mpsnr,
                        'average SSIM':mssim,
                        'avarage SAM': sam}, epoch) #通过这个我就可以看到，那个epoch的性能是最好的

        #保存best模型
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            best_epoch = epoch
            best_iter = cur_step
            best_ssim = mssim
            best_sam = sam
            torch.save({
                'epoch' : epoch,
                'gen': net.state_dict(),
                'gen_opt': hsid_optimizer.state_dict(),
            }, f"{save_model_path}/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth")

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f Best_SSIM %.4f Best_SAM %.4f]" % (epoch, cur_step, mpsnr, best_epoch, best_iter, best_psnr, best_ssim, best_sam))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, gen_epoch_loss, INIT_LEARNING_RATE))
        print("------------------------------------------------------------------")

        #保存当前模型
        torch.save({'epoch': epoch, 
                    'gen': net.state_dict(),
                    'gen_opt': hsid_optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    }, os.path.join(save_model_path,"model_latest.pth"))
    mpsnr_list_numpy = np.array(mpsnr_list)
    np.save(os.path.join(save_model_path, "mpsnr_per_epoch.npy"), mpsnr_list_numpy)    
    tb_writer.close()

if __name__ == '__main__':
    train_model_residual_lowlight_rdn()