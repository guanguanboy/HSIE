import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class HsiTrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiTrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['noisy'].astype(np.float32)
        label = mat['label'].astype(np.float32)

        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        # noisy_exp = np.expand_dims(noisy, axis=0)
        # label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(noisy), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)

class HsiLowlightTestDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiLowlightTestDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['lowlight'].astype(np.float32)
        label = mat['label'].astype(np.float32)

        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        # noisy_exp = np.expand_dims(noisy, axis=0)
        # label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(noisy), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)

def run_dataset_test():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    train_set = HsiTrainDataset('./data/test/')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

#run_dataset_test()


class HsiCubicTrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiCubicTrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['patch'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        cubic = mat['cubic'].astype(np.float32)
        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(noisy_exp), torch.from_numpy(cubic), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)


class HsiCubicTestDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiCubicTestDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.image_filenames.sort(key = lambda x: int(x[18:-4])) #升序排列文件名
        #print(self.image_filenames)

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['noisy'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        noisy_cubic = mat['cubic'].astype(np.float32)
        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)
        #noisy_cubic_exp = np.expand_dims(noisy_cubic, axis=0)

        return torch.from_numpy(noisy_exp), torch.from_numpy(noisy_cubic), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)

class HsiCubicLowlightTestDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiCubicLowlightTestDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        dataset_dir_len = len(dataset_dir)
        self.image_filenames.sort(key = lambda x: int(x[dataset_dir_len:-4])) #升序排列文件名
        #print(self.image_filenames)

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['noisy'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        noisy_cubic = mat['cubic'].astype(np.float32)
        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)
        #noisy_cubic_exp = np.expand_dims(noisy_cubic, axis=0)

        return torch.from_numpy(noisy_exp), torch.from_numpy(noisy_cubic), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)

def run_cubic_test_dataset():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    test_set = HsiCubicTestDataset('./data/test_cubic/')
    train_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

#run_cubic_test_dataset()

def run_cubic_test_dataset():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    test_set = HsiCubicLowlightTestDataset('./data/test_lowlight/cubic/')
    train_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

def data_aug(img, mode=0):#图像旋转0，90，180，270，逆时针
    # data augmentation
    if mode == 0: #不旋转
        return img
    elif mode == 1: #逆时针旋转90度
        return np.rot90(img,k=1,axes=(1,2))
    elif mode == 2:
        return np.rot90(img,k=2,axes=(1,2)) #逆时针旋转180度
    elif mode == 3:
        return np.rot90(img,k=3,axes=(1,2)) #逆时针旋转270度

class HsiCubicTrainDatasetAugment(Dataset):
    def __init__(self, dataset_dir, mode):
        super(HsiCubicTrainDatasetAugment, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.aug_mode = mode

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['patch'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        cubic = mat['cubic'].astype(np.float32)

        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)

        noisy_exp_aug = data_aug(noisy_exp, self.aug_mode).copy()
        label_exp_aug = data_aug(label_exp, self.aug_mode).copy()
        cubic_aug = data_aug(cubic, self.aug_mode).copy()

        return torch.from_numpy(noisy_exp_aug), torch.from_numpy(cubic_aug), torch.from_numpy(label_exp_aug)

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    run_cubic_test_dataset()