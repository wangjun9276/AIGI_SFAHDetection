from torchvision import datasets, models, transforms
import torch
import random
import cv2 as cv
import math
from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn import functional as F
import seaborn as sns
from torch import Tensor
import skimage
from scipy import ndimage
from scipy.fftpack import dct, idct, fft2, fftshift
from io import BytesIO
import imageio.v2 as imageio
import kornia




# from imageneAE.models import builer as builder
# import argparse
# from imageneAE import utils

# # parse the args
# print('=> parse the args ...')
# parser = argparse.ArgumentParser(description='Trainer for auto encoder')
# parser.add_argument('--arch', default='vgg16', type=str, 
#                     help='backbone architechture')
# parser.add_argument('--resume', type=str, default='./zj_ana/imageneAE/checkpoints/imagenet-vgg16.pth')
# parser.add_argument('--val_list', type=str)              

# args = parser.parse_args()

# args.parallel = 0
# args.batch_size = 1
# args.workers = 0

# print('=> modeling the ImageNet AE network ...')   
# model = builder.BuildAutoEncoder(args)

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

mode = 'gaussian' # [gaussian, localvar, poisson, salt, pepper, s&p, speckle, none]
jpeg_factor = 96

def jpeg_compression_in_buffer(x_in, jpg_quality):


    buf = BytesIO()
    x_in.save(buf, format='jpeg', quality=jpg_quality)

    with BytesIO(buf.getvalue()) as stream:
        x_jpg = imageio.imread(stream)

    x_jpg = Image.fromarray(np.uint8(x_jpg))

    return x_jpg

def compute(patch):
    weight, height = patch.size
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()

def patch_img(img, patch_size=32, height=256):
    img_width, img_height = img.size
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    for i in range(num_patch):
        patch_list.append(rp(img))
    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img = patch_list[0]
    return new_img

def fft_img(img):
    img = transforms.ToPILImage()(img)
    eposilon = 1e-10
    img = img.convert("L")
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+eposilon)
    # fimg = np.log(np.abs(f))
    fimg = fimg[:,:,np.newaxis]

    return fimg

def gray_fft(img, ker=3):
    gray_img = np.array(img).mean(axis=0)
    
    gimg = skimage.util.random_noise(gray_img, mode=mode)
    gray_img_blur = ndimage.median_filter(gimg, ker)
    # gray_gimg2 = skimage.util.random_noise(gray_img_blur, mode=mode)
    # gray_gimg_blur = ndimage.median_filter(gray_gimg2, ker)
    gray_img = gimg - gray_img_blur

    # image_grey_fourier = np.fft.fftshift(np.fft.fft2(gray_img))
    image_grey_fourier = fft2(gray_img)
    image_grey_fourier = fftshift(image_grey_fourier)

    # # log_fft = np.log(abs(image_grey_fourier))
    # angle_fft = np.abs(image_grey_fourier)*np.exp2(np.angle(image_grey_fourier)*1j)
    # angle_fft = np.fft.ifft2(angle_fft)
    # angle_fft = np.log(abs(angle_fft))

    log_fft = np.log(abs(image_grey_fourier) + 0.0001)
    fft_min = np.percentile(log_fft, 1)
    fft_max = np.percentile(log_fft, 99)
    log_fft = (log_fft - fft_min)/(fft_max - fft_min + 0.0001)
    log_fft = (log_fft - 0.5)/(0.5)
    # log_fft = np.expand_dims(np.log(abs(image_grey_fourier)), 2).repeat(3, axis=2) / 15
    # print(log_fft.shape)

    # plt.imshow(angle_fft)
    # plt.show()
    log_fft = log_fft[:,:,np.newaxis]
    # print(log_fft.shape)

    return log_fft

def judge_img(img):
    img_width, img_height = img.size
    # if(img_width <224 or img_height <224):
    if(img_width <256 or img_height <256):
        # img = img.resize((224, 224))
        img = img.resize((256, 256))
    return img

def npr_img(image: Tensor):

    factor = 0.5
    # factor = 2
    image = image.unsqueeze(dim=0)
    
    NPR = kornia.filters.gaussian_blur2d(image, (3, 3), (1.5, 1.5))
    # image = kornia.filters.Sobel()(image)
    # image = sobel_feature + image
    # The input dimensions are interpreted in the form:
    # `mini-batch x channels x [optional depth] x [optional height] x width`.
    # NPR = F.interpolate(image, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
    # NPR = F.interpolate(NPR,scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)
    NPR = image - NPR
    image = NPR * 2.0 / 3.0
    image = image.squeeze()

    return image

def patch_reisze(x, patch_size=16):
    img = torch.from_numpy(np.array(x)).permute(2,0,1)
    (c,h,w) = torch.from_numpy(np.array(img)).shape
    img_patch = img.view(c, h//patch_size, w//patch_size, patch_size, patch_size).contiguous()
    re_img = []
    for imgp in img_patch.view(c, -1, patch_size, patch_size).permute(1, 0, 2, 3):
        img_p = transforms.Resize((patch_size*2, patch_size*2), transforms.InterpolationMode.BICUBIC)(imgp)
        img_p = transforms.Resize((patch_size, patch_size))(imgp)
        re_img.append(img_p)
    re_img = torch.stack(re_img).permute(1, 0, 2, 3).view(c, h//patch_size, w//patch_size, patch_size, patch_size).contiguous()
    re_img = re_img.view(c, h, w)

    return transforms.ToPILImage()(re_img)


fft_func = transforms.Lambda(
            lambda img: fft_img(img))

patch_func = transforms.Lambda(
            lambda img: patch_img(img))

judge_func = transforms.Lambda(
            lambda img: judge_img(img))

npr_func = transforms.Lambda(
            lambda img: npr_img(img))

re_func = transforms.Lambda(
            lambda img: patch_reisze(img))

# ae_func = transforms.Lambda(
#             lambda img: AE_imagenet(img))

def train_augment():
    transform_list = []
    transform_list.extend([
        judge_func,
        # re_func,
        # patch_func,
        # transforms.RandomRotation(90),
        # transforms.ColorJitter(contrast=0.05, hue=0.05, brightness=0.05),
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.RandomCrop(256),
        # ae_func,
        transforms.Normalize(mean=MEAN["imagenet"], std=STD["imagenet"]),
        npr_func,
        fft_func,
        # transforms.ToTensor(),
    ])
    return transforms.Compose(transform_list)


class real_dataset(Dataset):
    def __init__(self, dataroot, transform):
        self.root_dir = dataroot
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.data = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if '0_real' in file_path and 'csv' not in file_path:
                    self.data.append((file_path, 0))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert("RGB")
        # print(img_path, img_path.endswith('png'))
        # if img_path.endswith('png'):
        #     image = jpeg_compression_in_buffer(image, jpg_quality=jpeg_factor)
        # image = Image.open(img_path).convert("RGB").resize((256, 256))
        image = self.transform(image)

        return image


class fake_dataset(Dataset):
    def __init__(self, dataroot, transform):
        self.root_dir = dataroot
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.data = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if '1_fake' in file_path and 'csv' not in file_path:
                    self.data.append((file_path, 1))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert("RGB")
        # print(img_path, img_path.endswith('png'))
        # if img_path.endswith('png'):
        #     image = jpeg_compression_in_buffer(image, jpg_quality=jpeg_factor)
        image = image.resize((256, 256))
        image = self.transform(image)

        return image


# type = 'stylegan2'
# type = 'ADM'
# type = 'cat'
# type = 'biggan'
# type = 'Glide'
# type = 'stargan'
# type = 'progan'
# type = 'Midjourney'
# type = 'Glide'
type = 'stable_diffusion_v_1_4'
# type = 'stylegan'

name = type

# root = '/mnt/sdb/wangjun/datasets/inthewild/'
root = '/media/sda/junwang/datasets/inthewild/'
test_folders = ['test', 'test', 'test', 'test', 'AIRESGANtest_0.25', 'test', 'AILDM128to512J96_0.5', 'progan_train_DiffR']
NPR_sufixs = ['NPR', 'NPRs', 'solbel', 'sol', 'NPRRGAN', 'NPR_AEDiff', 'NPR_AELDM', 'NPR_car']
test_folder_index = 3

dataroot = os.path.join(root, test_folders[test_folder_index], type)


data_loader_fake = torch.utils.data.DataLoader(fake_dataset(dataroot,train_augment()),
                                  batch_size=200,
                                  shuffle=False,
                                  num_workers=int(0))

data_loader_real = torch.utils.data.DataLoader(real_dataset(dataroot,train_augment()),
                                  batch_size=200,
                                  shuffle=False,
                                  num_workers=int(0))


for index,img in enumerate(data_loader_fake):
    if index == 0:
        res_fake = img
    elif index <= 1000:
        res_fake = torch.cat((img, res_fake), dim=0)
    else:
        res_fake = torch.cat((img,res_fake),dim=0)

# print(res_fake.shape)
res_fake = torch.mean(res_fake,dim=0).cpu().squeeze()
tensor_image = res_fake.float()
res_fake = tensor_image.numpy()
res_fake = (255 * res_fake).astype(np.uint8)



for index,img in enumerate(data_loader_real):
    if index == 0:
        res_real = img
    elif index <= 1000:
        res_real = torch.cat((img, res_real), dim=0)
    else:
        res_real = torch.cat((img,res_real),dim=0)


res_real = torch.mean(res_real,dim=0).cpu().squeeze()
tensor_image = res_real.float()
res_real = tensor_image.numpy()
res_real = (255 * res_real).astype(np.uint8)

difference = res_real - res_fake

plt.figure(figsize=(20,10),dpi=300)
plt.subplot(1,3,1)
plt.imshow(res_fake)
# plt.imshow(res_fake, 'gray')
plt.title(f'fake_img_{name}')
# plt.savefig(f'fake_img_{name}.pdf')


plt.subplot(1,3,2)
plt.imshow(res_real)
# plt.imshow(res_real, 'gray')
plt.title(f'real_img_{name}')
# plt.savefig(f'real_img_{name}.pdf')


abs_diff = np.abs(difference)
# plt.subplot(1,4,3)
# plt.imshow(abs_diff, 'gray')
# plt.title(f'real_img_{name}')

# sns.heatmap(res, cmap='hot')
plt.subplot(1,3,3)
sns.heatmap(abs_diff, cmap='coolwarm')
plt.show()
plt.savefig(f'./figures/{name}_{NPR_sufixs[test_folder_index]}.pdf', dpi=300)
# plt.close()