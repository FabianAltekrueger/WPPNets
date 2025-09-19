# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich.
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# SIAM Journal on Imaging Sciences, vol. 16(3), pp. 1033-1067.
#
# Please cite the paper, if you use the code.
# The file is an adapted version from 
# 
# J. Hertrich, A. Houdard and C. Redenbach. 
# Wasserstein Patch Prior for Image Superresolution. 
# ArXiv Preprint#2109.12880
# (https://github.com/johertrich/Wasserstein_Patch_Prior)
# 
# and 
# 
# A. Houdard, A. Leclaire, N. Papadakis and J. Rabin. 
# Wasserstein Generative Models for Patch-based Texture Synthesis. 
# ArXiv Preprin#2007.03408
# (https://github.com/ahoudard/wgenpatex)
#
# In this script, the core functions for WPPNets and WPPFlows are implemented.

import torch
from torch import nn
import numpy as np
import math
import skimage.io as io
from torch.utils.data import Dataset
import h5py

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def psnr(img1,img2,cut=0):
    '''
    input: two images as tensor, cut boundary cuts off the boundary of given size
    returns the psnr value
    '''
    if cut > 0:
        img1 = img1[:,:,cut:-cut,cut:-cut]
        img2 = img2[:,:,cut:-cut,cut:-cut]
    mse = torch.mean((img1-img2)**2)
    return 10 * torch.log10(1/mse)

def imread(img_name):
    '''
    loads an image as torch.tensor on the selected device
    '''
    np_img = io.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    if torch.max(tens_img) > 1:
        tens_img/=255                                                   
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)                        
    if tens_img.shape[2] > 3:                                       
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)  
    return tens_img.unsqueeze(0)    

class gaussian_downsample(nn.Module):
    '''
    Downsampling module with Gaussian filtering
    ''' 
    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=False)      
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t() 
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

class SR_operator(nn.Module):
    def __init__(self,scale_factor,gaussian_std,kernel_size=16):
        super().__init__()
        self.scale_factor = scale_factor
        self.gaussian_std = gaussian_std
        self.kernel_size = kernel_size
        self.down = gaussian_downsample(self.kernel_size,self.gaussian_std,
                            int(1/self.scale_factor),pad=True).to(DEVICE)
    
    def forward(self,img):
        return self.down(img)

class semidual(nn.Module):
    '''
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    '''    
    def __init__(self, inputy, device=DEVICE, usekeops=False):
        super().__init__()
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=device))
        self.yt = inputy.transpose(1,0) 
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
        if self.usekeops:
            from pykeops.torch import LazyTensor
    
    def reset_psi(self):
        self.psi = nn.Parameter(torch.zeros(self.yt.shape[1], device=DEVICE))
        
    def forward(self, inputx):
        if self.usekeops:
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            psi_conjugate_mean = torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin]) 
        else:
            cxy = torch.sum(inputx**2,1,keepdim=True) + self.y2 - 2*torch.matmul(inputx,self.yt)
            psi_conjugate_mean = torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0])             
        psi_mean = torch.mean(self.psi)
        loss = psi_conjugate_mean + psi_mean
        return loss

class patch_extractor(nn.Module):
    '''
    Module for creating custom patch extractor
    '''
    def __init__(self, patch_size ,center=False):
        super().__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.center = center
        self.patch_size = patch_size

    def forward(self, inp, batch_size=0):
        patches = self.im2pat(inp).transpose(2,1).reshape(-1,
                                                self.patch_size**2)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        if self.center:
            patches = patches - torch.mean(patches,-1).unsqueeze(-1)
        return patches

class SiC(Dataset):
    def __init__(self,datafile,split='train'):
        super().__init__()
        if split not in ['train','val']:
            raise  ValueError(f"Invalid split: '{split}'. Use 'train' or 'val'")
        self.datafile = datafile
        self.split = split
        self.h5f = h5py.File(self.datafile,'r')
        self.keys = list(self.h5f.keys())
        
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self,key):
        if self.split == 'train':
            obs = torch.tensor(np.array(self.h5f[self.keys[key]])).squeeze(0)
            return obs
        else:
            gt = torch.tensor(np.array(self.h5f[self.keys[key]]['gt'])).squeeze(0)
            obs = torch.tensor(np.array(self.h5f[self.keys[key]]['obs'])).squeeze(0)
            return gt, obs
