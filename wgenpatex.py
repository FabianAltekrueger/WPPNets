# This code belongs to the paper
#
# F. Altekrueger, J. Hertrich.
# WPPNets: Unsupervised CNN Training with Wasserstein Patch Priors for Image Superresolution.
# ArXiv preprint, to appear.
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
# In this script, the core functions for WPPNets are implemented.



import torch
from torch import nn
import numpy as np
import math
import skimage.io as io

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imread(img_name):
    """
    loads an image as torch.tensor on the selected device
    """ 
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
    """
    Downsampling module with Gaussian filtering
    """ 
    def __init__(self, kernel_size, sigma, stride, pad=False, dim=2):
        super(gaussian_downsample, self).__init__()
        self.dim=dim
        if dim==2:
            self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=False)
        elif dim==3:
            self.gauss = nn.Conv3d(1, 1, kernel_size, stride=stride, groups=1, bias=False)        
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x, dim=2):
        if self.pad and dim==2:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        elif self.pad and dim==3:
            x = torch.cat((x, x[:,:,:self.padsize,:,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize,:]), 3)
            x = torch.cat((x, x[:,:,:,:,:self.padsize]), 4)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        if self.dim==2:
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()	
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        elif self.dim==3:
            x_grid = torch.arange(kernel_size).view(kernel_size,1,1).repeat(1,kernel_size,kernel_size)
            y_grid = x_grid.permute(1,0,2)
            z_grid = x_grid.permute(2,1,0)
            xy_grid = torch.stack([x_grid,y_grid,z_grid],dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        if self.dim==2:
            return gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        elif self.dim==3:
            return gaussian_kernel.view(1, 1, kernel_size, kernel_size,kernel_size)

class semidual(nn.Module):
    """
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    input sind vektoren (also bilder in vektorform)
    """    
    def __init__(self, inputy, device=DEVICE, usekeops=False):
        super(semidual, self).__init__()
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=device))
        self.yt = inputy.transpose(1,0)	
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
    def forward(self, inputx,patch_weights):
        if self.usekeops:
            from pykeops.torch import LazyTensor
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            loss = torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin]) + torch.sum(patch_weights*self.psi)/torch.sum(patch_weights)
        else:
            cxy = torch.sum(inputx**2,1,keepdim=True) + self.y2 - 2*torch.matmul(inputx,self.yt)
            loss = torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0]) + torch.sum(patch_weights*self.psi)/torch.sum(patch_weights)
        return loss

class patch_extractor(nn.Module):
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False,center=False,dim=2):
        super(patch_extractor, self).__init__()
        self.dim=dim
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1
        self.center=center
        self.patch_size=patch_size

    def forward(self, input, batch_size=0):
        if self.pad and self.dim==2:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        elif self.pad and self.dim==3:
            input = torch.cat((input, input[:,:,:self.padsize,:,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize,:]), 3)
            input = torch.cat((input, input[:,:,:,:,:self.padsize]), 4)
        if self.dim==2:
            patches = self.im2pat(input).squeeze(0).transpose(1,0)
        elif self.dim==3:
            patches = self.im2pat(input[0]).squeeze(0).transpose(1,0).reshape(-1,input.shape[2],self.patch_size,self.patch_size)
            patches = patches.unfold(1,self.patch_size,1).permute(0,1,4,2,3)
            patches = patches.reshape(-1,self.patch_size**3)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        if self.center:
            patches = patches - torch.mean(patches,-1).unsqueeze(-1)
        return patches
