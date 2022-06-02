# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich.
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# ArXiv Preprint#2201.08157
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
import os
import skimage.metrics as sm
from natsort import os_sorted

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def psnr(tensor1,tensor2,cut_boundary = False):
    '''
    input: two images as tensor, cut boundary cuts off the boundary of given size
    returns the psnr value
    '''
    if not cut_boundary:
        img1 = np.clip(tensor1.squeeze().detach().cpu().numpy(),0,1)
        img2 = np.clip(tensor2.squeeze().detach().cpu().numpy(),0,1)
        return sm.peak_signal_noise_ratio(img1,img2)
    a = cut_boundary    
    img1 = np.clip(tensor1[:,:,a:-a,a:-a].squeeze().detach().cpu().numpy(),0,1)
    img2 = np.clip(tensor2[:,:,a:-a,a:-a].squeeze().detach().cpu().numpy(),0,1)
    return sm.peak_signal_noise_ratio(img1,img2)
    
def save_img(tensor_img, name):
	'''
	save img (tensor form) with the name
	'''
	img = np.clip(tensor_img.squeeze().detach().cpu().numpy(),0,1)
	io.imsave(str(name)+'.png', img)
	return     

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


class semidual(nn.Module):
    '''
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    input sind vektoren (also bilder in vektorform)
    '''    
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
    '''
    Module for creating custom patch extractor
    '''
    def __init__(self, patch_size, pad=False,center=False):
        super(patch_extractor, self).__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1
        self.center=center
        self.patch_size=patch_size

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1,0)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        if self.center:
            patches = patches - torch.mean(patches,-1).unsqueeze(-1)
        return patches

def Trainset(image_class, size=1000):
	'''
	create training set consisting of low resolution images
	input: size of the training set, set to 1000 by default
	''' 
	lr_img = torch.empty(0, device = DEVICE)
	picts = os.listdir('training_img/lr_' + image_class)
	for img in picts:
		tmp = imread('training_img/lr_' + image_class + '/' + img)
		lr_img = torch.cat([lr_img,tmp])
	idx = torch.randperm(lr_img.shape[0])[:size]	
	return lr_img[idx,...]

def Validationset(image_class):
	'''
	create validation set (list) consisting of labeled lr-hr images 
	'''
	val = []
	picts_hr = os_sorted(os.listdir('training_img/validation_' + image_class + '/val_hr'))
	picts_lr = os_sorted(os.listdir('training_img/validation_' + image_class + '/val_lr'))
	for i in range(len(picts_hr)):
		val_hr = imread('training_img/validation_' + image_class + '/val_hr/' + picts_hr[i])
		val_lr = imread('training_img/validation_' + image_class + '/val_lr/' + picts_lr[i])
		val.append([val_lr,val_hr])
	return val	
       
