# This code belongs to the paper
#
# F. Altekrüger and J. Hertrich. 
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# ArXiv Preprint#2201.08157
#
# Please cite the paper, if you use the code.
#
# The script estimates the forward operator based on a registered pair of
# high- and low-resolution image.

import torch
import torch.nn as nn
import skimage.io,skimage.transform
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(kernel_size, sigma,shift=torch.zeros(2,dtype=torch.float,device=DEVICE)):
    x_grid = torch.arange(kernel_size).view(kernel_size,1).repeat(1,kernel_size).to(DEVICE)
    y_grid = x_grid.permute(1,0)
    xy_grid = torch.stack([x_grid,y_grid],dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.*torch.tensor([1.,1.],dtype=torch.float,device=DEVICE)

    gaussian_kernel = (1./(torch.prod(2.*math.pi*variance)**.5))*torch.exp(-torch.sum((xy_grid - mean - shift)**2./(2*variance), dim=-1))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

class SR_operator(nn.Module):
    def __init__(self,scale,kernel_size=15):
        super(SR_operator,self).__init__()
        self.kernel=nn.Parameter(init_weights(kernel_size,2),requires_grad=False)
        self.bias=nn.Parameter(torch.tensor([0.],device=DEVICE,dtype=torch.float),requires_grad=False)
        self.scale=scale
        self.kernel_size=kernel_size

    def forward(self,x):
        kernel=torch.zeros_like(x)
        diffs_kernel=np.array([kernel.shape[2]-self.kernel_size,kernel.shape[3]-self.kernel_size])
        diffs_kernel_right=diffs_kernel//2
        diffs_kernel_left=diffs_kernel-diffs_kernel_right
        kernel[:,:,diffs_kernel_left[0]:-diffs_kernel_right[0],diffs_kernel_left[1]:-diffs_kernel_right[1]]=self.kernel.data
        kernel=torch.fft.ifftshift(kernel)
        x=torch.fft.fftn(x)/torch.prod(torch.tensor(x.shape,dtype=torch.float,device=DEVICE))
        kernel_four=torch.fft.fftn(kernel)
        x=kernel_four*x
        x[:,0,0,0]+=self.bias.data
        x=torch.fft.fftshift(x)
        hr_shape=x.shape[2:]
        if type(self.scale)==list:
            lr_shape=[int(np.round(s*self.scale[t])) for t,s in enumerate(hr_shape)]
        else:
            lr_shape=[int(np.round(s*self.scale)) for s in hr_shape]
        diffs=np.array([hr_shape[0]-lr_shape[0],hr_shape[1]-lr_shape[1]])
        diffs_left=diffs//2
        diffs_right=diffs-diffs_left
        diffs_right[diffs_right==0]=-np.max(list(x.shape))
        x=x[:,:,diffs_left[0]:-diffs_right[0],diffs_left[1]:-diffs_right[1]]
        x=x*torch.prod(torch.tensor(x.shape,dtype=torch.float,device=DEVICE))
        x=torch.real(torch.fft.ifftn(torch.fft.ifftshift(x)))
        return x

if __name__=='__main__':
    
    print(DEVICE)
    kernel_size=15
    hr_learn=skimage.io.imread('training_img/FS_registered_operator/FS_HR_estimate_operator.png').astype(np.float64)/255
    lr_learn=skimage.io.imread('training_img/FS_registered_operator/FS_LR_estimate_operator.png').astype(np.float64)/255
    lr_learn=skimage.transform.rescale(lr_learn,.5)
    diffs=np.array([hr_learn.shape[0]-lr_learn.shape[0],hr_learn.shape[1]-lr_learn.shape[1]])
    diffs_right=diffs//2
    diffs_left=diffs-diffs_right

    hr_four=np.fft.fftshift(np.fft.fftn(hr_learn))/np.prod(hr_learn.shape)
    hr_four_lr=hr_four[diffs_left[0]:-diffs_right[0],diffs_left[1]:-diffs_right[1]]
    lr_four=np.fft.fftshift(np.fft.fftn(lr_learn))/np.prod(lr_learn.shape)
    kernel_four_middle_phase=(lr_four/hr_four_lr)/np.abs(lr_four/hr_four_lr)
    kernel_four_middle_abs=np.abs(lr_four)/(np.abs(hr_four_lr)+1e-5)
    kernel_four_middle=kernel_four_middle_phase*kernel_four_middle_abs
    kernel_four_img=np.abs(kernel_four_middle)
    kernel_four=np.zeros_like(hr_four)
    kernel_four[diffs_left[0]:-diffs_right[0],diffs_left[1]:-diffs_right[1]]=kernel_four_middle
    kernel_=np.fft.ifftn(np.fft.ifftshift(kernel_four))
    kernel=np.real(kernel_)
    kernel=np.fft.fftshift(kernel)

    diffs_kernel=np.array([kernel.shape[0]-kernel_size,kernel.shape[1]-kernel_size])
    diffs_kernel_right=diffs_kernel//2
    diffs_kernel_left=diffs_kernel-diffs_kernel_right
    kernel_small=kernel[diffs_kernel_left[0]:-diffs_kernel_right[0],diffs_kernel_left[1]:-diffs_kernel_right[1]]
    bias=(kernel.sum()-kernel_small.sum())*(1-kernel_size**3/np.prod(hr_learn.shape))*np.mean(hr_learn)
    bias_per_point=bias/np.prod(hr_learn.shape)/np.mean(hr_learn)
    
    kernel_torch=torch.tensor(kernel_small[np.newaxis,np.newaxis,:,:],dtype=torch.float,device=DEVICE)
    my_operator=SR_operator(.5,kernel_size=kernel_size).to(DEVICE)
    my_operator.kernel.data=kernel_torch-bias_per_point
    my_operator.bias.data=torch.tensor([bias],dtype=torch.float,device=DEVICE)

    torch.save(my_operator.state_dict(),'model/estimated_operator.pt')
    print('Weights of estimated operator are saved!')
            
        
