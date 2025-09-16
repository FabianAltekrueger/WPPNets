# This code belongs to the paper
#
# F. Altekrüger and J. Hertrich. 
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# SIAM Journal on Imaging Sciences, vol. 16(3), pp. 1033-1067.
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical examples with the material 'SiC Diamonds'
# for a magnification factor 4 (accurate and inaccurate operator knowledge) 
# and magnification factor 6 in the paper.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import json
import os
import skimage.io as io
from tqdm import tqdm
import model.small_acnet
import utils

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

base = ''

class Wasserstein_trainer(nn.Module):
    def __init__(self,net,optimizer,operator,trainset,valset,
                    ref_pat,patch_size,lam,
                    patches_out=10000,psi_iter=20,keops=True,
                    center=False):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.operator = operator
        self.trainset = trainset
        self.valset = valset
        self.ref_pat = ref_pat
        self.patch_size = patch_size
        self.lam = lam
        self.patches_out = patches_out
        self.psi_iter = psi_iter
        self.keops = keops
        self.center = center
        self.data_fid = nn.MSELoss()
        #create semidual version of Wasserstein
        self.semidual_loss = utils.semidual(self.ref_pat,device=DEVICE,
                                    usekeops=self.keops)
        #patch extractor
        self.im2patch = utils.patch_extractor(self.patch_size,
                                        center=self.center)

    def WLoss(self,inp):
        '''
        Computes the proposed wasserstein loss 
        '''
        optim_psi = torch.optim.ASGD([self.semidual_loss.psi], 
                                    lr=1e-0, alpha=0.5, t0=1)
        tmp_inp = inp.detach()
        for i in range(self.psi_iter):
            sem = -self.semidual_loss(tmp_inp)
            optim_psi.zero_grad()
            sem.backward()
            optim_psi.step()
        self.semidual_loss.psi.data = optim_psi.state[self.semidual_loss.psi]['ax']
        reg = self.semidual_loss(inp) #wasserstein regularizer 
        return reg

    def train(self,batch_size,epochs,savename):
        train_loader = DataLoader(self.trainset,batch_size=batch_size,
                            shuffle=True,drop_last=True)
        val_loader = DataLoader(self.valset,
                            batch_size=1,shuffle=False)
        
        val_loss = float('inf')
        writer = SummaryWriter(
                        f'{base}checkpoints/tensorboard_logs/{savename}')

        for ep in range(epochs):
            loop = tqdm(train_loader, desc = f'Epoch {ep+1} / {epochs}')
            running_loss = 0 
            running_w2 = 0
            running_l2 = 0
            
            for obs in loop:
                self.optimizer.zero_grad()
                obs = obs.to(DEVICE)
                pred = self.net(obs)
                pred_pat = self.im2patch(pred)
                down_pred = self.operator(pred)
                
                wloss = self.WLoss(pred_pat)
                l2 = self.data_fid(down_pred,obs) #||f(G(y)) - y||^2
                loss = l2 + self.lam * wloss
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                running_w2 += wloss.item()
                running_l2 += l2.item()
                
            # tensorboard values
            writer.add_scalar(f'Loss/loss', running_loss/len(train_loader), ep+1)
            writer.add_scalar(f'Loss/w2', running_w2/len(train_loader), ep+1)
            writer.add_scalar(f'Loss/l2', running_l2/len(train_loader), ep+1)

            #validation
            with torch.no_grad():
                running_mse = 0
                for gt,obs in val_loader:
                    pred = self.net(obs.to(DEVICE))
                    mse = self.data_fid(pred,gt.to(DEVICE))
                    running_mse += mse.item()
            writer.add_scalar(f'Loss/val_mse', running_mse/len(val_loader), ep+1)
            if (ep+1)%50==0:
                writer.add_image('Valid', pred.squeeze(0), ep+1)
            writer.flush()
            
            if running_mse/len(val_loader) < val_loss:
                val_loss = running_mse/len(val_loader)
                torch.save({'net_state_dict':self.net.state_dict(), 
                            'epoch': ep}, 
                            f'{base}checkpoints/{savename}/minval.pth')
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Wasserstein Trainer',
                                usage='Implementation of the WPPNet')
    train_group = parser.add_argument_group('Training parameters')
    train_group.add_argument('-t', '--train', action='store_true', 
                            help='Training option.')
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--epochs', type=int, default=500)
    train_group.add_argument('-lr', '--learning_rate', default=1e-4)
    
    setting_group = parser.add_argument_group('Settings')
    setting_group.add_argument('--num', type=int, default=1,
                            help='Number of experiment for saving')
    setting_group.add_argument('-s', '--scale', type=int, default=4, 
                            choices=[4,6],
                            help='Scaling factor')
    setting_group.add_argument('-a', '--accurate', type=str, default='accurate',
                            choices = ['accurate','inaccurate'],
                            help='Decide between accurate and inaccurate \
                            operator knowledge')
    setting_group.add_argument('-p', '--patch_size', type=int, default=6)  
    setting_group.add_argument('--keops', action='store_false',
                            help='Use keops')
    setting_group.add_argument('--center', action='store_false',
                            help='Center the patches') 
    
    wloss_group = parser.add_argument_group('Wasserstein computation')                    
    wloss_group.add_argument('-l', '--lam', type=float, default=0.02,
                            help='Regularization strength')
    wloss_group.add_argument('-it', '--n_iter_psi', type=int, default=20,
                            help='Number of iterations for finding \
                            maximizer psi in dual Wasserstein formulation')
    wloss_group.add_argument('-pat_out', '--n_patches_out', type=int, 
                            default=10000, help='Number of patches from \
                            reference image to compute Wasserstein loss')
    args = parser.parse_args()
    
    if not os.path.isdir('results'):
       os.mkdir('results')    
    
    net = model.small_acnet.Net(scale=args.scale).to(DEVICE)

    print(f'Superresolution of the material SiC with magnification factor \
           {args.scale} and {args.accurate} operator knowledge')
    
    if args.train:
        image_class = 'SiC'
        if args.accurate == 'accurate':
            if args.scale == 4:
                std = 2
            elif args.scale == 6:
                std = 3
        else:
            std = 2.5  
        operator = utils.SR_operator(scale_factor=1/args.scale, 
                            gaussian_std = std)

        opti = torch.optim.Adam(net.parameters(), lr=args.learning_rate)    
        
        lr_train = utils.SiC(datafile = 
                    f'{base}training_img/train_{image_class}_x{args.scale}.h5',
                    split = 'train')
        val = utils.SiC(datafile = 
                    f'{base}training_img/val_{image_class}_x{args.scale}.h5',
                    split = 'val')
                    
        #create random patches of reference image
        im2patch = utils.patch_extractor(args.patch_size,center=args.center)
        reference_img = utils.imread('test_img/ref_SiC.png')        
        ref = im2patch(reference_img,args.n_patches_out)      
        
        wtrainer = Wasserstein_trainer(net=net,optimizer=opti,
                    operator=operator,trainset=lr_train,valset=val,
                    ref_pat = ref,patch_size=args.patch_size,
                    lam=args.lam,patches_out=args.n_patches_out,
                    psi_iter=args.n_iter_psi,keops=args.keops,
                    center=args.center)
                    
        savename = f'SiC_{args.accurate}_x{args.scale}_exp{args.num}'
        if not os.path.isdir(f'{base}checkpoints/{savename}'):
            os.makedirs(f'{base}checkpoints/{savename}')
        #save config
        with open(f'{base}checkpoints/{savename}/config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)            
        
        #start training process
        wtrainer.train(args.batch_size,args.epochs,savename)
        
        torch.save({'net_state_dict': wtrainer.net.state_dict(), 
                    'optimizer_state_dict': wtrainer.optimizer.state_dict()},
                    f'{base}checkpoints/{savename}/weights.pth')   
        
    else:
        with torch.no_grad():
            hr = utils.imread('test_img/hr_SiC.png')
            lr = utils.imread(f'test_img/lr_SiC_x{args.scale}.png')
            #  = operator(hr) + 0.01*torch.randn_like(operator(hr))
            name = f'checkpoints/SiC_{args.accurate}_x{args.scale}_exp{args.num}/minval.pth'
            weights = torch.load(name,map_location=DEVICE)
            net.load_state_dict(weights['net_state_dict'])
            pred = net(lr)
            io.imsave(f'results//pred_SiC_{args.accurate}_x{args.scale}_exp{args.num}.png',
                        np.clip(pred.squeeze().numpy(),0,1))
