# This code belongs to the paper
#
# F. Altekrüger and J. Hertrich. 
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# ArXiv Preprint#2201.08157
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical examples with the material 'SiC Diamonds'
# for a magnification factor 4 (accurate and inaccurate operator knowledge) 
# and magnification factor 6 in the paper.

import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import skimage.io as io
import model.small_acnet
import random
import utils
import argparse
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def Downsample(scale = 0.25, gaussian_std = 2):
    ''' 
    downsamples an img by factor 4 using gaussian downsample from utils.py
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = gaussian_std
    kernel_size = 16
    gaussian_down = utils.gaussian_downsample(kernel_size,gaussian_std,int(1/scale),pad=True) #gaussian downsample with zero padding
    return gaussian_down.to(DEVICE)

def WLoss(args, input_img, ref_pat, model, psi):
    '''
    Computes the proposed wasserstein loss fct consisting of a MSELoss and a Wasserstein regularizer
    '''
    lam = args.lam
    n_patches_out = args.n_patches_out
    patch_size = args.patch_size
    n_iter_psi = args.n_iter_psi
    keops = args.keops
    
    im2patch = utils.patch_extractor(patch_size,center=args.center)
    
    num_ref = ref_pat.shape[0] #number of patches of reference image
    patch_weights = torch.ones(num_ref,device=DEVICE,dtype=torch.float) #same weight for all patches
    
    semidual_loss = utils.semidual(ref_pat,usekeops=keops) 
    semidual_loss.psi.data = psi #update the maximizer psi from previous step
    pred = model(input_img) #superresolution of input_img
    
    #sum up all patches of whole batch
    inp_pat = torch.empty(0, device = DEVICE)
    for k in range(pred.shape[0]):
        inp = im2patch(pred[k,:,:,:].unsqueeze(0)) #use all patches of input_img
        inp_pat = torch.cat([inp_pat,inp],0)
    inp = inp_pat
    
    #gradient ascent to find maximizer psi for dual formulation of W2^2
    optim_psi = torch.optim.ASGD([semidual_loss.psi], lr=1e-0, alpha=0.5, t0=1)
    for i in range(n_iter_psi):
        sem = -semidual_loss(inp,patch_weights)
        optim_psi.zero_grad()
        sem.backward(retain_graph=True)
        optim_psi.step()
        
    semidual_loss.psi.data = optim_psi.state[semidual_loss.psi]['ax']
    psi = semidual_loss.psi.data #update psi
    
    reg = semidual_loss(inp,patch_weights) #wasserstein regularizer 
    
    down_pred = operator(pred) #downsample pred by scale_factor

    loss_fct = nn.MSELoss()
    loss = loss_fct(down_pred,input_img) #||f(G(y)) - y||^2
    total_loss = loss + lam * reg
    
    return [total_loss,loss,lam*reg,psi]


def training(trainset, model, reference_img, batch_size, epochs, args, opti):
    '''
    training process
    '''
    numb_train_img = trainset.shape[0] #number of all img
    
    #create random batches:
    idx = torch.randperm(numb_train_img)
    batch_lr = [] #list of batches
    for i in range(0,numb_train_img,batch_size):
        batch_lr.append(trainset[i:(i+batch_size),...])
    
    #create maximizer psi
    psi_length = args.n_patches_out #length of vector psi
    psi_list = []
    for i in range(len(batch_lr)):
        psi_list.append(torch.zeros(psi_length, device = DEVICE)) #create a list consisting of psi

    #create random patches of reference image
    im2patch = utils.patch_extractor(args.patch_size,center=args.center)
    ref = im2patch(reference_img,args.n_patches_out)
    
    a_psnr_list = [] #for validation
    loss_list = []; reg_list = []; MSE_list = [] #for plot

    for t in tqdm(range(epochs)):
        a_totalloss = 0; a_MSE = 0; a_reg = 0
        ints = random.sample(range(0,len(batch_lr)),len(batch_lr)) #random order of batches
        for i in tqdm(ints):
            psi_temp = psi_list[i] #choose corresponding saved maximizer psi  
            [total_loss,loss,reg,p] = WLoss(args, batch_lr[i], ref, model, psi_temp)  
    
            #backpropagation
            opti.zero_grad()
            total_loss.backward()
            opti.step()
            
            total_loss = total_loss.item(); loss = loss.item(); reg = reg.item()
            a_totalloss += total_loss; a_MSE += loss; a_reg += reg
            psi_list[i] = p #update psi
        a_totalloss = a_totalloss/len(batch_lr); a_MSE = a_MSE/len(batch_lr); a_reg = a_reg/len(batch_lr)
        loss_list.append(a_totalloss); MSE_list.append(a_MSE); reg_list.append(a_reg)
        
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        
        val_step = 10
        if (t+1)%val_step == 0:
            print(f'------------------------------- \nValidation step')
            val_len = len(args.val)
            a_psnr = 0
            for i in range(val_len):
                with torch.no_grad():
                    pred = net(args.val[i][0])
                psnr_val = utils.psnr(pred,args.val[i][1],40)
                a_psnr += psnr_val
            a_psnr = a_psnr / val_len
            print(f'Average Validation PSNR: {a_psnr}')    
            a_psnr_list.append(a_psnr)
            plt.plot(list(range(val_step,val_step*len(a_psnr_list)+val_step,val_step)),a_psnr_list, 'k')
            title = 'Avarage PSNR ' + str(round(a_psnr,2))
            plt.title(title)
            plt.savefig('checkpoints/ValidatonPSNR_SiC_'+operator_knowledge+'_x'+str(scale_factor)+'.pdf')
            plt.close()
            print(f'-------------------------------')
        
        #save a checkpoint
        if (t+1)%30 == 0:
            torch.save({'net_state_dict': model.state_dict()}, 'checkpoints/checkpoint_SiC_'+operator_knowledge+'_x'+str(scale_factor) +'.pth')
            with torch.no_grad():
                pred_hr = model(lr)
            if not os.path.isdir('checkpoints/tmp'):
                os.mkdir('checkpoints/tmp')
            utils.save_img(pred_hr,'checkpoints/tmp/pred'+str(t+1))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.plot(list(range(len(loss_list))), loss_list, 'k-.', label='avarage loss')
            plt.plot(list(range(len(MSE_list))), MSE_list, 'k-', label='avarage MSE')
            plt.plot(list(range(len(reg_list))), reg_list, 'k:', label='avarage Reg')
            plt.legend(loc='upper right')
            plt.yscale('log')
            plt.savefig('checkpoints/losscurve_SiC_'+operator_knowledge+'_x'+str(scale_factor)+'.pdf')
            plt.close()


    
retrain = False
if __name__ == '__main__':
    if not os.path.isdir('results'):
       os.mkdir('results')    
    
    scale_factor = 4 #choose between scale factor 4 and 6
    net = model.small_acnet.Net(scale=scale_factor).to(device=DEVICE)

    knowledge = ['accurate','inaccurate'] #choose operator knowledge
    operator_knowledge = knowledge[0]
    print('Superresolution of the material SiC with magnification factor ' +
           str(scale_factor) + ' and ' + operator_knowledge + ' operator knowledge')
    
    hr = utils.imread('test_img/hr_SiC.png')
    lr = utils.imread('test_img/lr_SiC_x' + str(scale_factor) + '.png')
    #  = operator(hr) + 0.01*torch.randn_like(operator(hr))
    if retrain:
        #inputs
        image_class = 'SiC'

        if operator_knowledge == 'accurate':
            if scale_factor == 4:
                operator = Downsample(scale = 1/scale_factor, gaussian_std = 2)
            elif scale_factor == 6:
                operator = Downsample(scale = 1/scale_factor, gaussian_std = 3)
        else:
            operator = Downsample(scale = 1/scale_factor, gaussian_std = 2.5)     

        lr_train = utils.Trainset(image_class = image_class+'_x'+str(scale_factor), size = 1000)
        val = utils.Validationset(image_class = image_class+'_x'+str(scale_factor))	
        lr_size = lr_train.shape[2]

        args=argparse.Namespace()
        args.lam=12.5/lr_size**2
        args.n_patches_out=10000
        args.patch_size=6
        args.n_iter_psi=20
        args.val = val
        args.keops = True
        args.center = False

        reference_img = utils.imread('test_img/ref_SiC.png')        
        
        #training process
        batch_size = 25
        if operator_knowledge == 'accurate':
            if scale_factor == 4:
                epochs = 450
                args.center = True
            if scale_factor == 6:
                epochs = 570
                args.lam=8/lr_size**2
        elif operator_knowledge == 'inaccurate':
            epochs = 420
        
        learning_rate = 1e-4
        OPTIMIZER = torch.optim.Adam(net.parameters(), lr=learning_rate)    
        
        training(lr_train,net,reference_img,batch_size,epochs,args=args,opti=OPTIMIZER)
        with torch.no_grad():
            pred = net(lr)
        torch.save({'net_state_dict': net.state_dict(), 'optimizer_state_dict': OPTIMIZER.state_dict()},
                    'results/weights_SiC_'+operator_knowledge+'_x'+str(scale_factor)+'.pth')        
        utils.save_img(pred,'results/W2_SiC_'+operator_knowledge+'_x'+str(scale_factor))
            
    if not retrain:
        weights = torch.load('results/weights_SiC_'+operator_knowledge+'_x'+str(scale_factor)+'.pth')
        net.load_state_dict(weights['net_state_dict'])
        pred = net(lr)
        utils.save_img(pred,'results/W2_SiC_'+operator_knowledge+'_x'+str(scale_factor))
