# This code belongs to the paper
#
# F. Altekrueger and J. Hertrich. 
# WPPNets: Unsupervised CNN Training with Wasserstein Patch Priors for Image Superresolution. 
# ArXiv Preprint#2201.08157
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical example with the texture 'Grass' in the paper.

import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import time
import skimage.io as io
import model.small_acnet
import random
import wgenpatex as wp
import skimage.metrics as sm
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

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

def Trainset(size=1000):
	'''
	create training set consisting of low resolution images
	'''
	t = []
	picts = os.listdir('training_img/lr_grass')
	for img in picts:
		lr = wp.imread('training_img/lr_grass/' + img)
		t.append(lr)
	rand_ints = random.sample(range(0,len(t)),size) #take random set of 1000 training images
	train = []
	for i in range(size):
		train.append(t[rand_ints[i]])
	return train

def Validationset():
	'''
	create validation set (list) consisting of labeled lr-hr images 
	'''
	val = []
	picts_hr = os.listdir('training_img/validation_grass/val_hr')
	picts_lr = os.listdir('training_img/validation_grass/val_lr')
	for i in range(len(picts_hr)):
		val_hr = wp.imread('training_img/validation_grass/val_hr/' + picts_hr[i])
		val_lr = wp.imread('training_img/validation_grass/val_lr/' + picts_lr[i])
		val.append([val_lr,val_hr])
	return val	

def Downsample(input_img, scale = 0.25):
    ''' 
    downsamples an img by factor 4 using gaussian downsample from wgenpatex.py
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = 2
    kernel_size = 16
    gaussian_down = wp.gaussian_downsample(kernel_size,gaussian_std,int(1/scale),pad=True) #gaussian downsample with zero padding
    out = gaussian_down(input_img)
    return out

########################################################################
#inputs
net = model.small_acnet.Net(scale=4).to(device=DEVICE)

t = Trainset()
val = Validationset()	

args=argparse.Namespace()
args.lam=0.02
args.n_patches_out=10000
args.patch_size=6
args.n_iter_psi=10
args.val = val
args.keops = True
args.center = False

reference_img = wp.imread('test_img/ref_grass.png')
real_test = wp.imread('test_img/hr_grass.png')
test_img = wp.imread('test_img/lr_grass.png')# = Downsample(real_test) + 0.01*torch.randn_like(Downsample(real_test))

learning_rate = 1e-4
OPTIMIZER = torch.optim.Adam(net.parameters(), lr=learning_rate)

def WLoss(args, input_img, ref_pat, model, psi):
    '''
    Computes the wasserstein loss fct consisting of a MSELoss and a Wasserstein regularizer
    '''
    lam = args.lam
    n_patches_out = args.n_patches_out
    patch_size = args.patch_size
    n_iter_psi = args.n_iter_psi
    keops = args.keops
    
    im2patch = wp.patch_extractor(patch_size,center=args.center)
    
    num_ref = ref_pat.shape[0] #number of patches of reference image
    patch_weights = torch.ones(num_ref,device=DEVICE,dtype=torch.float) #same weight for all patches
    
    semidual_loss = wp.semidual(ref_pat,usekeops=keops) 
    semidual_loss.psi.data = psi #update the maximizer psi from previous step
    pred = model(input_img) #x4 of input_img
    
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
    
    down_pred = Downsample(pred) #downsample pred by factor 4

    loss_fct = nn.MSELoss()
    loss = loss_fct(down_pred,input_img) #||f(G(y)) - y||^2
    #print(f'MSELoss: {loss.data.item():>7f} und WReg: {lam*reg.data.item():>7f}')
    total_loss = loss + lam * reg
    
    return [total_loss,loss,lam*reg,psi]


def train_loop(args, batch, model, ref_pat, psi, optimizer): 
    '''
    input:batch of img, patches of reference img
    '''
    [total_loss,loss,reg,p] = WLoss(args, batch, ref_pat, model, psi)  
    
    #backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    total_loss = total_loss.item(); loss = loss.item(); reg = reg.item()
    return [total_loss,loss,reg,p]

def training(trainset, model, reference_img, batch_size, epochs, args = args, opti = OPTIMIZER):
    '''
    training process
    '''
    
    lr_img = torch.empty(0, device = DEVICE)
    for i in range(len(trainset)):
        lr_img = torch.cat([lr_img,trainset[i]])
    numb_train_img = lr_img.shape[0] #number of all img
    
    #create random batches:
    batch_size = min(batch_size, numb_train_img)
    rand_int_batch = random.sample(range(0,numb_train_img),numb_train_img)
    batch_ints = []
    for i in range(numb_train_img//batch_size):	#integers of the batches
        batch_ints.append(rand_int_batch[i*batch_size:(i+1)*batch_size])
    
    batch_lr = [] #list of batches    
    for i in range(len(batch_ints)):
        batch_temp = torch.empty(0, device = DEVICE) 
        for k in range(len(batch_ints[i])):
            tens = lr_img[batch_ints[i][k]].unsqueeze(0)
            batch_temp = torch.cat([batch_temp,tens])
        batch_lr.append(batch_temp)
    
    #create maximizer psi
    psi_length = args.n_patches_out #length of vector psi
    psi_list = []
    for i in range(len(batch_ints)):
        psi_list.append(torch.zeros(psi_length, device = DEVICE)) #create a list consisting of psi

    #create random patches of reference image
    im2patch = wp.patch_extractor(args.patch_size,center=args.center)
    number_patches = (reference_img.shape[2] - args.patch_size + 1)**2 #total number of patches
    n_patches_ref = min(args.n_patches_out,number_patches)
    rand_int_pat = random.sample(range(0,number_patches),n_patches_ref)
    ref = im2patch(reference_img)
    ref_pat = torch.empty(0, device = DEVICE)
    for i in rand_int_pat:
        ref_pat = torch.cat([ref_pat,ref[i].unsqueeze(0)])
    ref = ref_pat
    
    a_psnr_list = [] #for validation
    loss_list = []; reg_list = []; MSE_list = [] #for plot
    start = time.time()
    for t in range(epochs):
        print(f'Epoch {t+1} / {epochs}\n-------------------------------')
        start_ep = time.time()
        a_totalloss = 0; a_MSE = 0; a_reg = 0
        ints = random.sample(range(0,len(batch_lr)),len(batch_lr)) #random order of batches
        b = 0
        for i in ints:
            b += 1
            psi_temp = psi_list[i] #choose corresponding saved maximizer psi  
            [total_loss,loss,reg,p] = train_loop(args,batch_lr[i], model, ref, optimizer = opti, psi = psi_temp)
            print(f'loss: {total_loss:>7f}           {b * batch_size}/{len(batch_lr) * batch_size}') 	
    
            a_totalloss += total_loss; a_MSE += loss; a_reg += reg
            psi_list[i] = p #update psi

        a_totalloss = a_totalloss/len(batch_lr); a_MSE = a_MSE/len(batch_lr); a_reg = a_reg/len(batch_lr)
        loss_list.append(a_totalloss); MSE_list.append(a_MSE); reg_list.append(a_reg)
        end_ep = time.time()
        print(f'average loss: {a_totalloss:>7f}      time to end: {(end_ep - start_ep)*(epochs-t-1)//3600} h {((end_ep - start_ep)*(epochs-t-1)%3600)//60} min {int(((end_ep - start_ep)*(epochs-t-1)%3600)%60)} s')
        
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        
        val_step = 10
        if (t+1)%val_step == 0:
            print(f'------------------------------- \nValidation step')
            val_len = len(args.val)
            a_psnr = 0
            for i in range(val_len):
                pred = net(args.val[i][0])
                psnr_val = psnr(pred,args.val[i][1],40)
                a_psnr += psnr_val
            a_psnr = a_psnr / val_len
            a_psnr_list.append(a_psnr)
            plt.plot(list(range(val_step,val_step*len(a_psnr_list)+val_step,val_step)),a_psnr_list, 'k')
            title = 'Avarage PSNR ' + str(round(a_psnr,2))
            plt.title(title)
            plt.savefig('checkpoints/ValidatonPSNR_grass.png')
            plt.close()
            print(f'-------------------------------')
        
        #save a checkpoint
        if (t+1)%30 == 0:
            torch.save({'net_state_dict': model.state_dict(), 'optimizer_state_dict': opti.state_dict()}, 'checkpoints/epoch' + str(t+1) + 'grass.pth')

            pred_hr = model(test_img)
            if not os.path.isdir('checkpoints/tmp'):
                os.mkdir('checkpoints/tmp')
            save_img(pred_hr,'checkpoints/tmp/pred'+str(t+1))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.plot(list(range(len(loss_list))), loss_list, 'k-.', label='avarage loss')
            plt.plot(list(range(len(MSE_list))), MSE_list, 'k-', label='avarage MSE')
            plt.plot(list(range(len(reg_list))), reg_list, 'k:', label='avarage Reg')
            plt.legend(loc='upper right')
            plt.yscale('log')
            plt.savefig('checkpoints/losscurve_grass.png')
            plt.close()
            
    end = time.time()
    print('Done! Total time was '  + str((int(end - start)//60)) + 'min ' + str(int(time.time() - t)%60) + 's')

retrain = True
if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.mkdir('results')
    if retrain:
        training(t,net,reference_img,25,420)
        #torch.save({'net_state_dict': net.state_dict(), 'optimizer_state_dict': OPTIMIZER.state_dict()}, 'results/weights_grass.pth')        
        pred = net(test_img)
        save_img(pred,'results/pred_w2_grass')
    if not retrain:
        weights = torch.load('results/weights_grass.pth')
        net.load_state_dict(weights['net_state_dict'])
        pred = net(test_img)
        save_img(pred,'results/pred_w2_grass')
