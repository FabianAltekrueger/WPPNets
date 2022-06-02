# This code belongs to the paper
#
# F. Altekrüger and J. Hertrich. 
# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 
# ArXiv Preprint#2201.08157
#
# Please cite the paper, if you use the code.
#
# The network WPPFlow in this code is adapted from 
# A. Lugmayr, M. Danelljan, L. Van Gool and R. Timofte
# SRFlow: Learning the super-resolution space with normalizing flow
# ECCV, 2020
#
# The function _add_downsample is taken from 
# A. Denker, M. Schmidt, J. Leuschner, P. Maaß.
# Conditional Invertible Neural Networks for Medical Imaging.
# MDPI Journal of Imaging, Inverse Problems and Imaging 7(11), 243 S., 2021. 
# available at
# https://github.com/jleuschn/cinn_for_imaging/blob/d71a308d90bd476c29c10a20950c8f9725fcb4b2/cinn_for_imaging/reconstructors/networks/cinn.py#L728

import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import math
import torch.nn.functional as F

def subnet_conv(c_in, c_out):
    last_conv = nn.Conv2d(128,  c_out, 3, padding=1)
    last_conv.weight.data.fill_(0)
    return nn.Sequential(nn.Conv2d(c_in, 128,   3, padding=1), nn.ReLU(),
                        last_conv)

def subnet_conv_1x1(c_in, c_out):
    last_conv = nn.Conv2d(128,  c_out, 1)
    last_conv.weight.data.fill_(0)
    return nn.Sequential(nn.Conv2d(c_in, 128,   1), nn.ReLU(),
                        last_conv)

def _add_downsample(nodes, clamping=1.5, use_act_norm=True):
    '''
    Downsampling operations. 
    '''
    nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                               name='reshape')) 
    for i in range(2):
        nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_conv,
                              'clamp':clamping}))
        if use_act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}))

def cond_flow_steps(nodes, cond, num_blocks, downsampling_level, clamping = 1.2):
    '''
    conditional flow steps
    input: nodes (network), condition, number of flow blocks
    '''
    for k in range(num_blocks):
		#conditional 3x3 convolution
        nodes.append(Ff.Node(nodes[-1].out0,
                     Fm.GLOWCouplingBlock,
                     {'subnet_constructor':subnet_conv, 'clamp':clamping},
                     conditions = cond,
                     name="conv_flow_{}.{}".format(downsampling_level,k)))
        #permutation in the channel dimension
        nodes.append(Ff.Node(nodes[-1].out0,
                     Fm.PermuteRandom,
                     {'seed':(k+1)*(downsampling_level+1)},
                     name='permute_flow_{}.{}'.format(downsampling_level,k)))
        #conditional affine transform/ affine injector
        nodes.append(Ff.Node(nodes[-1].out0, 
                     Fm.ConditionalAffineTransform,
                     {'subnet_constructor':subnet_conv, 'clamp':clamping},
                     conditions = cond,
                     name="affine_inject_{}.{}".format(downsampling_level,k)))
        #unconditional 1x1 convolution
        nodes.append(Ff.Node(nodes[-1].out0,
                     Fm.GLOWCouplingBlock,
                     {'subnet_constructor':subnet_conv_1x1, 'clamp':clamping},
                     name='conv1x1_flow_{}.{}'.format(downsampling_level,k)))
        #actnorm
        nodes.append(Ff.Node(nodes[-1].out0,Fm.ActNorm, {},
                     name='ActNorm_flow_{}.{}'.format(downsampling_level,k)))

def transition_step(nodes, downsampling_level, clamping = 1.2):
    '''
    transition step (last step in the scale)
    '''
    #unconditional 1x1 convolution
    nodes.append(Ff.Node(nodes[-1].out0,
                 Fm.GLOWCouplingBlock,
                 {'subnet_constructor':subnet_conv_1x1, 'clamp':clamping},
                 name='conv1x1_transition_{}'.format(downsampling_level)))
    #actnorm
    nodes.append(Ff.Node(nodes[-1].out0,Fm.ActNorm, {},
                 name='ActNorm_transition_{}'.format(downsampling_level)))

class WPPFlow(nn.Module):
    '''
    defines the WPPFlow
    input: superresolution scale, size of (quadratic) hr img
    '''
    def __init__(self, scale, hr_size):
        super().__init__()
        self.scale = scale
        if scale != 2 and scale != 4 and scale != 8:
            print(f'Error! Scale must be 2, 4 or 8, but it is {self.scale}.')
            exit() 
        self.hr_size = hr_size
        self.scales = 2 #number of different scales
        if self.scale == 8:
            self.scales = 3
        self.use_act_norm = True #in downsampling process
        self.num_blocks = 16
        if self.scale == 8:
            self.num_blocks = 10
        self.clamping = 1.2
        self.inn = self.build_inn()
        
    def forward(self,z,cond, rev=False):
        #use bicubic interpolation for the condition
        condition = []
        condition.append(cond)
        condition.append(F.interpolate(cond,scale_factor=2,mode='bicubic'))
        condition.append(F.interpolate(cond,scale_factor=4,mode='bicubic'))
        if self.scale == 8:
            condition.append(F.interpolate(cond,scale_factor=8,mode='bicubic'))
        condition = list(reversed(condition))
        if not rev:
            return self.inn(z, condition)
        else: 
            return self.inn(z, condition, rev=True)
        
    def build_inn(self):	
        nodes = [Ff.InputNode(1, self.hr_size, self.hr_size, name='input')]
        
        #downsample input
        for i in range(int(math.log(self.scale,2))):
            _add_downsample(nodes, use_act_norm=self.use_act_norm)

        #condition at different scales
        cond = []
        for i in range(self.scales+1):
            cond.append(Ff.ConditionNode(1,self.hr_size/(2**(i)),
                        self.hr_size/(2**(i)), name="cond_{}".format(i)))   
        #build the normalizing flow
        for s in range(int(self.scales)):
			#conditional flow step
            cond_flow_steps(nodes, cond[-s-1], num_blocks = self.num_blocks,
                            downsampling_level = s, clamping = self.clamping)
            #transition step
            transition_step(nodes, downsampling_level = s, clamping = self.clamping)
            #upsample step
            nodes.append(Ff.Node(nodes[-1], Fm.IRevNetUpsampling, {}))

        #conditional affine transform in the final scale    
        nodes.append(Ff.Node(nodes[-1].out0, 
                     Fm.ConditionalAffineTransform,
                     {'subnet_constructor':subnet_conv, 'clamp':1.2},
                     conditions = cond[0],
                     name="affine_inject_0.0"))        
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        return Ff.GraphINN(nodes+cond)
