# WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution. 

This code belongs to the paper [1] available at https://epubs.siam.org/doi/full/10.1137/22M1496542. A preprint can be found at https://arxiv.org/abs/2201.08157.
Please cite the paper, if you use this code.

The repository contains implementations of WPPNets and WPPFlows as introduced in [1]. It contains scripts for reproducing the numerical examples from Section 5.

Moreover, the file `utils.py` contains functions adapted from [2] available at https://github.com/johertrich/Wasserstein_Patch_Prior (see also [3]). Furthermore, the folder `model` contains a CNN adapted from [6] available at https://github.com/hellloxiaotian/ACNet and a normalizing flow adapted from [5].

The folders `test_img` and `training_img` contain parts of the textures from [4] and the used images of material microstructures have been acquired in the frame of the EU Horizon 2020 Marie Sklodowska-Curie Actions Innovative Training Network MUMMERING (MUltiscale, Multimodal and Multidimensional imaging for EngineeRING, Grant Number 765604) at the beamline TOMCAT of the SLS by A. Saadaldin, D. Bernard, and F. Marone Welford.

For questions and bug reports, please contact Fabian Altekrüger (fabian.altekrueger@hu-berlin.de) or Johannes Hertrich (j.hertrich@math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS  
2. USAGE AND EXAMPLES
3. REFERENCES

## 1. REQUIREMENTS

The code requires several Python packages. We tested the code with Python 3.9.7 and the following package versions:

- freia 0.2
- matplotlib 3.4.3
- numpy 1.21.2
- pykeops 1.5
- pytorch 1.10.0
- tqdm 4.62.3

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE AND EXAMPLES

You can start the training of the WPPNet or WPPFlow by calling the scripts. If you want to load the existing network, please set `retrain` to `False`. Checkpoints are saved automatically during training such that the progress of the reconstructions is observable. Feel free to vary the parameters and see what happens. 

### 2.1 WPPNets

#### TEXTURES

The script `run_texture.py` is the implementation of the superresolution example in [1, Section 5.1] for the Kylberg Textures [4] `grass` and `floor`, which are available at https://kylberg.org/kylberg-texture-dataset-v-1-0. The high-resolution ground truth and the reference image are different 600×600 sections cropped from the original texture images. Similarly, the low-resolution training data is generated by cropping 100×100 sections from the texture images, artificially downsampling it by a predefined forward operator f and adding Gaussian noise. For more details on the downsampling process, see [1, Section 5.1]. 

#### MATERIAL DATA

##### SiC Diamonds

The script `run_SiC.py` is the implementation of the superresolution examples in [1, Section 5.2-5.3] for the `SiC Diamonds`. The high-resolution ground truth and the reference image are different 600×600 slices extracted from a 3D image. The low-resolution training data is generated by cropping 100×100 sections from the material images, artificially downsampling it by a predefined forward operator f and adding Gaussian noise.

In the script you can choose between the scale factors 4 and 6. Moreover, for scale factor 4 there is a choice between an accurate (training and generation of the low-resolution images with the same forward operator) or inaccurate (generation of the low-resolution images as before, training with a bit different operator) forward operator knowledge. For details on the forward operator, see [1, Section 5.2-5.3].

##### Fontainebleau sandstone

The script `run_FS.py` is the implementation of the superresolution example in [1, Section 5.2] for the `Fontainebleau sandstone`. The high-resolution ground truth and the reference image are different 600×600 slices extracted from a 3D image. The low-resolution training data is generated by cropping 100×100 sections from the material images, artificially downsampling it by a predefined forward operator f and adding Gaussian noise.

The script `run_FS_estimated.py` is the implementation of the superresolution examples in [1, Section 5.3] for the `Fontainebleau sandstone`. Here, the forward operator is estimated using a registered, real-world data pair ($\tilde{x},\tilde{y}$) of a high- and low-resolution image, respectively. The training data consists of cropped 50x50 sections from real-world low-resolution data. 

### 2.2 WPPFlows

The script `run_WPPFlow.py` is the implementation of the uncertainty quantification in [1, Section 5.4] for the `SiC Diamonds`. Here you can choose between the scale factors 4 and 8. While the training data for scale factor 4 is the same as for WPPNets with scale factor 4, for scale factor 8 the low-resolution training data is generated by cropping 160×160 sections from the material images, artificially downsampling it by a predefined forward operator f and adding Gaussian noise. For details on the forward operator, see [1, Section 5.4].


## 3. REFERENCES

[1] F. Altekrüger, J. Hertrich.  
WPPNets and WPPFlows: The Power of Wasserstein Patch Priors for Superresolution.  
SIAM Journal on Imaging Sciences, vol. 16(3), pp. 1033-1067.

[2] J. Hertrich, A. Houdard and C. Redenbach.  
Wasserstein Patch Prior for Image Superresolution.  
IEEE Transactions on Computational Imaging, vol. 8, pp. 693-704, 2022.

[3] A. Houdard, A. Leclaire, N. Papadakis and J. Rabin.  
Wasserstein Generative Models for Patch-based Texture Synthesis.  
ArXiv Preprint#2007.03408

[4] G. Kylberg.  
The Kylberg texture dataset v. 1.0.  
Centre for Image Analysis, Swedish University of Agricultural Sciences and Uppsala University, 2011.

[5] A. Lugmayr, M. Danelljan, L. Van Gool and R. Timofte.  
SRFlow: Learning the super-resolution space with normalizing flow.  
ECCV, 2020.

[6] C. Tian, Y. Xu, W. Zuo, C.-W. Lin, and D. Zhang.  
Asymmetric CNN for image superresolution.  
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2021.
