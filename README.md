# ProvoGAN
Official Pytorch Implementation of Progressively Volumetrized Deep Generative Models for Data-Efficient Contextual Learning of MR Image Recovery (ProvoGAN) which is described in the [following](https://arxiv.org/abs/2011.13913) paper:

Mahmut Yurt and Muzaffer Özbey and Salman Ul Hassan Dar and Berk Tınaz and Kader Karlı Oğuz and Tolga Çukur Progressively volumetrized deep generative models for data-efficient contextual learning of MR image recovery. arXiv. 2022.

## Data
The *data* directory should be organized as the following structure. '.nii' files are for synthesis tasks, '.mat' files are for reconstruction tasks. REconstruction files should contain a variables named images_fs, images_us and map for the fully-sampled, under-sampled and sampling mask of target data:

```
data
│
└───train
|   |
│   └─── train subject 1
|   |         modality_1.nii
|   |         modality_2.nii
|   |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
│   └─── train subject 2
|   |         modality_1.nii
|   |         modality_2.nii
|   |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
│   ...
|
└───val
|   |
│   └─── val subject 1
|   |         modality_1.nii
|   |         modality_2.nii
|   |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
│   └─── val subject 2
|   |         modality_1.nii
|   |         modality_2.nii
|   |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
│   ...
|   
└───test
    |
    └─── test subject 1
    |         modality_1.nii
    |         modality_2.nii
    |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
    └─── test subject 2
    |         modality_1.nii
    |         modality_2.nii
    |         ...
|   |         target.nii
|   |         ...
|   |         target_Rx.mat
    ...
```

# Demo

## provoGAN

sample_run.sh file contains 3 consecutive training and testing commend. To run the code please organize your data folder as the explained structure. Also edit the following arguments according to your choices;

name - name of the experiment  <br />
lambda_A - weighting of the pixel-wise loss function  <br />
niter, n_iter_decay - number of epochs with normal learning rate and number of epochs for which the learning leate is decayed to 0. Total number of epochs is equal to sum of them  <br />
save_epoch_freq -frequency of saving models <br />
order -order selection for 3 stage of ProvoGAN <br />
input1 -constrast name of input modelity 1 <br />
input2 -constrast name of input modelity 2 <br />
out    -constrast name of target modelity  <br />

## sGAN
A sample run commend example for training and testing: 
```
python train.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 100 --dataset_mode provo_stage1 --norm batch --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD

```

```
python test_sGAN.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual --which_model_netG resnet_9blocks  --dataset_mode provo_stage1 --norm batch --phase test --output_nc 1 --input_nc 2 --gpu_ids 0 --serial_batches  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD
```

## vGAN
A sample run commend example for training and testing: 
```
python train.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual_vGAN --which_model_netG resnet_9blocks_3D --which_model_netD basic_3D  --which_direction AtoB --lambda_A 100 --dataset_mode vGAN --norm batch_3D --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD

```

```
python test_vGAN.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual_vGAN --which_model_netG resnet_9blocks_3D  --dataset_mode vGAN --norm batch_3D --phase test --output_nc 1 --input_nc 2 --gpu_ids 0 --serial_batches  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD
```

## transferGAN
For transferGAN, initially, a cross-section-based 2D network is trained, then its weights are used for weight initialization of volumetric 3D network.
A sample run commend example for trainings and testing:

2D pretrainig
```
python train.py --dataroot datasets/IXI --name transferGAN_pre_train --model pix2pix_perceptual --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 100 --dataset_mode provo_stage1 --norm batch --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD 
```

3D volumetric model with weight tranfer
```
python train.py --dataroot datasets/IXI --name transferGAN_sample --model pix2pix_perceptual_transferGAN --which_model_netG resnet_9blocks_3D --which_model_netD basic_3D  --which_direction AtoB --lambda_A 50 --dataset_mode vGAN --norm batch_3D --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD --checkpoints_dir_old /checkpoints/revisions/ --which_model_netG_old resnet_9blocks --name_old transferGAN_pre_train

```

```
python test_transferGAN.py --dataroot datasets/IXI --name transferGAN_sample --model pix2pix_perceptual_transferGAN --which_model_netG resnet_9blocks_3D  --dataset_mode vGAN --norm batch_3D --phase test --output_nc 1 --input_nc 2 --gpu_ids 0 --serial_batches  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD
```


## SC-GAN
A sample run commend example for training and testing: 
```
python train.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual_vGAN --which_model_netG unet_att_3D --which_model_netD basic_att_3D --which_direction AtoB --lambda_A 100 --dataset_mode vGAN --norm batch_3D --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD

```

```
python test_SC-GAN.py --dataroot datasets/IXI --name datasets/IXI --model pix2pix_perceptual_vGAN --which_model_netG unet_att_3D  --dataset_mode vGAN --norm batch_3D --phase test --output_nc 1 --input_nc 2 --gpu_ids 0 --serial_batches  --checkpoints_dir /checkpoints/revisions/ --input1 T1 --input2 T2 --out PD
```

## refineGAN
A sample run commend example for training and testing: 
```
python train.py --dataroot datasets/IXI --name refineGAN_sample --model pix2pix_perceptual_refineGAN --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 10 --dataset_mode refineGAN --norm batch --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 5  --checkpoints_dir /checkpoints/revisions/ --Rx 4 --data_type T1

```

```
python test_refineGAN.py --dataroot datasets/IXI --name refineGAN_sample --model pix2pix_perceptual_refineGAN --which_model_netG resnet_9blocks  --dataset_mode refineGAN --norm batch --phase test --output_nc 1 --input_nc 2 --gpu_ids 0 --serial_batches  --checkpoints_dir /checkpoints/revisions/ --Rx 4 --data_type T1
```

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{yurt2020progressively,
  title={Progressively volumetrized deep generative models for data-efficient contextual learning of MR image recovery},
  author={Yurt, Mahmut and {\"O}zbey, Muzaffer and Dar, Salman Ul Hassan and T{\i}naz, Berk and O{\u{g}}uz, Kader Karl{\i} and {\c{C}}ukur, Tolga},
  journal={arXiv preprint arXiv:2011.13913},
  year={2020}
}
```
For any questions, comments and contributions, please contact Muzaffer Özbey (muzaffer[at]ee.bilkent.edu.tr) <br />

(c) ICON Lab 2022

## Acknowledgments
This code uses libraries from [pGAN](https://github.com/icon-lab/pGAN-cGAN) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.