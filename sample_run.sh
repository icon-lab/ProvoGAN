order_c=ACS
gpu_c=0
in_type1=T2
in_type2=PD
out_type=T1



"""
#first run
"""


python train.py --dataroot ./ --name ${out_type}syn_${order_c}_stage1 --model pix2pix_perceptual --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 100 --dataset_mode provo_stage1 --norm batch --pool_size 0 --output_nc 1 --input_nc 2 --gpu_ids ${gpu_c} --niter 50 --niter_decay 50 --save_epoch_freq 5  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}


python test_stage1.py --dataroot ./ --name ${out_type}syn_${order_c}_stage1 --model pix2pix_perceptual --which_model_netG resnet_9blocks  --dataset_mode provo_stage1 --norm batch --phase test --output_nc 1 --input_nc 2 --gpu_ids ${gpu_c} --serial_batches  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}


"""
#2nd run
"""


python train.py --dataroot ./ --name ${out_type}syn_${order_c}_stage2 --model pix2pix_perceptual_residual --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 100 --dataset_mode provo_stage2 --norm batch --pool_size 0 --output_nc 1 --input_nc 3 --gpu_ids ${gpu_c} --niter 50 --niter_decay 50 --save_epoch_freq 5  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}


python test_stage2.py --dataroot ./ --name ${out_type}syn_${order_c}_stage2 --model pix2pix_perceptual_residual --which_model_netG resnet_9blocks  --dataset_mode provo_stage2 --norm batch --phase test --output_nc 1 --input_nc 3 --gpu_ids ${gpu_c} --serial_batches  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}

"""
#3rd run
"""


python train.py --dataroot ./ --name ${out_type}syn_${order_c}_stage3 --model pix2pix_perceptual_residual --which_model_netG resnet_9blocks  --which_direction AtoB --lambda_A 100 --dataset_mode provo_stage3 --norm batch --pool_size 0 --output_nc 1 --input_nc 3 --gpu_ids ${gpu_c} --niter 50 --niter_decay 50 --save_epoch_freq 5  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}


python test_stage3.py --dataroot ./ --name ${out_type}syn_${order_c}_stage3 --model pix2pix_perceptual_residual --which_model_netG resnet_9blocks  --dataset_mode provo_stage3 --norm batch --phase test --output_nc 1 --input_nc 3 --gpu_ids ${gpu_c} --serial_batches  --order ${order_c} --checkpoints_dir /checkpoints/revisions/ --input1 ${in_type1} --input2 ${in_type2} --out ${out_type}


