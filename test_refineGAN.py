import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from data_custom.data_load import load_nifty_volume_as_array
from data_custom.data_load import save_array_as_nifty_volume
import numpy as np
import torch
import math

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    in_type1=opt.input1
    in_type2=opt.input2
    out_type=opt.out


    for phase in ['train','val','test']:

        target_path=opt.dataroot+'/'+phase
        patients=os.listdir(target_path)
        order_c=opt.order
        for i in range(len(patients)):
    #        if i >= opt.how_many:
    #            break
            
            target_subject=os.path.join(target_path,patients[i])+'/'
            target_data=target_subject+opt.data_type+'_'+str(opt.Rx)+'x.mat'            
            f = h5py.File(target_data,  "r")
            images_us=f.get('images_us')
            images_us=np.array(images_us)
            normalizer=(abs(images_us).max())
            images_us=images_us/normalizer
            
            #mask
            mask_im=f.get('map')
            mask_im=np.array(mask_im)
            mask_im=np.expand_dims(mask_im,0)
            mask_im=np.tile(mask_im, (slice_size,1,1))
            mask_im=np.fft.ifftshift(mask_im,axes=(1,2))
                       
            images_fs=f.get('images_fs')
            images_fs=np.array(images_fs)
            images_fs=images_fs/normalizer
            f.close()

          





            shapes=images_fs.shape
            s1=shapes[0]
            s2=shapes[1]
            s3=shapes[2]



            fake_recon=np.zeros([s1,s2,s3])
            
            for ind in range(s1):

                data_x=np.array([images_fs[ind,:,:],mask_im[ind,:,:]])

                    
                data_x=np.expand_dims(data_x,axis=0)

                data={'A': torch.from_numpy(data_x), 'A_paths':target_subject}  

                model.set_input(data)
                model.test()
                

                fake_im=model.fake_B.cpu().data.numpy()
                fake_im=fake_im*0.5+0.5

                fake_recon[ind,:,:]=fake_im

            f = h5py.File(target_subject+opt.data_type+'_refineGAN_'+str(opt.Rx)+'x.mat',  "w")
            f.create_dataset('images_recon', data=fake_recon)
            f.close()
        

        
        

