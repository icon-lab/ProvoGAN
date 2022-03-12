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

            
            in_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+in_type1+'.nii'))
            in_im2=np.float32(load_nifty_volume_as_array(filename=target_subject+in_type2+'.nii'))
            out_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+out_type+'.nii'))
            

           
            

            out_im1[out_im1<0] = 0
            in_im1[in_im1<0] = 0
            in_im2[in_im2<0] = 0

            #subject based normalization
            out_im1=1.0*out_im1/out_im1.max()
            in_im1=1.0*in_im1/in_im1.max()
            in_im2=1.0*in_im2/in_im2.max()




            shapes=out_im1.shape
            s1=shapes[0]
            s2=shapes[1]
            s3=shapes[2]

            if order_c[0]=='A':
                slice_size=s1
            elif order_c[0]=='C':
                slice_size=s2
            else:
                slice_size=s3



            fake_syn=np.zeros([s1,s2,s3])
            
            for ind in range(slice_size):

                if order_c[0]=='A':
                    data_x=np.array([in_im1[ind,:,:],in_im2[ind,:,:]])
                    data_y=np.array([out_im1[ind,:,:]])
                elif order_c[0]=='C':
                    data_x=np.array([in_im1[:,ind,:],in_im2[:,ind,:]])
                    data_y=np.array([out_im1[:,ind,:]])
                else:    
                    data_x=np.array([in_im1[:,:,ind],in_im2[:,:,ind]])
                    data_y=np.array([out_im1[:,:,ind]])

                

                data_x=(data_x*1.0-0.5)*2
                data_y=(data_y*1.0-0.5)*2
                
                # data_x=np.float32(data_x)
                # data_y=np.float32(data_y)
                    
                data_x=np.expand_dims(data_x,axis=0)
                data_y=np.expand_dims(data_y,axis=0)

                data={'A': torch.from_numpy(data_x), 'B':torch.from_numpy(data_y), 'A_paths':target_subject, 'B_paths':target_subject}  

                model.set_input(data)
                model.test()
                

                fake_im=model.fake_B.cpu().data.numpy()
                fake_im=fake_im*0.5+0.5

                if order_c[0]=='A':
                    fake_syn[ind,:,:]=fake_im
                elif order_c[0]=='C':
                    fake_syn[:,ind,:]=fake_im
                else:
                    fake_syn[:,:,ind]=fake_im

            save_array_as_nifty_volume(fake_syn, filename=target_subject+out_type+'_syn_3dpro_'+order_c[0]+'_1.nii')
        

        
        

