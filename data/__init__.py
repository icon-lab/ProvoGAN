import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
import nibabel
import os
import SimpleITK as sitk

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)



def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


# def CreateDataset(opt):
    dataset = None
    # if opt.dataset_mode !='aligned_mat' and opt.dataset_mode !='unaligned_mat':
    #     if opt.dataset_mode == 'aligned':
    #         from data.aligned_dataset import AlignedDataset
    #         dataset = AlignedDataset()
    #     elif opt.dataset_mode == 'unaligned':
    #         from data.unaligned_dataset import UnalignedDataset
    #         dataset = UnalignedDataset()
    #     elif opt.dataset_mode == 'single':
    #         from data.single_dataset import SingleDataset
    #         dataset = SingleDataset()
    #     else:
    #         raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)  
    #         print("dataset [%s] was created" % (dataset.name()))
    #     dataset.initialize(opt)
         
    #custom data loader            
    if opt.dataset_mode == 'provo_stage1':
        order_c=opt.order
        dataset=[]
        target_path=opt.dataroot+'/'+opt.phase
        patients_target=os.listdir(target_path)

        for ind in range(len(patients_target)):
            target_subject=os.path.join(target_path,patients_target[ind])+'/'

            out_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'.nii'))
            #subject based normalization
            out_im=1.0*out_im/out_im.max()

            in_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input1+'.nii'))
            in_im1=1.0*in_im1/in_im1.max()

            in_im2=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input2+'.nii'))
            in_im2=1.0*in_im2/in_im2.max()

            out_im=(out_im-0.5)*2
            in_im1=(in_im1-0.5)*2
            in_im2=(in_im2-0.5)*2

            if order_c[0]=='A':
                slice_number=in_im1.shape[0]
            elif order_c[0]=='C':
                slice_number=in_im1.shape[1]
            else:
                slice_number=in_im1.shape[2]

            for slice_ind in range(slice_number):
                if order_c[0]=='A':
                    data_x=np.array([in_im1[slice_ind,:,:],in_im2[slice_ind,:,:]])
                    data_y=np.array([in_im2[slice_ind,:,:]])
                elif order_c[0]=='C':
                    data_x=np.array([in_im1[:,slice_ind,:],in_im2[:,slice_ind,:]])
                    data_y=np.array([in_im2[:,slice_ind,:]])
                else:
                    data_x=np.array([in_im1[:,:,slice_ind],in_im2[:,:,slice_ind]])
                    data_y=np.array([in_im2[:,:,slice_ind]])

                # data_x=np.expand_dims(data_x,axis=0)
                # data_y=np.expand_dims(data_y,axis=0)
                dataset.append({'A': torch.from_numpy(data_x), 'B':torch.from_numpy(data_y), 'A_paths':opt.dataroot, 'B_paths':opt.dataroot})


    elif opt.dataset_mode == 'provo_stage2':
        order_c=opt.order
        dataset=[]
        target_path=opt.dataroot+'/'+opt.phase
        patients_target=os.listdir(target_path)

        for ind in range(len(patients_target)):
            target_subject=os.path.join(target_path,patients_target[ind])+'/'

            out_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'.nii'))
            #subject based normalization
            out_im=1.0*out_im/out_im.max()

            in_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input1+'.nii'))
            in_im1=1.0*in_im1/in_im1.max()

            in_im2=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input2+'.nii'))
            in_im2=1.0*in_im2/in_im2.max()

            recovered_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'_syn_3dpro_'+order_c[0]+'_1.nii'))

            out_im=(out_im-0.5)*2
            in_im1=(in_im1-0.5)*2
            in_im2=(in_im2-0.5)*2
            recovered_im=(recovered_im-0.5)*2

            if order_c[1]=='A':
                slice_number=in_im1.shape[0]
            elif order_c[1]=='C':
                slice_number=in_im1.shape[1]
            else:
                slice_number=in_im1.shape[2]

            for slice_ind in range(slice_number):
                if order_c[1]=='A':
                    data_x=np.array([in_im1[slice_ind,:,:],in_im2[slice_ind,:,:],recovered_im[slice_ind,:,:]])
                    data_y=np.array([out_im1[slice_ind,:,:]])
                    data_fake1=np.array([recovered_im[slice_ind,:,:]])
                elif order_c[1]=='C':
                    data_x=np.array([in_im1[:,slice_ind,:],in_im2[:,slice_ind,:],recovered_im[:,slice_ind,:]])
                    data_y=np.array([out_im1[:,slice_ind,:]])
                    data_fake1=np.array([recovered_im[:,slice_ind,:]])
                else:
                    data_x=np.array([in_im1[:,:,slice_ind],in_im2[:,:,slice_ind],recovered_im[:,:,slice_ind]])
                    data_y=np.array([out_im1[:,:,slice_ind]])
                    data_fake1=np.array([recovered_im[:,:,slice_ind]])

                # data_x=np.expand_dims(data_x,axis=0)
                # data_y=np.expand_dims(data_y,axis=0)
                # data_fake1=np.expand_dims(data_fake1,axis=0)
                dataset.append({'A': torch.from_numpy(data_x), 'B':torch.from_numpy(data_y), 'fake1':torch.from_numpy(data_fake1), 'A_paths':opt.dataroot, 'B_paths':opt.dataroot})


    elif opt.dataset_mode == 'provo_stage3':
        order_c=opt.order
        dataset=[]
        target_path=opt.dataroot+'/'+opt.phase
        patients_target=os.listdir(target_path)

        for ind in range(len(patients_target)):
            target_subject=os.path.join(target_path,patients_target[ind])+'/'

            out_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'.nii'))
            #subject based normalization
            out_im=1.0*out_im/out_im.max()

            in_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input1+'.nii'))
            in_im1=1.0*in_im1/in_im1.max()

            in_im2=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input2+'.nii'))
            in_im2=1.0*in_im2/in_im2.max()

            recovered_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'_syn_3dpro_'+order_c+'_2.nii'))

            out_im=(out_im-0.5)*2
            in_im1=(in_im1-0.5)*2
            in_im2=(in_im2-0.5)*2
            recovered_im=(recovered_im-0.5)*2

            if order_c[2]=='A':
                slice_number=in_im1.shape[0]
            elif order_c[2]=='C':
                slice_number=in_im1.shape[1]
            else:
                slice_number=in_im1.shape[2]

            for slice_ind in range(slice_number):
                if order_c[2]=='A':
                    data_x=np.array([in_im1[slice_ind,:,:],in_im2[slice_ind,:,:],recovered_im[slice_ind,:,:]])
                    data_y=np.array([out_im1[slice_ind,:,:]])
                    data_fake1=np.array([recovered_im[slice_ind,:,:]])
                elif order_c[2]=='C':
                    data_x=np.array([in_im1[:,slice_ind,:],in_im2[:,slice_ind,:],recovered_im[:,slice_ind,:]])
                    data_y=np.array([out_im1[:,slice_ind,:]])
                    data_fake1=np.array([recovered_im[:,slice_ind,:]])
                else:
                    data_x=np.array([in_im1[:,:,slice_ind],in_im2[:,:,slice_ind],recovered_im[:,:,slice_ind]])
                    data_y=np.array([out_im1[:,:,slice_ind]])
                    data_fake1=np.array([recovered_im[:,:,slice_ind]])

                # data_x=np.expand_dims(data_x,axis=0)
                # data_y=np.expand_dims(data_y,axis=0)
                # data_fake1=np.expand_dims(data_fake1,axis=0)
                dataset.append({'A': torch.from_numpy(data_x), 'B':torch.from_numpy(data_y), 'fake1':torch.from_numpy(data_fake1), 'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
    
    if opt.dataset_mode == 'vGAN':
        order_c=opt.order
        dataset=[]
        target_path=opt.dataroot+'/'+opt.phase
        patients_target=os.listdir(target_path)

        for ind in range(len(patients_target)):
            target_subject=os.path.join(target_path,patients_target[ind])+'/'

            out_im=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.out+'.nii'))
            #subject based normalization
            out_im=1.0*out_im/out_im.max()

            in_im1=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input1+'.nii'))
            in_im1=1.0*in_im1/in_im1.max()

            in_im2=np.float32(load_nifty_volume_as_array(filename=target_subject+opt.input2+'.nii'))
            in_im2=1.0*in_im2/in_im2.max()

            out_im=(out_im-0.5)*2
            in_im1=(in_im1-0.5)*2
            in_im2=(in_im2-0.5)*2

            data_x=np.array([in_im1,in_im2])
            data_y=np.array([in_im2])
            dataset.append({'A': torch.from_numpy(data_x), 'B':torch.from_numpy(data_y), 'A_paths':opt.dataroot, 'B_paths':opt.dataroot})

    if opt.dataset_mode == 'refineGAN':
        dataset=[]
        target_path=opt.dataroot+'/'+opt.phase
        patients_target=os.listdir(target_path)

        for ind in range(len(patients_target)):
            target_subject=os.path.join(target_path,patients_target[ind])+'/'

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

            
            slice_number=images_fs.shape[0]

            for slice_ind in range(slice_number):
                data_x=np.array([images_fs[ind,:,:],mask_im[ind,:,:]])


                dataset.append({'A': torch.from_numpy(data_x), 'A_paths':opt.dataroot})
    #else:
    #    raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
