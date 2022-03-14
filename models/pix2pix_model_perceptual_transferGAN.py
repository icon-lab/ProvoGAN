import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from torchvision import models

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, 'constant', self.gpu_ids,opt.down_samp)

        
        #self.vgg=VGG16().cuda()
       	
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netG_old = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG_old, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt.down_samp)
            self.load_network_old(self.netG_old, 'G', opt.which_epoch)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        
        if self.isTrain:
            networks.print_network(self.netG_old)
            networks.print_network(self.netD)
            #weight transfer part

            print(self.netG.model[26].weight.shape)
            print(self.netG_old.model[26].weight.shape)
            print(self.netG.model[26].weight.data[0,0,3,:,:])
            print(self.netG_old.model[26].weight.data[0,0,:,:])        
            self.netG.model[1].weight.data[:,:,3,:,:]=self.netG_old.model[1].weight.data[:,:,:,:]
            self.netG.model[4].weight.data[:,:,1,:,:]=self.netG_old.model[4].weight.data[:,:,:,:]
            self.netG.model[7].weight.data[:,:,1,:,:]=self.netG_old.model[7].weight.data[:,:,:,:]
            self.netG.model[10].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[10].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[10].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[10].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[11].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[11].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[11].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[11].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[12].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[12].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[12].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[12].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[13].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[13].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[13].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[13].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[14].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[14].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[14].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[14].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[15].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[15].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[15].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[15].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[16].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[16].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[16].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[16].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[17].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[17].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[17].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[17].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[18].conv_block[1].weight.data[:,:,1,:,:]=self.netG_old.model[18].conv_block[1].weight.data[:,:,:,:]
            self.netG.model[18].conv_block[6].weight.data[:,:,1,:,:]=self.netG_old.model[18].conv_block[6].weight.data[:,:,:,:]
            self.netG.model[19].weight.data[:,:,1,:,:]=self.netG_old.model[19].weight.data[:,:,:,:]
            self.netG.model[22].weight.data[:,:,1,:,:]=self.netG_old.model[22].weight.data[:,:,:,:]
            self.netG.model[26].weight.data[:,:,3,:,:]=self.netG_old.model[26].weight.data[:,:,:,:]
            print(self.netG.model[26].weight.data[0,0,3,:,:])
        print('-----------------------------------------------')
        
        
        

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
       
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        mid_slice=int(self.real_A.shape[2]/2)
        real_A = util.tensor2im(self.real_A[:,:,mid_slice,:,:].data)
        fake_B = util.tensor2im(self.fake_B[:,:,mid_slice,:,:].data)
        real_B = util.tensor2im(self.real_B[:,:,mid_slice,:,:].data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

#VGG        
class VGG_OUTPUT(object):

    def __init__(self,relu2_2):
        self.__dict__ = locals()


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
#        for x in range(9, 16):
#            self.slice3.add_module(str(x), vgg_pretrained_features[x])
#        for x in range(16, 23):
#            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h = self.slice1(X)
        #h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
#        h = self.slice3(h)
#        h_relu3_3 = h
#        h = self.slice4(h)
#        h_relu4_3 = h
        return VGG_OUTPUT(h_relu2_2)

