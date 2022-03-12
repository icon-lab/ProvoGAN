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
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt.down_samp)
        # self.vgg=VGG16().cuda()
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        
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
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        #Vgg loss
        # if self.opt.vgg_layer==1:
        #     self.VGG_real=self.vgg(self.real_B.expand([1,3,256,256])).relu1_2
        #     self.VGG_fake=self.vgg(self.fake_B.expand([1,3,256,256])).relu1_2            
        # elif self.opt.vgg_layer==2:            
        #     self.VGG_real=self.vgg(self.real_B.expand([1,3,256,256])).relu2_2
        #     self.VGG_fake=self.vgg(self.fake_B.expand([1,3,256,256])).relu2_2
        # self.VGG_loss=self.criterionL1(self.VGG_fake,self.VGG_real)* self.opt.lambda_vgg
        
        #Vgg loss from github version
        # self.VGG_real=self.vgg(self.real_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        # self.VGG_fake=self.vgg(self.fake_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        # self.VGG_loss=self.criterionL1(self.VGG_fake,self.VGG_real)* self.opt.lambda_vgg

        
        self.loss_G = self.loss_G_L1 #+ self.VGG_loss
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()


        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([#('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            # ('G_VGG', self.VGG_loss.data[0]),
                            #('D_real', self.loss_D_real.data[0]),
                            #('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        #self.save_network(self.netD, 'D', label, self.gpu_ids)

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

