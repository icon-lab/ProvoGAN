def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        #assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_perceptual':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual import Pix2PixModel
        model = Pix2PixModel()        
    elif opt.model == 'pix2pix_perceptual_vGAN':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual_vGAN import Pix2PixModel
        model = Pix2PixModel()   
    elif opt.model == 'pix2pix_perceptual_residual':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual_residual import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_perceptual_att':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual_att import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_perceptual_transferGAN':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual_transferGAN import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_perceptual_refineGAN':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model_perceptual_refineGAN import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_cnn':
        #assert(opt.dataset_mode == 'aligned')
        from .pix2pix_cnn_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
