import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D
import numpy as np


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class GANModel(BaseModel):
    def name(self):
        return 'GANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default GAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['D_A', 'G_A', 'R']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['R', 'G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['R', 'G_A']

        # load/define networks
        self.netR = networks3D.define_R(opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = networks3D.define_G(opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionR = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()),
                                                lr=opt.lr_R, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
                        
            self.optimizers = []
            self.optimizers.append(self.optimizer_R)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device) # 1, 288, 288, 4
        self.real_B = input[1 if AtoB else 0].to(self.device) # 1, 288, 288, 4
        self.hf_A = input[2].to(self.device) # 1, 288, 288, 4
        self.hf_B = input[3].to(self.device) # 1, 288, 288, 4

    def forward(self):
        self.hf_B_like, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6 = self.netR(self.hf_A)
        self.fake_B = self.netG_A(self.real_A, f5.detach(), f4.detach(), f3.detach(), f2.detach(), f11.detach(), f12.detach(), c2.detach(), c3.detach(), c4.detach(), c5.detach(), c6.detach())
        if np.isnan(self.fake_B.mean().item()):
            print('------------------------------------------------------------')
            print('one')
            print('生成器的输入是否有nan？', self.real_A.mean().item())
            print('生成器的输出有{}个nan'.format(np.isnan(self.fake_B.detach().cpu()).sum().item()))
            torch.save(self.real_A, './G_input_K.pt')
            torch.save(self.fake_B, './G_output_K.pt')
            torch.save(self.netG_A.state_dict(), './model_G_produce_nan_K.pth')
            print('------------------------------------------------------------')

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        if np.isnan(loss_D_real.item()):
            print('------------------------------------------------------------')
            print('three')
            print('B mean:', real.mean().item(), 'shape:', real.shape, 'max:', real.max().item(), 'min:', real.min().item())
            print('D_A(B) mean:', pred_real.mean().item(), 'shape:', pred_real.shape, 'max:', pred_real.max().item(), 'min:', pred_real.min().item())
            print('------------------------------------------------------------')

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        if np.isnan(loss_D_fake.item()):
            print('------------------------------------------------------------')
            print('four')
            print('fake_B mean:', fake.mean().item(), 'shape:', fake.shape, 'max:', fake.max().item(), 'min:', fake.min().item())
            print('D_A(fake_B) mean:', pred_fake.mean().item(), 'shape:', pred_fake.shape, 'max:', pred_fake.max().item(), 'min:', pred_fake.min().item())
            print('------------------------------------------------------------')

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G_A(self):
        L1_lambda = self.opt.L1_lambda
        # self.loss_G_A_adver = self.criterionGAN(self.netD_A(self.fake_B), True)
        # self.loss_G_A_l1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) + L1_lambda * self.criterionL1(self.fake_B,self.real_B)
        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.loss_G_A_adver + L1_lambda * self.loss_G_A_l1
        if np.isnan(self.loss_G_A.item()):
            print('------------------------------------------------------------')
            print('two')
            print('A mean:', self.real_A.mean().item(), 'shape:', self.real_A.shape, 'max:', self.real_A.max().item(), 'min:', self.real_A.min().item())
            print('fake_B mean:', self.fake_B.mean().item(), 'shape:', self.fake_B.shape, 'max:', self.fake_B.max().item(), 'min:', self.fake_B.min().item())
            print('fake_B中有多少nan:', np.isnan(self.fake_B.detach().cpu()).sum().item())
            print('D_A(fake_B) mean:', self.netD_A(self.fake_B).mean().item(), 'shape:', self.netD_A(self.fake_B).shape, 'max:', self.netD_A(self.fake_B).max().item(), 'min:', self.netD_A(self.fake_B).min().item())
            print('------------------------------------------------------------')
            print(100 / 0)
        
        self.loss_G_A.backward()
    
    def backward_R(self):
        self.loss_R = self.criterionR(self.hf_B_like, self.hf_B)
        self.loss_R.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # R
        self.optimizer_R.zero_grad()
        self.backward_R()
        self.optimizer_R.step()
        # G_A
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_G.zero_grad()
        self.backward_G_A()
        self.optimizer_G.step()
        # D_A
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.optimizer_D.step()
