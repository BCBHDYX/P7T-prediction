import torch
from .base_model import BaseModel
from . import networks3D
from .gan_model import GANModel


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = GANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='_A',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        self.visual_names = ['real_A', 'hf_A', 'fake_B', 'hf_B_like']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['R', 'G' + opt.model_suffix]

        self.netR = networks3D.define_R(opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = networks3D.define_G(opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


    def set_input(self, input):
        self.real_A = input[0].to(self.device) # 1, 1, 288, 288, 4
        self.hf_A = input[1].to(self.device) # 1, 1, 288, 288, 4

    def forward(self):
        self.hf_B_like, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6 = self.netR(self.hf_A)
        self.fake_B = self.netG_A(self.real_A, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6)
