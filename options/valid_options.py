from .base_options import BaseOptions


class ValidOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)        
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument("--stride_layer", type=int, nargs=1, default=1, help="Stride size in z direction")
        parser.add_argument("--model_num", type=int, nargs=1, default=200, help="Num of models to be estimated")
        parser.add_argument("--valid_size", type=int, nargs=1, default=2, help="Size of validation set")
        parser.add_argument("--valid_root_path", type=str, default='/home3/HWGroup/daiyx/German_data/final_train/')
        parser.add_argument("--img_3t_path", type=str, default='/home3/HWGroup/daiyx/German_data/final_train/0.nii')
        parser.add_argument("--img_3thh_path", type=str, default='/home3/HWGroup/daiyx/German_data/Cross-validation/a_c_gd__gl_go/valid/3thh/0.nii')
        #parser.add_argument('--which_epoch', type=str, default='1', help='which epoch to load? set to latest to use latest cached model')
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser

