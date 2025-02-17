from .base_options_AD import BaseOptions
class TestOptions(BaseOptions):
    def initialize(self, parser):  
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--img_path", type=str, default='/home/daiyx/IMAGEN/demo_code/test_data')     ####### Attention: Data path for the test set
        parser.add_argument("--img_3t_path", type=str, default='/3T/')                                    ####### Attention: Source 3T Data
        parser.add_argument("--img_3thh_path", type=str, default='/highfrequency_3thh/')                  ####### Attention: 2nd highfrequency
        parser.add_argument("--img_7t_like_path", type=str, default='/P7T/') 							####### Attention: P7T data
        parser.add_argument("--img_fusion_like_path", type=str, default='/P7T-fusion/') 					######## not important
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='160', help='which epoch to load? set to latest to use latest cached model') ######## Here are the model parameters I trained and saved
        parser.add_argument("--stride_layer", type=int, nargs=1, default=1, help="Stride size in z direction")
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser
