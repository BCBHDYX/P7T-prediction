import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    if gpu_ids != '-1':
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_R(ngf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = RNetwork(ngf, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_G(ngf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = GNetwork(ngf, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
            
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# ================================= R-Network ================================= #
class RNetwork(nn.Module):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm3d):
        super(RNetwork, self).__init__()
        # construct unet structure
        unet_block = RSubmodule1(ngf, norm_layer)
        unet_block = RSubmodule2(ngf, norm_layer, unet_block)
        unet_block = RSubmodule3(ngf, norm_layer, unet_block)
        unet_block = RSubmodule4(ngf, norm_layer, unet_block)
        unet_block = RSubmodule5(ngf, norm_layer, unet_block)
        unet_block = RSubmodule6(ngf, norm_layer, unet_block)
        
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class RSubmodule1(nn.Module):
    def __init__(self, ngf, norm_layer):
        super(RSubmodule1, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 16, ngf * 32, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 32),
            nn.ReLU(True),
            nn.Conv3d(ngf * 32, ngf * 32, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 32),
            nn.ReLU(True)
        )
        self.part2 = nn.Sequential(
            # nn.ConvTranspose3d(ngf * 32, ngf * 16, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 32, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16)
        )

    def forward(self, x):
        f12 = self.part1(x)
        return torch.cat([x, self.part2(f12)], 1), f12
    
class RSubmodule2(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(RSubmodule2, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 8, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True),
            nn.Conv3d(ngf * 16, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 32, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 16, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 16, ngf * 8, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 16, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8)
        )

    def forward(self, x):
        f11 = self.part1(x)
        submodule_output, f12 = self.part2(f11)
        c2 = self.part3(submodule_output)
        return torch.cat([x, self.part4(c2)], 1), f11, f12, c2 

class RSubmodule3(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(RSubmodule3, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 4, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 16, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 8, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4)
        )
    
    def forward(self, x):
        f2 = self.part1(x)
        submodule_output, f11, f12, c2 = self.part2(f2)
        c3 = self.part3(submodule_output)
        return torch.cat([x, self.part4(c3)], 1), f2, f11, f12, c2, c3

class RSubmodule4(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(RSubmodule4, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 2, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            nn.Conv3d(ngf * 4, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 4, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 4, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2)
        )
    
    def forward(self, x):
        f3 = self.part1(x)
        submodule_output, f2, f11, f12, c2, c3 = self.part2(f3)
        c4 = self.part3(submodule_output)
        return torch.cat([x, self.part4(c4)], 1), f3, f2, f11, f12, c2, c3, c4

class RSubmodule5(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(RSubmodule5, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.Conv3d(ngf * 2, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 4, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 2, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf)
        )
    
    def forward(self, x):
        f4 = self.part1(x)
        submodule_output, f3, f2, f11, f12, c2, c3, c4 = self.part2(f4)
        c5 = self.part3(submodule_output)
        return torch.cat([x, self.part4(c5)], 1), f4, f3, f2, f11, f12, c2, c3, c4, c5

class RSubmodule6(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(RSubmodule6, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv3d(1, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, 1, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.Tanh()
        )

    def forward(self, x):
        f5 = self.part1(x)
        submodule_output, f4, f3, f2, f11, f12, c2, c3, c4, c5 = self.part2(f5)
        c6 = self.part3(submodule_output)
        return self.part4(c6), f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6


# ================================= G-Network ================================= #
class GNetwork(nn.Module):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm3d):
        super(GNetwork, self).__init__()
        # construct unet structure
        unet_block = GSubmodule1(ngf, norm_layer)
        unet_block = GSubmodule2(ngf, norm_layer, unet_block)
        unet_block = GSubmodule3(ngf, norm_layer, unet_block)
        unet_block = GSubmodule4(ngf, norm_layer, unet_block)
        unet_block = GSubmodule5(ngf, norm_layer, unet_block)
        unet_block = GSubmodule6(ngf, norm_layer, unet_block)
        
        self.model = unet_block

    def forward(self, input, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6):
        return self.model(input, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6)


class GSubmodule1(nn.Module):
    def __init__(self, ngf, norm_layer):
        super(GSubmodule1, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 16, ngf * 32, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 32),
            nn.ReLU(True),
            nn.Conv3d(ngf * 32, ngf * 32, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 32),
            nn.ReLU(True)
        )
        self.part2 = nn.Sequential(
            # nn.ConvTranspose3d(ngf * 32, ngf * 16, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 32, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16)
        )

    def forward(self, x, f11, f12):
        x = x * f11 + x
        y = self.part1(x)
        y = y * f12 + y
        y = self.part2(y)
        return torch.cat([x, y], 1)
    
class GSubmodule2(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(GSubmodule2, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 8, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True),
            nn.Conv3d(ngf * 16, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 32, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 32, ngf * 16, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 16),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 16, ngf * 8, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 16, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8)
        )

    def forward(self, x, f2, f11, f12, c2):
        x = x * f2 + x
        y = self.part1(x)
        y = self.part2(y, f11, f12)
        y = self.part3(y)
        y = self.part4(torch.cat([c2, y], 1))
        return torch.cat([x, y], 1)

class GSubmodule3(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(GSubmodule3, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 4, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 16, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 16, ngf * 8, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 8, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4)
        )
    
    def forward(self, x, f3, f2, f11, f12, c2, c3):
        x = x * f3 + x
        y = self.part1(x)
        y = self.part2(y, f2, f11, f12, c2)
        y = self.part3(y)
        y = self.part4(torch.cat([c3, y], 1))
        return torch.cat([x, y], 1)

class GSubmodule4(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(GSubmodule4, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf * 2, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            nn.Conv3d(ngf * 4, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * 4, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 4, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2)
        )
    
    def forward(self, x, f4, f3, f2, f11, f12, c2, c3, c4):
        x = x * f4 + x
        y = self.part1(x)
        y = self.part2(y, f3, f2, f11, f12, c2, c3)
        y = self.part3(y)
        y = self.part4(torch.cat([c4, y], 1))
        return torch.cat([x, y], 1)

class GSubmodule5(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(GSubmodule5, self).__init__()
        self.part1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Conv3d(ngf, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.Conv3d(ngf * 2, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 4, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 4, ngf * 2, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=(2,2,1), stride=(2,2,1)),
            nn.Upsample(scale_factor=(2.0, 2.0, 1.0), mode='trilinear', align_corners=True),
            nn.Conv3d(ngf * 2, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf)
        )
    
    def forward(self, x, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5):
        x = x * f5 + x
        y = self.part1(x)
        y = self.part2(y, f4, f3, f2, f11, f12, c2, c3, c4)
        y = self.part3(y)
        y = self.part4(torch.cat([c5, y], 1))
        return torch.cat([x, y], 1)

class GSubmodule6(nn.Module):
    def __init__(self, ngf, norm_layer, submodule):
        super(GSubmodule6, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv3d(1, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        self.part2 = submodule
        self.part3 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        self.part4 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf, kernel_size=(3,3,1), padding=(1,1,0)),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, 1, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.Tanh()
        )

    def forward(self, x, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5, c6):
        y = self.part1(x)
        y = self.part2(y, f5, f4, f3, f2, f11, f12, c2, c3, c4, c5)
        y = self.part3(y)
        y = self.part4(torch.cat([c6, y], 1))
        return y
        

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # kw = 4
        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
      
        if use_sigmoid:
            sequence += [nn.Sigmoid()]   

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

