import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                # NiftiDataset.Resample(opt.new_resolution, opt.resample),
                NiftiDataset.Augmentation(),
                # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                # NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
                ]

    train_set = NifitDataSet(opt.train_path, which_direction=opt.which_direction, transforms=trainTransforms, shuffle_labels=False, train=True)
    print('lenght train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)  # Here are then fed to the network with a defined batch size

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for img_3t, img_7t, img_3thh, img_fusion in train_loader:
            train_patch_set = PatchDataSetInterp(img_3t, img_7t, img_3thh, img_fusion, patch_size_z=opt.patch_size[2], stride_z=opt.stride_z)
            train_patch_loader = DataLoader(train_patch_set, batch_size=opt.patch_batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)

            for i, data in enumerate(train_patch_loader):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_steps += opt.patch_batch_size
                epoch_iter += opt.patch_batch_size
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.patch_batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                    model.save_networks('latest')

                iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

