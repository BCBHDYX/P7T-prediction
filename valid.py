import os
from options.valid_options import ValidOptions
import sys
from utils.NiftiDataset import *
from models import create_model
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import SimpleITK as sitk
import nibabel as nib
from skimage import metrics
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False
def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image

def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches


def inference_interp_valid(model, img_3t_path, img_3thh_path, patch_size_z, stride_z, gpu_ids, batch_size=1):

    # read image file
    img_3t = read_image(img_3t_path)        
    img_3t = Normalization(img_3t)  # set intensity 0-255
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img_3t = castImageFilter.Execute(img_3t)


    img_3thh = read_image(img_3thh_path)
    img_3thh = Normalization(img_3thh)  # set intensity 0-255
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img_3thh = castImageFilter.Execute(img_3thh)


    # create empty label in pair with transformed image (all values are 0)
    img_7t_like_tfm = sitk.Image(img_3t.GetSize(), sitk.sitkFloat32)
    img_7t_like_tfm.SetOrigin(img_3t.GetOrigin())
    img_7t_like_tfm.SetDirection(img_3t.GetDirection())
    img_7t_like_tfm.SetSpacing(img_3t.GetSpacing())

    img_fusion_like_tfm = sitk.Image(img_3t.GetSize(), sitk.sitkFloat32)
    img_fusion_like_tfm.SetOrigin(img_3t.GetOrigin())
    img_fusion_like_tfm.SetDirection(img_3t.GetDirection())
    img_fusion_like_tfm.SetSpacing(img_3t.GetSpacing())

    sample = {'3t': img_3t, '3thh': img_3thh, '7t_like': img_7t_like_tfm, 'fusion_like': img_fusion_like_tfm}
    
    img_3t_tfm, img_3thh_tfm, img_7t_like_tfm, img_fusion_like_tfm = sample['3t'], sample['3thh'], sample['7t_like'], sample['fusion_like']
    
    # convert image to numpy array
    img_3t_np = sitk.GetArrayFromImage(img_3t_tfm)
    img_3thh_np = sitk.GetArrayFromImage(img_3thh_tfm)
    img_7t_like_np = sitk.GetArrayFromImage(img_7t_like_tfm)
    img_7t_like_np = np.asarray(img_7t_like_np, np.float32)
    img_fusion_like_np = sitk.GetArrayFromImage(img_fusion_like_tfm)
    img_fusion_like_np = np.asarray(img_fusion_like_np, np.float32)
    
    # unify numpy and sitk orientation
    img_3t_np = np.transpose(img_3t_np, (2, 1, 0))
    img_3thh_np = np.transpose(img_3thh_np, (2, 1, 0))
    img_7t_like_np = np.transpose(img_7t_like_np, (2, 1, 0))
    img_fusion_like_np = np.transpose(img_fusion_like_np, (2, 1, 0))
    
    '''
    # ----------------- Padding the image if the z dimension still is not even ----------------------
    img_3t_np = np.pad(img_3t_np, ((10, 10), (10, 10), (0, 2)), 'edge')
    img_3thh_np = np.pad(img_3thh_np, ((10, 10), (10, 10), (0, 2)), 'edge')
    img_7t_like_np = np.pad(img_7t_like_np, ((10, 10), (10, 10), (0, 2)), 'edge')
    img_fusion_like_np = np.pad(img_fusion_like_np, ((10, 10), (10, 10), (0, 2)), 'edge')
    '''
    Padding = False
    

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(img_7t_like_np.shape)

    # prepare image batch indices
    knum = int(math.ceil((img_3t_np.shape[2] - patch_size_z) / float(stride_z))) + 1
    
    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for k in range(knum):
        if patch_total % batch_size == 0:
            ijk_patch_indicies_tmp = []

        kstart = k * stride_z
        if kstart + patch_size_z > img_3t_np.shape[2]:  # for last patch
            kstart = img_3t_np.shape[2] - patch_size_z
        kend = kstart + patch_size_z

        ijk_patch_indicies_tmp.append([0, img_3t_np.shape[0], 0, img_3t_np.shape[1], kstart, kend])

        if patch_total % batch_size == 0:
            ijk_patch_indices.append(ijk_patch_indicies_tmp)

        patch_total += 1

    img_3t_batches = prepare_batch(img_3t_np, ijk_patch_indices)
    img_3thh_batches = prepare_batch(img_3thh_np, ijk_patch_indices)

    for i in tqdm(range(len(img_3t_batches))):

        img_3t_batch = img_3t_batches[i]
        img_3t_batch = (img_3t_batch - 127.5) / 127.5
        img_3t_batch = torch.from_numpy(img_3t_batch[np.newaxis, :, :, :])
        if gpu_ids != '-1':
            img_3t_batch = Variable(img_3t_batch.cuda())
        else:
            img_3t_batch = Variable(img_3t_batch)
        
        img_3thh_batch = img_3thh_batches[i]
        img_3thh_batch = (img_3thh_batch - 127.5) / 127.5
        img_3thh_batch = torch.from_numpy(img_3thh_batch[np.newaxis, :, :, :])
        if gpu_ids != '-1':
            img_3thh_batch = Variable(img_3thh_batch.cuda())
        else:
            img_3thh_batch = Variable(img_3thh_batch)

        model.set_input([img_3t_batch, img_3thh_batch])
        model.test()
        pred = model.get_current_visuals() # "pred" is a dictionary

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]

        # 7T-like
        img_7t_like = pred['fake_B']
        img_7t_like = img_7t_like.squeeze().data.cpu().numpy()
        img_7t_like = (img_7t_like * 127.5) + 127.5        
        img_7t_like_np[istart:iend, jstart:jend, kstart:kend] += img_7t_like[:, :, :]

        # 7T-highhigh-like
        img_fusion_like = pred['hf_B_like']
        img_fusion_like = img_fusion_like.squeeze().data.cpu().numpy()
        img_fusion_like = (img_fusion_like * 127.5) + 127.5        
        img_fusion_like_np[istart:iend, jstart:jend, kstart:kend] += img_fusion_like[:, :, :]

        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))

    # eliminate overlapping region using the weighted value
    img_7t_like_np = (np.float32(img_7t_like_np) / np.float32(weight_np) + 0.01)
    img_fusion_like_np = (np.float32(img_fusion_like_np) / np.float32(weight_np) + 0.01)

    
    # removed the 1 pad on z
    if Padding is True:
        img_7t_like_np = img_7t_like_np[10:310, 10:310, 0:250]
        img_fusion_like_np = img_fusion_like_np[10:310, 10:310, 0:250]
    
    return img_7t_like_np, img_fusion_like_np


def eval(image_np, label_path, fusion_mode):
    input = image_np   
    
    if not fusion_mode: # label is 7t
        gs = nib.load(label_path).get_fdata()
    else: # label is fusion of 7th & 7thh
        img_7th = nib.load(label_path[0]).get_fdata()
        img_7th = 255.0 * (img_7th - img_7th.min()) / (img_7th.max() - img_7th.min())
        img_7thh = nib.load(label_path[1]).get_fdata()
        img_7thh = 255.0 * (img_7thh - img_7thh.min()) / (img_7thh.max() - img_7thh.min())
        gs = img_7th + img_7thh
    gs = 255.0 * (gs - gs.min()) / (gs.max() - gs.min())
    
    mse = np.square(input - gs).mean()
    psnr = 10 * np.log10(255.0**2 / mse)
    ssim = metrics.structural_similarity(input, gs, data_range=255.0)
    
    return psnr, ssim


if __name__ == '__main__':

    opt = ValidOptions().parse()

    # evaluation
    epoch_recorder = np.arange(1, opt.model_num + 1)
    psnr_7t_like_recorder = np.zeros(opt.model_num)
    psnr_fusion_like_recorder = np.zeros(opt.model_num)
    ssim_7t_like_recorder = np.zeros(opt.model_num)
    ssim_fusion_like_recorder = np.zeros(opt.model_num)

    for epoch in range(1, opt.model_num + 1):
        # evaluation
        psnr_7t_like = 0
        psnr_fusion_like = 0
        ssim_7t_like = 0
        ssim_fusion_like = 0

        for case_id in range(opt.valid_size):
            opt.which_epoch = str(epoch)
            opt.img_3t_path = opt.valid_root_path + '3t/{}.nii'.format(case_id)
            opt.img_3thh_path = opt.valid_root_path + '3thh/{}.nii'.format(case_id)

            model = create_model(opt)
            model.setup(opt)
            img_7t_like_np, img_fusion_like_np = inference_interp_valid(model, opt.img_3t_path, opt.img_3thh_path, opt.patch_size[2], opt.stride_layer, opt.gpu_ids, 1)

            # evaluation
            # 7T-like
            label_path_7t = opt.valid_root_path + '7t/{}.nii'.format(case_id)
            case_psnr_7t_like, case_ssim_7t_like = eval(img_7t_like_np, label_path_7t, fusion_mode=False)
            psnr_7t_like += case_psnr_7t_like
            ssim_7t_like += case_ssim_7t_like
            # 7T-fusion-like
            label_path_7th = opt.valid_root_path + '7th/{}.nii'.format(case_id)
            label_path_7thh = opt.valid_root_path + '7thh/{}.nii'.format(case_id)
            label_path_fusion = [label_path_7th, label_path_7thh]
            case_psnr_fusion_like, case_ssim_fusion_like = eval(img_fusion_like_np, label_path_fusion, fusion_mode=True)
            psnr_fusion_like += case_psnr_fusion_like
            ssim_fusion_like += case_ssim_fusion_like
        
        # evaluation
        psnr_7t_like /= opt.valid_size
        psnr_7t_like_recorder[epoch-1] = psnr_7t_like
        psnr_fusion_like /= opt.valid_size
        psnr_fusion_like_recorder[epoch-1] = psnr_fusion_like
        ssim_7t_like /= opt.valid_size
        ssim_7t_like_recorder[epoch-1] = ssim_7t_like
        ssim_fusion_like /= opt.valid_size
        ssim_fusion_like_recorder[epoch-1] = ssim_fusion_like
    
    # evaluation
    plt.figure(1)
    plt.title('PSNR of {} epochs'.format(opt.model_num))
    plt.plot(epoch_recorder, psnr_7t_like_recorder, label='7T-like')
    plt.plot(epoch_recorder, psnr_fusion_like_recorder, label='7T-fusion-like')
    plt.legend()
    plt.savefig('PSNR.png')
    plt.figure(2)
    plt.title('SSIM of {} epochs'.format(opt.model_num))
    plt.plot(epoch_recorder, ssim_7t_like_recorder, label='7T-like')
    plt.plot(epoch_recorder, ssim_fusion_like_recorder, label='7T-fusion-like')
    plt.legend()
    plt.savefig('SSIM.png')

    # evaluation
    print('psnr_7t_like: max={:.4f}, epoch={}'.format(psnr_7t_like_recorder.max(), epoch_recorder[psnr_7t_like_recorder.argmax()]))
    print('psnr_fusion_like: max={:.4f}, epoch={}'.format(psnr_fusion_like_recorder.max(), epoch_recorder[psnr_fusion_like_recorder.argmax()]))
    print('ssim_7t_like: max={:.4f}, epoch={}'.format(ssim_7t_like_recorder.max(), epoch_recorder[ssim_7t_like_recorder.argmax()]))
    print('ssim_fusion_like: max={:.4f}, epoch={}'.format(ssim_fusion_like_recorder.max(), epoch_recorder[ssim_fusion_like_recorder.argmax()]))