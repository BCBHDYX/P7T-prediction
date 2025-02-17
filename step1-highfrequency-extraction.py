import os
import glob
import numpy as np
import nibabel as nib

# Set paths
DataPath = '/home/daiyx/IMAGEN/demo_code/test_data/3T/'
SavePath_2 = '/home/daiyx/IMAGEN/demo_code/test_data/highfrequency_3th/'
SavePath_3 = '/home/daiyx/IMAGEN/demo_code/test_data/highfrequency_3th/'

# Create output directories if they do not exist
if not os.path.exists(SavePath_2):
    os.makedirs(SavePath_2)
if not os.path.exists(SavePath_3):
    os.makedirs(SavePath_3)

# Create masks
Mask_h = np.ones((256,256), dtype=np.float32)
Mask_h[107:148, 107:148] = 0  

Mask_l = np.zeros((256,256), dtype=np.float32)
Mask_l[107:148, 107:148] = 1

# Get all NIfTI files in the specified directory
files = glob.glob(os.path.join(DataPath, '*.nii'))

for f in files:
    # Load NIfTI file
    img = nib.load(f)
    X = img.get_fdata().astype(np.float32)
    # Get image dimensions
    sx, sy, sz = X.shape

    # Initialize result arrays
    A = np.zeros_like(X, dtype=np.float32)
    C = np.zeros_like(X, dtype=np.float32)

    # Replace NaN values with 0
    X = np.nan_to_num(X)

    # Process each slice
    # In MATLAB: for j = 1:256, here we assume the Z-dimension is at least 256
    # Otherwise, use range(min(256, sz)) to adapt to smaller sizes.
    for j in range(min(256, sz)):
        # Perform 2D FFT and shift
        F_im = np.fft.fftshift(np.fft.fft2(X[:,:,j]))

        # Frequency domain masking
        Y_h = F_im * Mask_h
        Y_l = F_im * Mask_l

        # Inverse FFT and take absolute value
        A_slice = np.abs(np.fft.ifft2(np.fft.ifftshift(Y_h)))
        A[:,:,j] = A_slice

        # Repeat processing on the result of A
        F_im_h = np.fft.fftshift(np.fft.fft2(A_slice))
        Y_h_h = F_im_h * Mask_h
        C_slice = np.abs(np.fft.ifft2(np.fft.ifftshift(Y_h_h)))
        C[:,:,j] = C_slice

    # Save the result as NIfTI files
    out_img_A = nib.Nifti1Image(A, img.affine, img.header)
    out_img_C = nib.Nifti1Image(C, img.affine, img.header)

    nib.save(out_img_A, os.path.join(SavePath_2, os.path.basename(f)))
    nib.save(out_img_C, os.path.join(SavePath_3, os.path.basename(f)))