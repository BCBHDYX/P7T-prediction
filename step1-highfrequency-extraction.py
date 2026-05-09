import os
import glob
import numpy as np
import nibabel as nib

# ============================================================
# Configuration
# ============================================================

DataPath = '/home/daiyx/IMAGEN/demo_code/test_data/3T/'

# HF1 output
SavePath_2 = '/home/daiyx/IMAGEN/demo_code/test_data/highfrequency_3th/'

# HF2 output
SavePath_3 = '/home/daiyx/IMAGEN/demo_code/test_data/highfrequency_3thh/'

os.makedirs(SavePath_2, exist_ok=True)
os.makedirs(SavePath_3, exist_ok=True)

# Center low-frequency block size in k-space
# 9 means removing the central 9×9 low-frequency region
MASK_SIZE = 9

# Percentile-based suppression threshold
# 90 means suppressing k-space coefficients whose magnitude is above the 90th percentile
PERCENTILE = 90.0


# ============================================================
# Functions
# ============================================================

def create_highpass_mask(shape, center_size):
    """
    Create a 2D high-pass mask.

    The central low-frequency square block is set to 0,
    and the remaining high-frequency region is set to 1.
    """
    h, w = shape

    center_size = int(min(center_size, h, w))
    if center_size < 1:
        return np.zeros((h, w), dtype=np.float32)

    half = center_size // 2
    cy, cx = h // 2, w // 2

    y1 = cy - half
    y2 = cy - half + center_size
    x1 = cx - half
    x2 = cx - half + center_size

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h, y2)
    x2 = min(w, x2)

    mask_h = np.ones((h, w), dtype=np.float32)
    mask_h[y1:y2, x1:x2] = 0.0

    return mask_h


def percentile_suppression_kspace(F, percentile=90.0):
    """
    Suppress extreme high-magnitude k-space coefficients.

    Coefficients with magnitude above the specified percentile
    are set to zero.

    Parameters
    ----------
    F : np.ndarray
        Complex-valued k-space data after high-pass masking.

    percentile : float
        Percentile threshold. For example, 90 means values above
        the 90th percentile of |F| are suppressed.

    Returns
    -------
    F_out : np.ndarray
        Suppressed k-space data.
    """
    mag = np.abs(F)

    # Use only non-zero coefficients to estimate the threshold.
    # This avoids the masked central zero region dominating the percentile.
    nonzero_mag = mag[mag > 0]

    if nonzero_mag.size == 0:
        return F

    thr = np.percentile(nonzero_mag, percentile)

    F_out = F.copy()
    F_out[mag > thr] = 0

    return F_out


def highpass_2d_with_percentile_suppression(slice2d, center_size=9, percentile=90.0):
    """
    Extract the high-frequency component from a 2D slice using:
    1. k-space high-pass filtering by removing the central low-frequency block;
    2. percentile-based suppression of extreme k-space coefficients;
    3. inverse FFT to obtain the high-frequency magnitude image.
    """
    slice2d = slice2d.astype(np.float32)

    mask_h = create_highpass_mask(slice2d.shape, center_size)

    # 2D FFT and shift low frequency to the center
    F_im = np.fft.fftshift(np.fft.fft2(slice2d))

    # Step 1: remove central low-frequency block
    Y_h = F_im * mask_h

    # Step 2: suppress extreme high-magnitude k-space coefficients
    Y_h = percentile_suppression_kspace(Y_h, percentile=percentile)

    # Step 3: inverse FFT and take magnitude
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(Y_h)))

    return out.astype(np.float32)


def save_nifti(data, reference_img, out_path):
    """
    Save data as a NIfTI image using the affine and header from reference_img.
    """
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)

    out_img = nib.Nifti1Image(data.astype(np.float32), reference_img.affine, header)
    nib.save(out_img, out_path)


# ============================================================
# Main workflow
# ============================================================

# Get all NIfTI files
files = sorted(
    glob.glob(os.path.join(DataPath, '*.nii')) +
    glob.glob(os.path.join(DataPath, '*.nii.gz'))
)

print(f"Found {len(files)} NIfTI files.")

for f in files:
    print(f"\nProcessing: {f}")

    # Load NIfTI file
    img = nib.load(f)
    X = img.get_fdata().astype(np.float32)

    # Replace NaN and Inf values
    X = np.nan_to_num(X)

    if X.ndim != 3:
        print(f"  WARNING: expected 3D image, got shape {X.shape}. Skipped.")
        continue

    sx, sy, sz = X.shape
    print(f"  Shape: {X.shape}")

    # Initialize HF1 and HF2 arrays
    A = np.zeros_like(X, dtype=np.float32)  # HF1
    C = np.zeros_like(X, dtype=np.float32)  # HF2

    # Process each axial slice along the Z dimension
    for j in range(sz):
        slice2d = X[:, :, j]

        # HF1: high-frequency component from the original image
        A_slice = highpass_2d_with_percentile_suppression(
            slice2d,
            center_size=MASK_SIZE,
            percentile=PERCENTILE
        )
        A[:, :, j] = A_slice

        # HF2: second-order high-frequency component from HF1
        C_slice = highpass_2d_with_percentile_suppression(
            A_slice,
            center_size=MASK_SIZE,
            percentile=PERCENTILE
        )
        C[:, :, j] = C_slice

    # Save output files
    base_name = os.path.basename(f)

    out_path_A = os.path.join(SavePath_2, base_name)
    out_path_C = os.path.join(SavePath_3, base_name)

    save_nifti(A, img, out_path_A)
    save_nifti(C, img, out_path_C)

    print(f"  Saved HF1 to: {out_path_A}")
    print(f"  Saved HF2 to: {out_path_C}")

print("\nAll done.")
