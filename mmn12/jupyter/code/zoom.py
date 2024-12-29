import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import cv_utils

def zoom_grayscale_image_fourier(image, zoom_factor):
    # Step 1: Apply Fourier Transform
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    
    # Step 2: Zero-pad in frequency domain
    rows, cols = image.shape
    padded_rows = int(rows * zoom_factor)
    padded_cols = int(cols * zoom_factor)
    
    pad_row_start = (padded_rows - rows) // 2
    pad_col_start = (padded_cols - cols) // 2
    
    # Create a zero-padded frequency domain
    padded_f_shifted = np.zeros((padded_rows, padded_cols), dtype=complex)
    # Apply the existing frequencies
    padded_f_shifted[pad_row_start:pad_row_start + rows, pad_col_start:pad_col_start + cols] = f_shifted
    
    # Step 3: Inverse Fourier Transform
    f_ishifted = ifftshift(padded_f_shifted)
    zoomed_image = np.abs(ifft2(f_ishifted)) # extract magnitudes
    
    # Step 4: Normalizing the resulting image
    zoomed_image -= zoomed_image.min()
    zoomed_image /= zoomed_image.max()
    zoomed_image *= 255
    
    return zoomed_image.astype(np.uint8)