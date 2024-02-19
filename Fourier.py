import numpy as np 
from Essentials import clipping,show_magnitude_fourier,normalize
from Filters import uv_main_transform

def inverse_fourier(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
    return np.abs(img_back).clip(0, 255).astype(np.int32)

def fourier_transform(img):
    f = np.fft.fft2(img, axes=(0, 1))
    fshift = np.fft.fftshift(f)
    return fshift

def fourier_transform_function(img):
    f = np.fft.fft2(img, axes=(0, 1))
    fshift = np.fft.fftshift(f)
    return fshift

def non_normalized_inverse_fourier_transform_function(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
    return img_back.real

def fourier_uv_sharpening(image_int,k):

    fourier_image = fourier_transform_function(image_int)
    reduced_uv_fourier_image = uv_main_transform(fourier_image)

    inverted_fourier_uv = non_normalized_inverse_fourier_transform_function(reduced_uv_fourier_image)

    fourier_uv_final = clipping(np.add(image_int, np.multiply(k, inverted_fourier_uv)))

    return fourier_uv_final.astype('uint8'), normalize(inverted_fourier_uv),show_magnitude_fourier(reduced_uv_fourier_image, 20)
