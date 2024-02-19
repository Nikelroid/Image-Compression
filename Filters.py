import numpy as np
from Essentials import get_gaussian_kernel,normalize_binary,normalize

def highpass(sigma, image_int):
    h = image_int.shape[0]
    w = image_int.shape[1]
    gauss_kernel = get_gaussian_kernel(np.max(image_int.shape) + 1,2, sigma)
    result = normalize_binary(gauss_kernel)
    result = 1 - result
    dif = int(np.abs(h - w) / 2)
    if h > w:
        result = result[:h, dif:dif + w]
    elif w > h:
        result = result[dif:dif + h, :w]
    else:
        result = result[:h - 1, :w - 1]

    return result,normalize(result)

def lowpass(sigma, image_int):
    h = image_int.shape[0]
    w = image_int.shape[1]
    gauss_kernel = get_gaussian_kernel(np.max(image_int.shape) + 1,2, sigma)
    result = normalize_binary(gauss_kernel)
    dif = int(np.abs(h - w) / 2)
    if h > w:
        result = result[:h, dif:dif + w]
    elif w > h:
        result = result[dif:dif + h, :w]
    else:
        result = result[:h - 1, :w - 1]

    return result,normalize(result)

def uv_main_transform(img):
    # gets F and returns 4𝜋2(𝑢2+𝑣2) 𝐹
    reduced_uv_fourier_image = np.zeros_like(img)
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, h):
        for j in range(0, w):
            reduced_uv_fourier_image[i, j] = (((i - (h / 2)) ** 2) + ((j - (w / 2)) ** 2)) * img[i, j]
    return reduced_uv_fourier_image * (4 * (np.pi ** 2))