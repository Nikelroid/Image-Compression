import numpy as np 

def normalize(img):
    return (img - img.min()) * (255 / (img.max() - img.min()))

def normalize_binary(img):
    return (img - img.min()) * (1 / (img.max() - img.min()))

def get_gaussian_kernel(l,coff ,sig):
    limit = int(l / coff)
    raw = np.zeros((l, l))
    for i in range(-limit, limit+1):
        for j in range(-limit, limit+1):
            raw[i + limit, j + limit] = np.exp(-((i ** 2) + (j ** 2)) / (2 * (sig ** 2)))
    return raw

def show_magnitude(img, n):
    return (n * np.log(np.abs(img) + 1)).astype('uint8')

def show_magnitude_fourier(img, n):
    return normalize(n * np.log(np.abs(img) + 1)).astype('uint8')

def clipping(img):
    return (img.clip(0, 255)).astype('uint8')