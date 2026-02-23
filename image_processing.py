import numpy as np
import scipy.ndimage as nd


def rotate_image(img, angle):
    return nd.rotate(img, -angle)


def shift_image(img, x_shift, y_shift):
    return nd.shift(img, (x_shift, y_shift))


def flip_image_horizontal(img):
    return np.fliplr(img)


def flip_image_vertical(img):
    return np.flipud(img)


def invert_image(img):
    return -img


def log_image(img):
    return np.log(img)


def threshold_image_min(img, min_val):
    img[img < min_val] = min_val
    return img


def threshold_image_max(img, max_val):
    img[img > max_val] = max_val
    return img


def median_filter_image(img, size):
    return nd.median_filter(img, size=size)


def gaussian_filter_image(img, sigma):
    return nd.gaussian_filter(img, sigma=sigma)
