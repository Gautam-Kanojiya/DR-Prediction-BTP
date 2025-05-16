# preprocessing.py - Preprocess retinal images for QNN

import cv2
import numpy as np
import pywt
import tensorflow as tf

def preprocess_image(img):
    """
    Complete preprocessing pipeline for QNN:
    - Grayscale conversion
    - Histogram equalization
    - Discrete Wavelet Transform (DWT)
    - Gaussian & Gabor filtering
    - K-means segmentation
    - Resize to 4x4 & normalize
    """
    new_row, new_col = 2848, 4288
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)
    resized = tf.image.resize(equ[..., None], (new_row, new_col)).numpy().squeeze()

    # DWT
    coeffs = pywt.dwt2(resized, 'haar')
    equ2 = pywt.idwt2(coeffs, 'haar')

    # Gabor filters
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta, 12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)

    def apply_filters(im, kernels):
        images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
        return np.max(images, 0)

    equ3 = apply_filters(equ2, filters)

    # K-means clustering
    Z = equ3.reshape((-1, 4)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res2 = center[label.flatten()].reshape(equ3.shape)

    # Final resizing and normalization
    final = tf.image.resize(res2[..., None], (4, 4)).numpy().squeeze()
    final = final / 255.0
    return final