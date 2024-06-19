import numpy as np
import cv2
import math
# old version of scikit-image: 
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
# from skimage.measure import compare_mse as mse

# new version of scikit-image：
from skimage.metrics import structural_similarity # ssim
from skimage.metrics import peak_signal_noise_ratio # psnr
from skimage.metrics import mean_squared_error # mse
'''
# =======================================
# metric, PSNR and SSIM
# =======================================
'''


# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    '''
    :param img1: [H,W,C], uint8, 0-255
    :param img2:
    :param border:
    :return:
    '''
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_batch(gt_imgs, pred_imgs, border=0, use_sk = True):
    '''
    :param gt_imgs: [N,H,W,C], uint8, 0-255
    :param pred_imgs:[N,H,W,C], uint8, 0-255
    :param border:
    :param use_sk: doesn't matter, same result, similar speed
    :return:
    '''
    if not gt_imgs.shape == gt_imgs.shape:
        raise ValueError('Input images must have the same dimensions.')
    N,h, w = gt_imgs.shape[:3]

    if use_sk: # 16.753364324569702 s
        psnr = 0
        for i in range(N):
            # ssim += structural_similarity(gt_imgs[i],pred_imgs[i],data_range=255, multichannel=True)
            psnr += peak_signal_noise_ratio(gt_imgs[i], pred_imgs[i], data_range=255)
        psnr = psnr/N
    else: # 18.978911876678467 s
        gt_imgs = gt_imgs[:, border:h - border, border:w - border]
        pred_imgs = pred_imgs[:, border:h - border, border:w - border]

        gt_imgs = gt_imgs.astype(np.float64)
        pred_imgs = pred_imgs.astype(np.float64)

        mse = np.mean((gt_imgs - pred_imgs) ** 2, axis=(1, 2, 3))
        psnr = np.where(mse == 0, float('inf'), 20 * np.log10(255.0 / np.sqrt(mse)))
    return psnr

# ----------
# SSIM
# ----------
def calculate_ssim_batch(gt_imgs, pred_imgs, border=0, use_sk = True):
    '''
    :param gt_imgs: [N,H,W,C], uint8, 0-255
    :param pred_imgs:[N,H,W,C], uint8, 0-255
    :param border:
    :param use_sk: set to True, much faster
    :return:
    '''
    if not gt_imgs.shape == gt_imgs.shape:
        raise ValueError('Input images must have the same dimensions.')
    N,h, w = gt_imgs.shape[:3]


    ssim = 0
    for i in range(N):
        if use_sk: #
            # print('ssim',structural_similarity(gt_imgs[i],pred_imgs[i],data_range=255,channel_axis=2))
            ssim += structural_similarity(gt_imgs[i],pred_imgs[i],data_range=255,channel_axis=2)
        else: #
            # print('ssim',calculate_ssim(gt_imgs[i], pred_imgs[i],border=border))
            ssim += calculate_ssim(gt_imgs[i], pred_imgs[i],border=border)
    ssim = ssim/N

    return ssim

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255], [H,W,C], uint8,
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()