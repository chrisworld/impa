# --
# chris debug file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import time

from skimage import io
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
from scipy.ndimage.filters import *


def compute_psnr(img1, img2):
    """
      @param: img1 first input Image
      @param: img2 second input Image

      @return: Peak signal-to-noise ratio between the first and second image
    """
    # mse
    mse = np.sum(np.power((img1 - img2), 2)) / (img1.shape[2] * img1.shape[1] * img1.shape[0])
    return 10 * np.log10( 1 / mse)


def compute_mean(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the mean
      of this window.

      @return: image containing the mean for each pixel
    """
    return


def compute_variance(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (\sigma^2) for each pixel
    """
    return


def compute_a(F, I, m, mu, variance, filter_size, epsilon):
    """
      Compute the intermediate result 'a' as described in the task (equation 4)

      @param: F input image
      @param: I guidance image
      @param: m mean of input image
      @param: mu mean of guidance image
      @param: variance of guidance image
      @param: filter_size
      @param: epsilon smoothing parameter

      @return: image containing a_k for each pixel
    """
    return


def compute_b(m, a, mu):
    """
      Compute the intermediate result 'b' as described in the task (equation 5)

      @param: m mean of input image
      @param: a
      @param: mu mean of guidance image

      @return: image containing b_k for each pixel
    """
    return m - a * mu


def compute_q(mean_a, mean_b, I):
    """
      Compute the final filtered result 'q' as described in the task (equation 6)
      @return: filtered image
    """
    return


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):

    return


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):

    # F is input_img, I is is guidance_img

    # mp is mean of F in wp
    m = compute_mean(input_img, filter_size)

    # compute mean and variance of guidance image
    mu = compute_mean(guidance_img, filter_size)
    variance = compute_variance(guidance_img, filter_size)

    # compute a
    compute_a(input_img, guidance_img, m, mu, variance, filter_size, epsilon)

    # compute b
    compute_b(m, a, mu)

    return


def prepare_imgs(input_filename, downsample_ratio):
    """
      Prepare the images for the guided upsample filtering

      @param: input_filename Filename of the input image
      @param: downsample_ratio ratio between the filter input resolution and the guidance image resolution

      @returns:
        input_img: the input image of the filter
        guidance_img: the guidance image of the filter
        reference_img: the high resolution reference image, this should only be used for calculation of the PSNR and plots for comparison
    """

    # read reference image
    reference_img = io.imread(input_filename)
    print('reference_img: ', reference_img.shape)

    # guidance image to grey-scale
    guidance_img = rgb2gray(reference_img)

    # resize images
    input_img = rescale(reference_img, 1 / downsample_ratio, multichannel=True, mode='reflect', anti_aliasing=True)
    print('scaled: ', input_img.shape)


    # plots
    #
    # plt.figure(1)
    # plt.imshow(guidance_img)
    # plt.imshow(guidance_img, cmap='gray', interpolation='none')
    # plt.show()




    return input_img, guidance_img, reference_img


def plot_result(input_img, guidance_img, filtered_img):
    pass


if __name__ == "__main__":
    start_time = time.time()

    # Set Parameters
    downsample_ratio = 4
    filter_size = 3
    epsilon = 0.1

    # Parse Parameter
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    input_filename = sys.argv[1]

    # Prepare Images
    input_img, guidance_img, initial_img = prepare_imgs(input_filename, downsample_ratio)

    # Perform Guided Upsampling

    # approach (1):
    filtered_img_1 = guided_upsampling(resize(input_img, guidance_img.shape), guidance_img, filter_size, epsilon)

    # approach (2):
    filtered_img_2 = guided_upsampling(input_img, guidance_img, filter_size, epsilon)

    # Calculate PSNR
    psnr_filtered_1 = compute_psnr(filtered_img_1, initial_img)
    psnr_upsampled_1 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

    psnr_filtered_2 = compute_psnr(filtered_img_2, initial_img)
    psnr_upsampled_2 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

    print('Runtime: {} - [Approach 1: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}] [Approach 2: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}]'.format(time.time() - start_time, psnr_filtered_2, psnr_upsampled_2,
                                                                                                                                                           psnr_filtered_1, psnr_upsampled_1))

    # Plot result
    plot_result(input_img, guidance_img, filtered_img_2)
    plot_result(input_img, guidance_img, filtered_img_1)
