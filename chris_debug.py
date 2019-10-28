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
      For each pixel in the image consider a window of size
      filter_size x filter_size around the pixel and compute the mean
      of this window.

      @return: image containing the mean for each pixel
    """
    print("mean: ", image.shape)

    # init to zeros
    mu = np.zeros(image.shape)

    # run through each pixel
    for row in range(image.shape[0]):

      # skip edges
      if (row - filter_size) < 0 or (row + filter_size) > image.shape[0]:
        continue

      for col in range(image.shape[1]):

        # skip edges
        if (col - filter_size) < 0 or (col + filter_size) > image.shape[1]:
          continue

        # the pixel means
        mu[row, col] = np.mean(image[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1])

    return mu


def compute_variance(image, filter_size):
    """
      For each pixel in the image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (sigma^2) for each pixel
    """
    print("var: ", image.shape)

    # init to zeros
    var = np.zeros(image.shape)

    # run through each pixel
    for row in range(image.shape[0]):

      # skip edges
      if (row - filter_size) < 0 or (row + filter_size) > image.shape[0]:
        continue

      for col in range(image.shape[1]):

        # skip edges
        if (col - filter_size) < 0 or (col + filter_size) > image.shape[1]:
          continue

        # the pixel variances
        var[row, col] = np.var(image[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1])


    return var


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

    # init to zeros
    a = np.zeros(F.shape)

    print("calc a")
    # run through each pixel
    for row in range(F.shape[0]):

      # skip edges
      if (row - filter_size) < 0 or (row + filter_size) > F.shape[0]:
        continue

      for col in range(F.shape[1]):

        # skip edges
        if (col - filter_size) < 0 or (col + filter_size) > F.shape[1]:
          continue

        # the pixel
        a[row, col] = 1 / (filter_size * filter_size) * np.sum(
        F[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1] *
        I[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1] -
        m[row, col] * mu[row, col]) / (variance[row, col] + epsilon)

    return a


def compute_b(m, a, mu):
    """
      Compute the intermediate result 'b' as described in the task (equation 5)

      @param: m mean of input image
      @param: a
      @param: mu mean of guidance image

      @return: image containing b_k for each pixel
    """
    print('calc b:')
    #print('m: ', m)
    #print('a: ', a)
    #print('mu: ', mu)
    return m - a * mu


def compute_q(mean_a, mean_b, I):
    """
      Compute the final filtered result 'q' as described in the task (equation 6)
      @return: filtered image
    """
    print('calc q:')

    return mean_a * I + mean_b


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):

    return


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):

    # F is the input_img, I is the grey guidance_img

    # compute mean and variance of guidance image
    mu = compute_mean(guidance_img, filter_size)
    variance = compute_variance(guidance_img, filter_size)

    # apply the filter for each channel

    # mp is mean of F in wp
    m = compute_mean(input_img[:, :, 1], filter_size)

    # compute a
    a = compute_a(input_img[:, :, 1], guidance_img, m, mu, variance, filter_size, epsilon)

    # compute b
    b = compute_b(m, a, mu)

    # compute Uq
    Uq = compute_q(np.mean(a), np.mean(b), guidance_img)

    return Uq


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
    filter_size = 1
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
