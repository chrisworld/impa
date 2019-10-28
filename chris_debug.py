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
    #print("mean: ", image.shape)

    # primitive
    # # init to zeros
    # mu = np.zeros(image.shape)

    # # run through each pixel
    # for row in range(image.shape[0]):

    #   # skip edges
    #   if (row - filter_size) < 0 or (row + filter_size) > image.shape[0]:
    #     continue

    #   for col in range(image.shape[1]):

    #     # skip edges
    #     if (col - filter_size) < 0 or (col + filter_size) > image.shape[1]:
    #       continue

    #     # the pixel means
    #     mu[row, col] = np.mean(image[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1])

    # return mu

    return uniform_filter(image, filter_size, mode='reflect')


def compute_variance(image, filter_size):
    """
      For each pixel in the image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (sigma^2) for each pixel
    """
    #print("var: ", image.shape)

    # # init to zeros
    # var = np.zeros(image.shape)

    # # run through each pixel
    # for row in range(image.shape[0]):

    #   # skip edges
    #   if (row - filter_size) < 0 or (row + filter_size) > image.shape[0]:
    #     continue

    #   for col in range(image.shape[1]):

    #     # skip edges
    #     if (col - filter_size) < 0 or (col + filter_size) > image.shape[1]:
    #       continue

    #     # the pixel variances
    #     var[row, col] = np.var(image[row - filter_size : row + filter_size + 1, col - filter_size : col + filter_size + 1])

    # return var
    return compute_mean(np.power(image, 2), filter_size) - np.power(compute_mean(image, filter_size), 2)


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

    r = (filter_size - 1) // 2

    print("calc a:")
    # run through each pixel
    for row in range(F.shape[0]):

      # skip edges
      if (row - filter_size) < 0 or (row + filter_size) > F.shape[0]:
        continue

      for col in range(F.shape[1]):

        # skip edges
        if (col - filter_size) < 0 or (col + filter_size) > F.shape[1]:
          continue


        #print(F[row - r : row + r + 1, col - r : col + r + 1])
        #print(I[row - r : row + r + 1, col - r : col + r + 1])
        #print(m[row, col])
        #print(mu[row, col])
        #print('var: ', variance[row, col])

        patch = np.sum(F[row - r : row + r + 1, col - r : col + r + 1] * I[row - r : row + r + 1, col - r : col + r + 1] - m[row, col] * mu[row, col])

        # the pixel
        a[row, col] = patch / (variance[row, col] + epsilon) / (filter_size * filter_size)

        #print(patch)
        #print(a[row, col])

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
    return m - a * mu


def compute_q(mean_a, mean_b, I):
    """
      Compute the final filtered result 'q' as described in the task (equation 6)
      @return: filtered image
    """
    print('calc q:')
    return mean_a * I + mean_b


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):

    # F is the input_img, I is the grey guidance_img

    # compute mean and variance of guidance image
    mu = compute_mean(guidance_img, filter_size)

    variance = compute_variance(guidance_img, filter_size)

    # print(variance)
    # plt.figure(1)
    # plt.imshow(variance, cmap='gray', interpolation='none')
    # plt.show()

    # mp is mean of F in wp
    m = compute_mean(input_img, filter_size)

    # compute a
    a = compute_a(input_img, guidance_img, m, mu, variance, filter_size, epsilon)

    # compute b
    b = compute_b(m, a, mu)

    # print(b)
    # plt.figure(1)
    # plt.imshow(a, cmap='gray', interpolation='none')
    # plt.show()

    # compute Uq
    return compute_q(compute_mean(a, filter_size), compute_mean(b, filter_size), guidance_img)
    


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):

    # Init output image
    Uq = np.zeros(input_img.shape)

    # apply the filter for each channel
    for color in range(Uq.shape[2]):

      print("color: ", color)

      # guided filter
      Uq[:, :, color] = np.clip(calculate_guided_image_filter(input_img[:, :, color], guidance_img, filter_size, epsilon), 0, 1.0)

      print("max uq: ", np.max(Uq[:, :, color]))
      print("min uq: ", np.min(Uq[:, :, color]))
      # plt.figure(1)
      # plt.imshow(guidance_img)
      # plt.imshow(Uq[:, :, 0], cmap='gray', interpolation='none')
      # plt.show()

    # plots
    #
    plt.figure(2)

    # plt.subplot(131), plt.imshow(Uq[:, :, 0]), plt.colorbar(fraction=0.035);
    # plt.subplot(132), plt.imshow(Uq[:, :, 1]), plt.colorbar(fraction=0.035);
    # plt.subplot(133), plt.imshow(Uq[:, :, 2]), plt.colorbar(fraction=0.035);

    plt.imshow(Uq)
    #plt.imshow(Uq[:, :, 0], cmap='gray', interpolation='none')
    plt.show()

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

    # filter radius
    r = 1

    # filter window size
    filter_size = 2 * r + 1

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
