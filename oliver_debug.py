import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import time

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage.filters import *


def compute_psnr(img1, img2):
    """
      @param: img1 first input Image
      @param: img2 second input Image

      @return: Peak signal-to-noise ratio between the first and second image
    """
    # compute difference between images, pointwise square them and sum
    # them up 
    MSE = 1./np.size(img1) * np.sum(np.square(img1 - img2))
    PSNR = 10*np.log10(1./MSE)
    return PSNR


def compute_mean(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the mean
      of this window.

      @return: image containing the mean for each pixel
    """
    return uniform_filter(image, filter_size, mode="reflect")


def compute_variance(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (\sigma^2) for each pixel
    """

    #return uniform_filter(np.square(image), filter_size, mode="reflect") - np.square(uniform_filter(image, filter_size, mode="reflect"))
    return compute_mean(np.square(image), filter_size)- np.square(uniform_filter(image, filter_size, mode="reflect"))


def compute_a(F, I, m, mu, variance, filter_size, epsilon):
    """
      Compute the intermediate result 'a' as described in the task (equation 4)

      @param: F input image
      @param: I guidance image
      @param: m mean of input image
      @param: mu mean of guidance image
      @param: variance of guidance imag
      @param: filter_size
      @param: epsilon smoothing parameter
@return: image containing a_k for each pixel
    """
    #m_p = uniform_filter(F, filter_size, mode="reflect")
    #mu_p = uniform_filter(I, filter_size, mode="reflect" )
    
    return (uniform_filter(F*I, filter_size, mode="reflect") - (m*mu))/(variance + epsilon)


def compute_b(m, a, mu):
    """
      Compute the intermediate result 'b' as described in the task (equation 5)
kk
      @param: m mean of input image
      @param: a
      @param: mu mean of guidance image

      @return: image containing b_k for each pixel
    """
    return m - a*mu


def compute_q(mean_a, mean_b, I):
    """
      Compute the final filtered result 'q' as described in the task (equation 6)
      @return: filtered image
    """
    return mean_a * I + mean_b


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):

    # scenario 1, input image F = F_H and guided filter 
    # I = G

    F = input_img
    I = guidance_img
    m_p = compute_mean(F, filter_size)
    mu_p = compute_mean(I, filter_size)

    variance = compute_variance(I,filter_size)

    a_p = compute_a(F,I,m_p,mu_p, variance, filter_size, epsilon)
    b_p = compute_b(m_p, a_p, mu_p)

    #a_ = compute_mean(a_p, filter_size)
    #b_ = compute_mean(b_p, filter_size)

    return a_p, b_p


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):
    # test dimension whether it is case one or two 
    # and then call the corresponding function
    # scenario 1: everything stays the same in terms
    # of sizes
    U_q = np.zeros([guidance_img.shape[0],guidance_img.shape[1], input_img.shape[2]])

    for color in range(input_img.shape[2]):
      if (input_img.shape[0:2] == guidance_img.shape[0:2]):
        I = guidance_img
        F = input_img[:,:,color]
        a, b = calculate_guided_image_filter(F, I, filter_size, epsilon)
        a_ = compute_mean(a, filter_size)
        b_ = compute_mean(b, filter_size)
      # calculate coefficients a and b with the low resolution
      # image + low res guidance
      else:
        I = resize(guidance_img, input_img.shape[0:2])
        F = input_img[:,:,color]
        a, b = calculate_guided_image_filter(F, I, filter_size, epsilon)
        a = resize(a, guidance_img.shape)
        b = resize(b, guidance_img.shape)
        a_ = compute_mean(a, filter_size)
        b_ = compute_mean(b, filter_size)
        #a_ = resize(a, guidance_img.shape)
        #b_ = resize(b, guidance_img.shape)
      U_q[:,:,color] = compute_q(a_, b_, guidance_img)
    return np.clip(U_q,0.0,1.0)


def prepare_imgs(input_filename, upsample_ratio):
    """
      Prepare the images for the guided upsample filtering
      @param: input_filename Filename of the input image
      @param: upsample_ratio ratio between the filter input resolution and the guidance image resolution

      @returns:
        input_img: the input image of the filter
        guidance_img: the guidance image of the filter
        reference_img: the high resolution reference image, this should only be used for calculation of the PSNR and plots for comparison
    """
    initial_img = io.imread(input_filename)/255
    #initial_img = resize(initial_img, (initial_img.shape[0]//4, initial_img.shape[1]//4, initial_img.shape[2]), anti_aliasing=True)
    input_img = resize(initial_img, (initial_img.shape[0]// upsample_ratio, initial_img.shape[1] // upsample_ratio, initial_img.shape[2]), anti_aliasing=True)
    guidance_img = rgb2gray(initial_img)

    return input_img, guidance_img, initial_img


def plot_result(input_img, guidance_img, filtered_img):
    inputImag = plt.figure(1)
    ax = plt.Axes(inputImag, [0.,0.,1.,1.])
    ax.set_axis_off()
    inputImag.add_axes(ax)
    plt.imshow(input_img)

    filteredImag = plt.figure(frameon=False)
    ax = plt.Axes(filteredImag, [0.,0.,1.,1.])
    ax.set_axis_off()
    filteredImag.add_axes(ax)
    plt.imshow(filtered_img)
    plt.show()
    
    pass


if __name__ == "__main__":
    start_time = time.time()

    r = 4
    # Set Parameters
    downsample_ratio = 4 # TODO
    filter_size = (2*r+1)# TODO
    epsilon = 0.01 # TODO

    # Parse Parameter
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    input_filename = sys.argv[1]

    # Prepare Images
    input_img, guidance_img, initial_img = prepare_imgs(input_filename, downsample_ratio)

    # Perform Guided Upsampling
    # approach (1):
    pre_filter = time.time()
    filtered_img_1 = guided_upsampling(resize(input_img, guidance_img.shape), guidance_img, filter_size, epsilon)
    elapsed1 = time.time() - pre_filter
  
    # approach (2):
    pre_filter = time.time()
    filtered_img_2 = guided_upsampling(input_img, guidance_img, filter_size, epsilon)
    elapsed2 = time.time() - pre_filter
    #print(title)
    # Calculate PSNR

    psnr_filtered_1 = compute_psnr(filtered_img_1, initial_img)
    psnr_upsampled_1 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)
    title1 = "gitignore/approach1_ds{}_r{}_e{}_elapsed{:.2f}_psnr{:.2f}_psnrUp{:.2f}".format(downsample_ratio, r, int(abs(np.log10(epsilon))), elapsed1, psnr_filtered_1, psnr_upsampled_1).replace(".","_")

    psnr_filtered_2 = compute_psnr(filtered_img_2, initial_img)
    psnr_upsampled_2 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)
    title2 = "gitignore/approach2_ds{}_r{}_e{}_elapsed{:.2f}_psnr{:.2f}_psnrUp{:.2f}".format(downsample_ratio, r, int(abs(np.log10(epsilon))), elapsed2, psnr_filtered_2, psnr_upsampled_2).replace(".","_")
    print('--results \n downsample ratio: {:d}, filter size: {:d}, epsilon: {:.2f} \n Runtime: {} - \n [Approach 1: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}] \n [Approach 2: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}]'
      .format(downsample_ratio, filter_size, epsilon, time.time() - start_time, psnr_filtered_1, psnr_upsampled_1, psnr_filtered_2, psnr_upsampled_2))

    print(title1)
    io.imsave(title1+".png", filtered_img_1)
    print(title2)
    io.imsave(title2+".png", filtered_img_2)
    # Plot result
    plot_result(input_img, guidance_img, filtered_img_2)
    plot_result(input_img, guidance_img, filtered_img_1)
