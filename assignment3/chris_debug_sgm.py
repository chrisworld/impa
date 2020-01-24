import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows


def calc_wta(cv):
    """
    The winner takes all algortihm
    """

    # init
    H, W, D = cv.shape
    wta = np.zeros((H, W))

    # run though each pixel
    for y in np.arange(H):

        for x in np.arange(W):

            # arg min of all disparities
            wta[y, x] = np.argmin(cv[y, x], axis=0)
            #print("wta: [{}, {}] idx: [{}]".format(y, x, wta[y, x]))

    # plot result
    plt.figure()
    plt.imshow(wta, cmap='plasma')
    plt.colorbar()
    plt.show()

    return wta


def dp_chain(g, f, m):
    '''
        g: unary costs with shape (H,W,D)
        f: pairwise costs with shape (H,W,D,D)
        m: messages with shape (H,W,D)
    '''
    # TODO
    return


def compute_cost_volume_sad(left_image, right_image, D, radius):
    """
    Sum of Absolute Differences (SAD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """

    # get dimensions
    H, W = left_image.shape

    # get disparity images I(x-d, y)
    Imgs_d = np.zeros((D, H, W))

    # calculate disparity image
    for d in np.arange(D):

        # move right image to right x-direction
        Imgs_d[d] = np.roll(right_image, d, axis=1)

        #plt.figure()
        #plt.imshow(Imgs_d[:, :, d], cmap='gray')
        #plt.tight_layout()
        #plt.show()

    # hop size in pixel
    hop = 1

    #I0_patches = view_as_windows(left_image, (radius, radius), step=hop)
    #print("I0_patches", I0_patches.shape)

    # copy left image
    I0 = np.copy(left_image)

    # init cv
    cv = np.zeros((H, W, D))

    for d, I_d in enumerate(Imgs_d):

        #I_d_patches = view_as_windows(I_d, (radius, radius), step=hop)
        #print("I_d_patches: ", I_d_patches.shape)

        # similarity measure sad

        # compute cost volume
        cv[:, :, d] = np.abs(I0 - I_d)

    return cv


def compute_cost_volume_ssd(left_image, right_image, D, radius):
    """
    Sum of Squared Differences (SSD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    # TODO
    return


def compute_cost_volume_ncc(left_image, right_image, D, radius):
    """
    Normalized Cross Correlation (NCC) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    # TODO
    return


def get_pairwise_costs(H, W, D, weights=None):
    """
    :param H: height of input image
    :param W: width of input image
    :param D: maximal disparity
    :param weights: edge-dependent weights (necessary to implement the bonus task)
    :return: pairwise_costs of shape (H,W,D,D)
             Note: If weight=None, then each spatial position gets exactly the same pairwise costs.
             In this case the array of shape (D,D) can be broadcasted to (H,W,D,D) by using np.broadcast_to(..).
    """
    # TODO
    return


def compute_sgm(cv, f):
    """
    Compute the SGM
    :param cv: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    # TODO
    return


def plot_imgs(im0g, im1g):
    """
    just plot the images
    """
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()
    plt.show()


def main():

    # path to images
    img_path = '../ignore/ass3_data/'

    # Load input images
    im0 = imread(img_path + "Adirondack_left.png")
    im1 = imread(img_path + "Adirondack_right.png")

    # convert to gray values
    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    # plot image data
    print("image1 shape: ", im0g.shape)
    print("image2 shape: ", im1g.shape)
    #plot_imgs(im0g, im1g)

    # maximal disparity
    D_max = 64

    # filter radius
    filter_radius = 5

    # Use either SAD, NCC or SSD to compute the cost volume
    cv = compute_cost_volume_sad(im0g, im1g, D_max, filter_radius)
    #cv = compute_cost_volume_ssd(im0g, im1g, D_max, filter_radius)
    #cv = compute_cost_volume_ncc(im0g, im1g, D_max, filter_radius)


    # compute wta algorithm
    calc_wta(cv)

    # Compute pairwise costs
    H, W, D = cv.shape
    f = get_pairwise_costs(H, W, D)

    # Compute SGM
    disp = compute_sgm(cv, f)

    # Plot result
    plt.figure()
    plt.imshow(disp)
    plt.show()


if __name__== "__main__":
    main()
