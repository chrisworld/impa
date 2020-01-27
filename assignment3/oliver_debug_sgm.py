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

    # hop size in pixel
    hop = 1

    # copy left image
    I0 = np.copy(left_image)

    # create patches of left image
    I0_patches = view_as_windows(I0, (radius, radius), step=hop)

    # get patch shape
    H_p, W_p, rh, rw = I0_patches.shape

    # init cv
    cv = np.zeros((H_p, W_p, D))
    
    # compare left with right image with different disparities
    for d, Id in enumerate(Imgs_d):

        # get patches
        Id_patches = view_as_windows(Id, (radius, radius), step=hop)

        # compute cost volume with similarity measure sad
        cv[:, :, d] = np.sum(np.sum(np.abs(I0_patches - Id_patches), axis=2), axis=2)

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

    # get dimensions
    H, W = left_image.shape

    # get disparity images I(x-d, y)
    Imgs_d = np.zeros((D, H, W))

    # calculate disparity image
    for d in np.arange(D):

        # move right image to right x-direction
        Imgs_d[d] = np.roll(right_image, d, axis=1)

    # hop size in pixel
    hop = 1

    # copy left image
    I0 = np.copy(left_image)

    # create patches of left image
    I0_patches = view_as_windows(I0, (radius, radius), step=hop)

    # get patch shape
    H_p, W_p, rh, rw = I0_patches.shape

    # init cv
    cv = np.zeros((H_p, W_p, D))
    
    # compare left with right image with different disparities
    for d, Id in enumerate(Imgs_d):

        # get patches
        Id_patches = view_as_windows(Id, (radius, radius), step=hop)

        # compute cost volume with similarity measure sad
        cv[:, :, d] = np.sum(np.sum(np.power(I0_patches - Id_patches, 2), axis=2), axis=2)

    return cv


def compute_cost_volume_ncc(left_image, right_image, D, radius):
    """
    Normalized Cross Correlation (NCC) cost volume
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

    # hop size in pixel
    hop = 1

    # copy left image
    I0 = np.copy(left_image)

    # create patches of left image
    I0_patches = view_as_windows(I0, (radius, radius), step=hop)

    # calculate means
    mu_I0_p = np.mean(np.mean(I0_patches, axis=2), axis=2)

    # free of mean
    I0_p_mu = I0_patches - mu_I0_p[:, :, np.newaxis, np.newaxis]
    I0_p_mu_sq = np.power(I0_p_mu, 2)

    # make it free of mean
    H_p, W_p, rh, rw = I0_patches.shape

    # init cv
    cv = np.zeros((H_p, W_p, D))
    
    # compare left with right image with different disparities
    for d, Id in enumerate(Imgs_d):

        # get patches
        Id_patches = view_as_windows(Id, (radius, radius), step=hop)

        # calculate means
        mu_Id_p = np.mean(np.mean(Id_patches, axis=2), axis=2)

        # make it free of mean
        Id_p_mu = Id_patches - mu_Id_p[:, :, np.newaxis, np.newaxis]
        Id_p_mu_sq = np.power(Id_p_mu, 2)

        # calculate cost-volume
        cv[:, :, d] = -1 * np.sum(np.sum(I0_p_mu * Id_p_mu, axis=3), axis=2) / np.sqrt( np.sum(np.sum(I0_p_mu_sq, axis=3), axis=2) * np.sum(np.sum(Id_p_mu_sq, axis=3), axis=2) ) 

    return cv

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
    L1 = 0.1
    L2 = 0.2

    # matrix where each row houses values from
    # 0,1,..,D-1
    indices = np.tile(np.arange(0,D),(D,1))
    pw_cost = np.double(np.abs(indices - indices.T))
    pw_cost[pw_cost==1] = L1
    pw_cost[pw_cost>1] = L2
    
    return np.broadcast_to(pw_cost, (H, W, D, D))


def compute_sgm(cv, f):
    """
    Compute the SGM
    :param cv: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    H,W,D = cv.shape

    msg = np.zeros((H,W,D))

    # rewrite - just need it for one dimension, as it tests it just along 
    # either horizontal or vertical and will be then combined.
   # for i in range(H-1):
   #     for j in range(W-1):
   #         msg[:,i+1] = np.min(cv[i,j] + msg[i,j,:] + f[i,j,:,:])
   # old one dimensional cas: 
   # for i in range(dim-1):
   #        msg[:,i+1] = np.min(unary_cost[:,i] + msg[:,i] + pairwise_cost, axis = 1)
    

    return

def compute_msg(cv, f):
    dim = cv.shape[0]
    msg = np.zeros(cv.shape)

    for i in range(dim-1):
        msg[:,i+1,:] = np.min(cv[:,i,:,np.newaxis] + \
            msg[:,i,:,np.newaxis] + f[:,i,:,:], axis=1)

    return msg


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
