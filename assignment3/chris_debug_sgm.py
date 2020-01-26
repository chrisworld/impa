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

    return wta


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

    # hyper params
    L1 = 0.1
    L2 = 0.2

    # no weights
    if weights == None:

        # all the same pairwise costs
        costs = np.ones((D, D)) * L2

        # run through disparity map
        for i, zi in enumerate(range(D)):
            for j, zj in enumerate(range(D)):

                # same label
                if zi == zj:

                    # zeros cost on same level
                    costs[i, j] = 0;

                # one label diff
                elif np.abs(zi - zj) == 1:

                    # cost for one level diff
                    costs[i, j] = L1;

        # broadcast the costs as they are the same for each nodes
        return np.broadcast_to(costs, (H, W, D, D))

    return 0


def binary_costs(z, f, L1=0.1, L2=0.2):
    """
    binary cost between two nodes
    z: labels [ch x nodes]
    f: pairwise cost [ch x nodes x disparities x disparities]
    """

    fb = np.copy(f)



    # else
    return 


def dp_chain(g, f, m):
    """
        g: unary costs with shape (H,W,D)
        f: pairwise costs with shape (H,W,D,D)
        m: messages with shape (H,W,D)
    """
    
    # get shape 
    H, W, D = m.shape

    # g as channel: [ch x nodes x disparities]
    for i in range(W - 1):

        # parallel computation
        m[:, i + 1] = np.min(m[:, i, :, np.newaxis] + f[:, i] + g[:, i, :, np.newaxis], axis=1)

    return m


def compute_sgm(cv, f):
    """
    Compute the SGM
    :param cv: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """

    # get shape
    H, W, D = cv.shape

    # init disparity map
    disp_map = np.zeros((H, W))

    # for all four directions
    directions = ['L', 'R', 'U', 'D']

    # init messages
    m = np.zeros((len(directions), H, W, D))

    # calculate disparity map
    z = calc_wta(cv)

    # compute paiwise costs
    for a, direction in enumerate(directions):

        print("message direction: ", direction)

        # horizontal backward messages
        if direction == 'L':

            # run the chain
            m[a] = dp_chain(cv[:, ::-1], f, m[a])

        # horizontal forward messages
        elif direction == 'R':

            # run the chain
            m[a] = dp_chain(cv, f, m[a])

        # vertical backward messages
        elif direction == 'U':

            # run the chain
            m[a] = np.moveaxis(dp_chain(np.moveaxis(cv, 0, 1)[:, ::-1], np.moveaxis(f, 0, 1), np.moveaxis(m[a], 0, 1)), 0, 1)

        # vertical forward messages
        elif direction == 'D':

            # run the chain
            m[a] = np.moveaxis(dp_chain(np.moveaxis(cv, 0, 1), np.moveaxis(f, 0, 1), np.moveaxis(m[a], 0, 1)), 0, 1)

    # believes
    b = np.zeros((H, W, D))

    print("compute believes: ")

    for i in range(H):
        for j in range(W):

            b[i, j] = cv[i, j] + np.sum(m[:, i, j, :], axis=0)

    # get disp_map
    print("disp map: ")

    disp_map = np.argmin(b, axis=2)

    return disp_map


def plot_imgs(im0g, im1g):
    """
    just plot the images
    """
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()
    plt.show()


def plot_disp_map(disp_map):
    """
    plot winner takes all algorithm
    """
    plt.figure()
    #plt.imshow(wta, cmap='plasma')
    plt.imshow(disp_map, cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def main():

    # path to images
    img_path = '../ignore/ass3_data/'

    # some pre computed files
    cv_file = img_path + 'cv.npy'

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
    #cv = compute_cost_volume_sad(im0g, im1g, D_max, filter_radius)
    #cv = compute_cost_volume_ssd(im0g, im1g, D_max, filter_radius)
    #cv = compute_cost_volume_ncc(im0g, im1g, D_max, filter_radius)

    # save cost volume
    #np.save(cv_file, cv)

    # load pre computed cv file
    cv = np.load(cv_file)

    # print cv shape
    print("cv: ", cv.shape)

    # compute wta algorithm -> merely on cost-volume
    d_wta = calc_wta(cv)
    print("d_wta: ", d_wta.shape)

    # plot wta
    #plot_disp_map(d_wta)

    # Compute pairwise costs
    H, W, D = cv.shape
    f = get_pairwise_costs(H, W, D)

    print("f: ", f.shape)

    # Compute SGM
    d_sgm = compute_sgm(cv, f)

    print("disp_map sgm: ", d_sgm.shape)

    # plot disparity map
    plot_disp_map(d_sgm)


if __name__== "__main__":
    main()
