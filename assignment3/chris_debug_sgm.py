import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows

# for bonus task
from scipy import ndimage

# otsu threshold for bonus task
from skimage import filters


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


def get_padded_disp_imgs(left_image, right_image, D, radius):
    """
    pad the images according to filter radius and make disparities of right image I1
    """

    # get dimensions
    H, W = left_image.shape

    # copy and pad images
    I0 = np.pad(left_image, radius//2)
    I1 = np.pad(right_image, radius//2)

    # get disparity images I(x-d, y)
    I1_d = np.zeros((D, H + radius - 1, W + radius - 1))

    # calculate disparity image
    for d in np.arange(D):

        # move right image to right x-direction
        I1_d[d] = np.roll(I1, d, axis=1)

    return I0, I1_d


def compute_cost_volume_selector(im0g, im1g, D_max, filter_radius, d_method='NCC'):
    """
    Select cost volume from distance measure methods SAD, NCC or SSD
    """

    if d_method == 'SAD':
        return compute_cost_volume_sad(im0g, im1g, D_max, filter_radius)

    elif d_method == 'SSD':
        return compute_cost_volume_ssd(im0g, im1g, D_max, filter_radius)

    elif d_method == 'NCC':
        return compute_cost_volume_ncc(im0g, im1g, D_max, filter_radius)

    print('***no valid distance measure method selected, select from [SAD, SSD, NCC]')

    return 0



def compute_cost_volume_sad(left_image, right_image, D, radius):
    """
    Sum of Absolute Differences (SAD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """

    # padded imgs with disparity of right image
    I0, Imgs_d = get_padded_disp_imgs(left_image, right_image, D, radius)

    # hop size in pixel
    hop = 1

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

    # padded imgs with disparity of right image
    I0, Imgs_d = get_padded_disp_imgs(left_image, right_image, D, radius)

    # hop size in pixel
    hop = 1

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

    # padded imgs with disparity of right image
    I0, Imgs_d = get_padded_disp_imgs(left_image, right_image, D, radius)

    # hop size in pixel
    hop = 1

    # create patches of left image
    I0_patches = view_as_windows(I0, (radius, radius), step=hop)

    # calculate means
    mu_I0_p = np.mean(np.mean(I0_patches, axis=2), axis=2)

    # substract mean
    I0_p_mu = I0_patches - mu_I0_p[:, :, np.newaxis, np.newaxis]
    I0_p_mu_sq = np.power(I0_p_mu, 2)

    # patch shapes
    H_p, W_p, rh, rw = I0_patches.shape

    # init cv
    cv = np.zeros((H_p, W_p, D))
    
    # compare left with right image with different disparities
    for d, Id in enumerate(Imgs_d):

        # get patches
        Id_patches = view_as_windows(Id, (radius, radius), step=hop)

        # calculate means
        mu_Id_p = np.mean(np.mean(Id_patches, axis=2), axis=2)

        # substract of mean
        Id_p_mu = Id_patches - mu_Id_p[:, :, np.newaxis, np.newaxis]
        Id_p_mu_sq = np.power(Id_p_mu, 2)

        # calculate cost-volume
        cv[:, :, d] = -1 * np.sum(np.sum(I0_p_mu * Id_p_mu, axis=3), axis=2) / np.sqrt( np.sum(np.sum(I0_p_mu_sq, axis=3), axis=2) * np.sum(np.sum(Id_p_mu_sq, axis=3), axis=2) ) 

    return cv


def get_edge_weights(im0g):
    """
    calculate edge dependend weights for bonus task
    """

    # calculate edges with sobel filter
    sx = ndimage.sobel(im0g, axis=0, mode='constant')
    sy = ndimage.sobel(im0g, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    # calculate weights
    weights = sob / np.max(sob)

    # plt.figure()
    # plt.imshow(weights, cmap='gray')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # use simpler values for weights with threshold
    otsu_thresh = filters.threshold_otsu(weights)

    #print("otsu_thresh: ", otsu_thresh)

    # strong edge
    weights[weights > otsu_thresh] = 1.0

    # homogenious regions -> more penalty
    weights[weights <= otsu_thresh] = 2.0

    # other weighting
    #weights = (1 - weights) + 1.0

    return weights


def get_pairwise_costs(H, W, D, weights=None, L1=0.1, L2=0.2):
    """
    :param H: height of input image
    :param W: width of input image
    :param D: maximal disparity
    :param weights: edge-dependent weights (necessary to implement the bonus task)
    :return: pairwise_costs of shape (H,W,D,D)
             Note: If weight=None, then each spatial position gets exactly the same pairwise costs.
             In this case the array of shape (D,D) can be broadcasted to (H,W,D,D) by using np.broadcast_to(..).
    """

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
    f = np.broadcast_to(costs, (H, W, D, D))

    # weights
    if weights is not None:
        return f * weights[:, :, np.newaxis, np.newaxis]

    return f


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

        print("chain: ", i)
        # parallel computation
        m[:, i+1] = np.min(m[:, i, :, np.newaxis] + f[:, i] + g[:, i, :, np.newaxis], axis=1)

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

    # four directions for sgm algorithm
    directions = ['L', 'R', 'U', 'D']

    # init messages
    m = np.zeros((len(directions), H, W, D))

    # init believes
    b = np.zeros((H, W, D))

    # init disparity map
    disp_map = np.zeros((H, W))

    # compute paiwise costs
    for a, direction in enumerate(directions):

        print("message direction: ", direction)

        # horizontal backward messages
        if direction == 'L':

            # plt.figure()
            # plt.imshow(calc_wta(cv[:, ::-1]), cmap='jet')
            # plt.show()
            
            # run the chain
            m[a] = dp_chain(cv[:, ::-1], f[:, ::-1], m[a])[:, ::-1]

            # plt.figure()
            # plt.imshow(calc_wta(m[a]), cmap='jet')
            # plt.show()

        # horizontal forward messages
        elif direction == 'R':

            # plt.figure()
            # plt.imshow(calc_wta(cv), cmap='jet')
            # plt.show()

            # run the chain
            m[a] = dp_chain(cv, f, m[a])

            # plt.figure()
            # plt.imshow(calc_wta(m[a]), cmap='jet')
            # plt.show()

        # vertical backward messages
        elif direction == 'U':

            # plt.figure()
            # plt.imshow(calc_wta(np.moveaxis(cv, 0, 1)[:, ::-1]), cmap='jet')
            # plt.show()

            # run the chain
            m[a] = np.moveaxis(dp_chain(np.moveaxis(cv, 0, 1)[:, ::-1], np.moveaxis(f, 0, 1)[:, ::-1], np.moveaxis(m[a], 0, 1)), 0, 1)[::-1, :]

            # plt.figure()
            # plt.imshow(calc_wta(m[a]), cmap='jet')
            # plt.show()

        # vertical forward messages
        elif direction == 'D':

            # plt.figure()
            # plt.imshow(calc_wta(np.moveaxis(cv, 0, 1)), cmap='jet')
            # plt.show()

            # run the chain
            m[a] = np.moveaxis(dp_chain(np.moveaxis(cv, 0, 1), np.moveaxis(f, 0, 1), np.moveaxis(m[a], 0, 1)), 0, 1)

            # plt.figure()
            # plt.imshow(calc_wta(m[a]), cmap='jet')
            # plt.show()

    # compute believes
    b = cv + np.sum(m, axis=0)

    # get disp_map
    disp_map = np.argmin(b, axis=2)

    return disp_map


def acc_disp_map(d_img0, d_img1, mask, X=1):
    """
    accuracy calculation of disparity maps
    """
    return np.sum(mask * (np.abs(d_img0 - d_img1) <= X)) / np.sum(mask == 1)


def plot_imgs(im0g, im1g):
    """
    just plot the images
    """
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()
    plt.show()


def plot_disp_map(disp_map, d_method, L1, L2, acc, print_to_file=True, fig_path='./'):
    """
    plot winner takes all algorithm
    """
    plt.figure()
    #plt.imshow(wta, cmap='plasma')
    plt.imshow(disp_map, cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.axis('off')

    if print_to_file:
        plt.savefig(fig_path + 'disp_map_' + d_method + '_' + str(L1).replace('.', 'p') + '_' + str(L2).replace('.', 'p') + '_acc-' + str(acc)[0:6].replace('.', 'p') + '.png', dpi=150)

    else:
        plt.show()



def load_imgs(img_path, img_name, dark_img, use_dark_img=False):
    """
    load input images
    """

    # left image for reference
    im0 = imread(img_path + img_name + "_left.png")

    # right image dark
    if img_name == 'Adirondack' and use_dark_img:
        im1 = imread(img_path + dark_img + "_right.png")

    # right image normal
    else:
        im1 = imread(img_path + img_name + "_right.png")

    # ground truth
    d_gt = imread(img_path + img_name + "_gt.png")

    # mask
    mask = imread(img_path + img_name + "_mask.png") // 255

    # convert to gray values
    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    return im0g, im1g, d_gt, mask


def main():

    # path to img data and save location
    img_path = '../ignore/ass3_data/'
    fig_path = '../ignore/ass3_data/figs/'

    # some pre computed files
    cv_file = img_path + 'cv.npy'
    d_sgm_file = img_path + 'd_sgm.npy'

    # image names
    img_names = ['Adirondack', 'cones']
    dark_img = ['AdirondackE']

    # choose image
    img_name = img_names[0]

    # loads the images and converts them to grey
    im0g, im1g, d_gt, mask = load_imgs(img_path, img_name, dark_img, use_dark_img=False)

    # plot image data
    print("----image: {}-----".format(img_name))
    print("image1 shape: ", im0g.shape)
    print("image2 shape: ", im1g.shape)
    print("gt shape: ", d_gt.shape)
    print("mask shape: ", mask.shape)

    # load pre computed files
    cv = np.load(cv_file)
    #d_sgm = np.load(d_sgm_file)

    # maximal disparity
    D_max = 64

    # filter radius
    filter_radius = 5

    # distance measure methods
    #d_methods = ['SAD', 'SSD', 'NCC']
    d_methods = ['NCC']

    # pairwise cost of levels
    #L1_set = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    #L2_set = [0.2, 0.5, 1.5, 0.5, 0.8, 1.2]
    #L1_set = [0.01, 0.01, 0.1, 0.1]
    #L2_set = [0.2, 1.2, 2.0, 5.0]

    L1_set = [0.1]
    L2_set = [2.0]

    # bonus task
    #weights = None
    weights = get_edge_weights(im0g)

    print(weights)

    # run through each method
    for d_method in d_methods:

        # L1, L2 set tests
        for L1, L2 in zip(L1_set, L2_set):

            print("...compute cv")
            # compute cost volume
            #cv = compute_cost_volume_selector(im0g, im1g, D_max, filter_radius, d_method=d_method)

            # shape
            H, W, D = cv.shape

            print("...compute pairwise cost")
            # pairwise cost
            f = get_pairwise_costs(H, W, D, weights=weights, L1=L1, L2=L2)
            print("shape: ", f[0:2, 0:2])
            print("shape: ", f.shape)

            print("...compute sgm")
            # Compute SGM
            d_sgm = compute_sgm(cv, f)

            # calculate accuracy
            acc = acc_disp_map(d_gt, d_sgm, mask)

            # print
            print("Method: {} L1: [{:.1f}] L2: [{:.1f}] acc: [{:.4f}]".format(d_method, L1, L2, acc))

            # --
            # plots
            #plot_imgs(im0g, im1g)
            #plot_disp_map(calc_wta(cv))
            plot_disp_map(d_sgm, d_method, L1, L2, acc, print_to_file=True, fig_path=fig_path)


    # --
    # save files for speedup

    #np.save(cv_file, cv)
    #np.save(d_sgm_file, d_sgm)



if __name__== "__main__":
    main()
