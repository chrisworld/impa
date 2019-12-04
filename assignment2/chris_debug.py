import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.util import view_as_windows
from skimage.color import rgb2gray
from numpy import linalg as LA
import numba
import pickle
import collections
import glob

rng = np.random.RandomState(seed=42)


def compute_psnr(img1, img2):
    """
    :param img1:
    :param img2:
    :return: the PSNR between img1 and img2
    """
    mse = np.mean((img1 - img2)**2)
    return (10 * np.log10(1.0 / mse))


def reconstruct_average(P):
    """
    :param P: (MM,NN,W,W)
    :return: (M,N)
    """
    MM, NN, w, _ = P.shape
    M = MM + w - 1
    N = NN + w - 1
    p = np.zeros((M, N))
    c = np.zeros((M, N))
    for x in range(0, w):
        for y in range(0, w):
            p[y:MM + y, x:NN + x] += P[:, :, y, x]
            c[y:MM + y, x:NN + x] += 1
    p /= c
    return p


def wiener_filter(U, F, E, precisions, means, weights, lamb):
    """
    Applies the wiener filter to N patches each having K pixels.
    The parameters of a learned GMM with C kernels are passed as an argument.

    :param U: (N,K) denoised patches from previous step
    :param F: (N,K) noisy patches
    :param E: (K,K) matrix that projects patches onto a set of zero-mean patches
    :param precisions: (C,K,K) precisions of the GMM
    :param means: (C,K) mean values of the GMM
    :param weights: (C) weights for each kernel of the GMM
    :param lamb: lambda parameter of the Wiener filter
    :return: (N,K) result of the wiener filter, equivalent to x_i^~ in Algorithm 1
    """
    
    # get some numbers
    K = U.shape[1]

    # get k for closest kernel to the patch
    # TODO:
    k = 1

    # apply wiener filter
    xi = np.linalg.inv(lamb * np.identity(K) + E.T * precisions[k] * E) * (lamb * F + precisions[k] * E * means[k])


    return xi


def get_noisy_img(clean_img):
    """
    Adds noise on the given input image

    :param clean_img:
    :return:
    """
    assert(clean_img.min()>=0.0)
    assert(clean_img.max()<=1.0)
    assert(len(clean_img.shape)==2)

    sigma = 25.0 / 255.0
    noisy_img = clean_img + rng.randn(*clean_img.shape) * sigma

    return noisy_img


def get_e_matrix(K):
    """
    Returns a matrix that projects a patch onto the set of zero-mean patches

    :param K: total number of pixels in a patch
    :return: (K,K) projection matrix
    """
    return np.identity(K) - 1 / K * np.outer(np.ones(K), np.ones(K))


def train_gmm(X, C, max_iter, plot=False):
    """
    Trains a GMM with the EM algorithm
    :param X: (N,K) N image patches each having K pixels that are used for training the GMM
    :param C: Number of kernels in the GMM
    :param max_iter: maximum number of iterations
    :param plot: set to true to plot steps of the algorithm
    :return: alpha: (C) weight for each kernel
             mu: (C,K) mean for each kernel
             sigma: (C,K,K) covariance matrix of the learned model
    """
    
    K = X.shape[1]

    # randomly init gaussians
    # weight of kernels -> must sum to one
    alpha = np.squeeze(np.random.dirichlet(np.ones(C), size=1))

    # means of each kernel
    mu = np.random.random((C, K))

    # covariance matrix of each kernel
    sigma = np.random.random((C, K, K))

    # functions for log sum trick
    f_zk = lambda x, mu, sigma: - np.dot(np.dot((x - mu).T, (LA.inv(sigma))), (x - mu)) / 2 
    f_log_ck = lambda a, sigma: np.log(a) - (len(sigma) * np.log(2 * np.pi) + LA.slogdet(sigma)) / 2

    # TODO: 
    # init vars
    zk = 0
    log_ck = 0

    # for each kernel
    for c in range(C):

        print("kernel: ", c)

        # helper vars
        zk = f_zk(X[0], mu[c], sigma[c])
        log_zk = f_log_ck(alpha[c], sigma[c])

    print("zk: ", zk)
    print("log_ck: ", log_ck)

    # take max of zk
    #z = max(zk)

    # iterate
    # for j in range(max_iter):

    #     # log sum trick
    #     gamma = np.exp(log_ck(alpha, sigma) + zk(x, mu, sigma) - )


    return alpha, mu, sigma


def load_imgs(dir):
    files = glob.glob('{}/*.png'.format(dir))
    imgs = [ski.img_as_float(ski.io.imread(fname)) for fname in files]

    return imgs


def denoise():
    # TODO: Find appropiate parameters
    C = 2  # Number of mixture components
    W = 5  # Window size
    K = W**2  # Number of pixels in each patch

    train_imgs = load_imgs("../ignore/train_set")
    val_imgs = load_imgs("../ignore/valid_set")
    test_imgs = np.load("../ignore/test_set.npy", allow_pickle=True).item()

    # TODO: Create array X of shape (N,K) containing N image patches with K pixels in each patch. X are the patches to train the GMM.

    # hop size of patching
    hop = W

    # init
    X = np.empty((0, K))

    # train set: make patches for each file
    for img in train_imgs:

        # make patches
        X_img = view_as_windows(img, (W, W), step=hop)

        # get patch size
        n_patches = X_img.shape[0] * X_img.shape[1]

        # concatenate patches
        X = np.concatenate((X, np.reshape(X_img, (n_patches, K))))


    # simple
    #np.load("../ignore/simple.npy")

    # (N, K) patches
    print("X: ", X.shape)


    # test patches
    #ski.io.imshow(np.reshape(X_val[0], (W, W)))
    #plt.show()

    

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X[0:10], C=C, max_iter=30)
     
    # The Wiener filter requires the precision matrix which is the inverted covariance matrix
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6) 

    
    # TODO: For the train and validation set use the get_noisy_img function to add noise on images.
    train_imgs_noise = get_noisy_img(train_imgs)
    val_imgs_noise = get_noisy_img(train_imgs)


    # TODO: Create array F of shape (N,K) containing N image patches with K pixels in each patch. F are the patches to denoise.

    # init
    X_val_noise = np.empty((0, K))

    # validation set with noise
    for img in val_imgs_noise:

        # make patches
        X_img = view_as_windows(img, (W, W), step=hop)

        # get patch size
        n_patches = X_img.shape[0] * X_img.shape[1]

        # concatenate patches
        X_val_noise = np.concatenate((X_val_noise, np.reshape(X_img, (n_patches, K))))

    print("X_val_noise: ", X_val_noise.shape)

    # patches to denoise
    F = X_val_noise[0:10]


    # TODO: Set parameter for Algorithm 1
    lamb = 1
    alpha = 0.5
    maxiter = 10

    E = get_e_matrix(K)

    # Use Algorithm 1 for Patch-Denoising
    # Initialize with the noisy image patches
    U = F.copy()  

    # iterations
    for iter in range(0, maxiter):

        # wiener filter
        U = alpha * U + (1 - alpha) * wiener_filter(U, F, E, gmm['precisions'], gmm['mu'], gmm['alpha'], lamb)

        # reconstruction
        u = reconstruct_average(U.reshape(MM, NN, W, W))

        # PSNR
        psnr_denoised = compute_psnr(u, clean_img)
        print("Iter: {} - PSNR: {}".format(iter, psnr_denoised))

    
    psnr_noisy = compute_psnr(noisy_img, clean_img)
    psnr_denoised = compute_psnr(u, clean_img)

    print("PSNR noisy: {} - PSNR denoised: {}".format(psnr_noisy, psnr_denoised))


if __name__ == "__main__":
    denoise()