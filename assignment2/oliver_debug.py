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
    pass


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

    N, K = X.shape

    # initialize mu, alpha and sigma
    #----------------------------------------------------------
    alpha = np.squeeze(np.random.dirichlet(np.ones(C), size=1))
    mu = np.zeros((C,K))
    sigma = np.zeros((C,K,K))
    for c in range(C):
        cov = np.random.randn(K,K)
        sigma[c,:,:] = cov.T @ cov
    #----------------------------------------------------------



    # calculate argument of e^(arg) for each patch and kernel
    #----------------------------------------------------------
    z = np.zeros((C,N))
    for i in range(max_iter):
        #print(sigma)
        #print('---------------------')
        inv_sigma = LA.inv(sigma)
        for c in range(C):
            for n in range(N):
                x_i = X[n,:] - mu[c,:]
                z[c,n] = -1/2 * x_i @ inv_sigma[c,:,:] @ x_i.T
        z_k = np.max(z, axis=1) 
        #z = np.max(x, axis=1)
        #----------------------------------------------------------

        # loop over all patches and sum up along the kernels
        #----------------------------------------------------------
        c_z = alpha * 1/np.sqrt((2*np.pi)**K * np.exp(LA.slogdet(sigma)[1]))
        logArg = c_z[:,np.newaxis] * np.exp(z - z_k[:,np.newaxis])

    #   logArg = np.zeros((C,N))
    #   for n in range(N):
    #       for c in range(C):
    #           logArg[c,n] = alpha[c] * 1/np.sqrt((2*np.pi)**K * LA.det(sigma[c,:,:])) * np.exp(x[c,n]-z_k[c])
        #----------------------------------------------------------
        
        # calculate gamma for each patch
        #----------------------------------------------------------
        log_c = np.log(alpha) - (K/2 * np.log(2*np.pi) + 1/2 * LA.slogdet(sigma)[1])
        gamma = np.exp(log_c[:,np.newaxis] + z - (z_k[:,np.newaxis] + np.log(np.sum(logArg, axis=0))))
        #gamma = np.exp(log_c[:,np.newaxis] + x - (z + np.log(np.sum(logArg, axis=0))))
    #   gamma_ = np.zeros((C,N))
    #   for n in range(N):
    #       for c in range(C):
    #           gamma_[c,n] = log_c[c] + x[c,n] - (z_k[c] + np.log(np.sum(logArg[:,n])))
        #----------------------------------------------------------

        alpha = 1./N * np.sum(gamma, axis=1)
        mu = (gamma @ X)/np.sum(gamma[:,:,np.newaxis], axis=1)

        addedNoise = np.eye(K) * 10**(-6)
    #    t = np.zeros((C,K))
    #   for n in range(N):
    #       for c in range(C):
    #           t[c,:] += (gamma[c,n]*X[n,:]) / np.sum(gamma[c,:])

        sigma = np.zeros((C,K,K))
        for n in range(N):
            for c in range(C):
                x_ = X[n,:] - mu[c,:]
                x_i = np.outer(gamma[c,n]* x_, x_)
                sigma[c,:,:] += x_i / np.sum(gamma[c,:])
        sigma = sigma + addedNoise
    #pass
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

    train_imgs = load_imgs("train_set")
    val_imgs = load_imgs("valid_set")
    test_imgs = np.load("test_set.npy", allow_pickle=True).item()

    # TODO: Create array X of shape (N,K) containing N image patches with K pixels in each patch. X are the patches to train the GMM.

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X, C=C, max_iter=30)
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6)  # The Wiener filter requires the precision matrix which is the inverted covariance matrix

    # TODO: For the train and validation set use the get_noisy_img function to add noise on images.
    # TODO: Create array F of shape (N,K) containing N image patches with K pixels in each patch. F are the patches to denoise.

    # TODO: Set parameter for Algorithm 1
    # lamb =
    # alpha =
    # maxiter =

    E = get_e_matrix(K)

    # Use Algorithm 1 for Patch-Denoising
    U = F.copy()  # Initialize with the noisy image patches
    for iter in range(0, maxiter):
        U = alpha * U + (1 - alpha) * wiener_filter(U, F, E, gmm['precisions'], gmm['mu'], gmm['alpha'], lamb)
        u = reconstruct_average(U.reshape(MM, NN, W, W))

        psnr_denoised = compute_psnr(u, clean_img)
        print("Iter: {} - PSNR: {}".format(iter, psnr_denoised))

    psnr_noisy = compute_psnr(noisy_img, clean_img)
    psnr_denoised = compute_psnr(u, clean_img)

    print("PSNR noisy: {} - PSNR denoised: {}".format(psnr_noisy, psnr_denoised))


if __name__ == "__main__":
    denoise()