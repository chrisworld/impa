import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.util import view_as_windows
from skimage.color import rgb2gray
from numpy import linalg as LA
from skimage import io
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


def make_dictionary(d):
    N,M,M = d.shape
    NH = np.int32(np.sqrt(N))
    dict = np.zeros((NH*M, NH*M))
    ii = 0
    idx = 0
    for i in range(0,NH):
        jj = 0
        for j in range(0,NH):
            dd = np.copy(d[idx,:,:])
            dd -= dd.min()
            dd /= dd.max()
            dict[ii:ii+M, jj:jj+M] = dd
            jj += M
            idx += 1
        ii +=M
    return dict


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
    N, K = U.shape
    C = weights.shape[0]
    X = F @ E
    z = -1./2 * np.einsum('cnp, cnpq, cnq -> cn',(X-means[:,None,:]), precisions[:,None,:,:], (X-means[:,None,:]))

    sign, logdet = LA.slogdet(LA.inv(precisions))
    # first implement 
    k = z + np.log(weights)[:,np.newaxis] - (K/2. * np.log(2*np.pi) + 1./2 * logdet )[:,np.newaxis]
    # calculate log of weighted likelihood
    #k = z + np.log(weights[:,np.newaxis]) - (K/2. * np.log(2*np.pi) + 1./2. * LA.slogdet(LA.inv(precisions))[1])[:,np.newaxis]

    output = np.zeros((N,K))
    for n in range(N):
        # determine component with maximum likelihood
        k_max = np.argmax(k[:,n])
        # calculate the inverse term of the wiener filter
        inv_arg = LA.inv(np.eye(K)*lamb + E.T@ precisions[k_max] @ E)
        # save the result in output
        output[n,:] = inv_arg @ (U[n,:] * lamb + precisions @ E @ means[k_max,:])

    return output

    # wiener filter: 
    

    #pass


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
    cov = np.random.randn(K,K)
    sigma = np.repeat((cov.T @ cov)[np.newaxis,:,:], C, axis=0)
    #----------------------------------------------------------
    # calculate argument of e^(arg) for each patch and kernel
    #----------------------------------------------------------
    z = np.zeros((C,N))
    for i in range(max_iter):

        # calculate argument of exponent
        inv_sigma = LA.inv(sigma + np.eye(K)*1e-6)
        z = -1./2 * np.einsum('cnp, cnpq, cnq -> cn',(X-mu[:,None,:]), inv_sigma[:,None,:,:], (X-mu[:,None,:]))
        #z_k = np.max(z, axis=1) 
        z_k = np.max(z)
        #----------------------------------------------------------
        
        sign, logdet = LA.slogdet(sigma+np.eye(K)*1e-6)
        # first implement 
        log_c = np.log(alpha)[:,np.newaxis] - (K/2. * np.log(2*np.pi) + 1./2 * logdet )[:,np.newaxis]
        #c = alpha * 1./np.sqrt((2*np.pi)**K * sign * np.exp(logdet))
        gamma = np.exp(z + log_c - (z_k+ np.log(np.sum(np.exp(log_c + z-z_k), axis=0))))
        #gamma = np.exp(z + log_c - (z_k[:, np.newaxis] + np.log(np.sum(np.exp(log_c + z-z_k[:,np.newaxis]), axis=0))))

        #----------------------------------------------------------
        alpha = np.einsum('cn->c',gamma)/N
        mu = np.einsum('cn,nk -> ck', gamma, X)/gamma.sum(1)[:,None]
        sigma = np.einsum('cn,cnp,cnq -> cqp', gamma, X-mu[:,None,:], \
            X-mu[:,None,:])/gamma.sum(axis=1)[:,None,None]

        
        #print('----------------------------------------------------------------------------------')

    return alpha, mu, sigma


def load_imgs(dir):
    files = glob.glob('{}/*.png'.format(dir))
    imgs = [ski.img_as_float(ski.io.imread(fname)) for fname in files]

    return imgs

def plot(mu, precisions, w):
    plt.figure(2, figsize=(5,5))

    plt.subplot(121)
    plt.imshow(mu.reshape(w,w), cmap="gray")

    plt.subplot(122)

    eigval, eigvec = LA.eig(precisions)
    filters = np.zeros((w*w, w, w))
    for i in range(0,w*w):
        filters[i,:,:] = eigvec[:,i].reshape(w,w)
    dict = make_dictionary(filters)
    plt.imshow(dict, cmap="gray")
    plt.show()



def denoise():
    # TODO: Find appropiate parameters
    C = 4  # Number of mixture components
    W = 9  # Window size
    K = W**2  # Number of pixels in each patch

    train_imgs = load_imgs("../ignore/train_set")
    val_imgs = load_imgs("../ignore/valid_set")
    test_imgs = np.load("../ignore/test_set.npy", allow_pickle=True).item()

    # TODO: Create array X of shape (N,K) containing N image patches with K pixels in each patch. X are the patches to train the GMM.
    X = np.zeros((0,K))
    MM, NN, w, _ = view_as_windows(train_imgs[0], (W,W), step=1).shape
    for i in train_imgs:
        X = np.vstack((X, view_as_windows(i, (W,W),step=1).reshape([-1,K])))

    rnd_idx = np.random.choice(range(np.shape(X)[0]), 1000, replace=False)

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X[rnd_idx], C=C, max_iter=30)
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6)  # The Wiener filter requires the precision matrix which is the inverted covariance matrix

    plot(gmm['mu'][0], gmm['precisions'][0], w)
    # TODO: For the train and validation set use the get_noisy_img function to add noise on images.
    # noisy training imgs
    # init noisy imgs lists
    train_noisy_imgs = []
    val_noisy_imgs = []

    for img in train_imgs:
        train_noisy_imgs.append(get_noisy_img(img))

    # noisy validation imgs
    for img in val_imgs:
        val_noisy_imgs.append(get_noisy_img(img))

    # TODO: Create array F of shape (N,K) containing N image patches with K pixels in each patch. F are the patches to denoise.

    F = np.zeros((0,K))
    for i in train_imgs:
        F = np.vstack((F, view_as_windows(i, (W,W),step=1).reshape([-1,K])))

    # TODO: Set parameter for Algorithm 1
    lamb = 100
    alpha = 0.001
    maxiter = 3

    E = get_e_matrix(K)

    clean_img = val_imgs[0]
    noisy_img = val_noisy_imgs[0]
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