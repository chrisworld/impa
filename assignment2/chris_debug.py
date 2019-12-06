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
    N, K = U.shape
    C = weights.shape[0]

    # init filtered patches
    xh = np.zeros((N, K))

    # init vars
    nom = np.zeros((C, K))
    denom = np.zeros((C, K, K))
    log_c = np.zeros(C)
    z = np.zeros(C)

    dim_norm = K / 2 * np.log(2 * np.pi)


    # Pre-Computations to make it faster
    for k in range(C):

        # nominator and denom
        nom[k] =  np.dot(precisions[k], np.dot(E, means[k]))
        denom[k] = np.linalg.inv(lamb * np.identity(K) + E.T * precisions[k] * E)  

        # normal distribution normalization term
        log_c[k] = np.log(weights[k]) - dim_norm + LA.slogdet(precisions[k])[1] / 2


    # run through each patch
    for i, yi in enumerate(U):

        # determine best kernel for actual noisy patch xi -> F[i]
        for k in range(C):

            # determine exp term in normal dist
            z[k] = -np.dot( np.dot((F[i] - means[k]).T, precisions[k]), F[i] - means[k] ) / 2
        
        # get maximum likelihood
        k_max = np.argmax(log_c + z)

        # apply wiener filter
        xh[i] = np.dot(denom[k_max], lamb * yi + nom[k_max])

    return xh


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


    # calculate argument of e^(arg) for each patch and kernel
    #----------------------------------------------------------
    z = np.zeros((C,N))
    for i in range(max_iter):

        inv_sigma = LA.inv(sigma)
        for c in range(C):
            for n in range(N):
                x_i = X[n,:] - mu[c,:]
                z[c,n] = -1/2 * x_i @ inv_sigma[c,:,:] @ x_i.T
        z_k = np.max(z, axis=1)
        print(z_k)


        # loop over all patches and sum up along the kernels
        #----------------------------------------------------------
        c_z = alpha * 1/np.sqrt((2*np.pi)**K * np.exp(LA.slogdet(sigma)[1]))
        logArg = c_z[:,np.newaxis] * np.exp(z - z_k[:,np.newaxis])

        
        # calculate gamma for each patch
        #----------------------------------------------------------
        log_c = np.log(alpha) - (K/2 * np.log(2*np.pi) + 1/2 * LA.slogdet(sigma)[1])
        gamma = np.exp(log_c[:,np.newaxis] + z - (z_k[:,np.newaxis] + np.log(np.sum(logArg, axis=0))))


        alpha = 1./N * np.sum(gamma, axis=1)
        mu = (gamma @ X)/np.sum(gamma[:,:,np.newaxis], axis=1)

        addedNoise = np.eye(K) * 10**(-6)

        sigma = np.zeros((C,K,K))
        for n in range(N):
            for c in range(C):
                x_ = X[n,:] - mu[c,:]
                x_i = np.outer(gamma[c,n]* x_, x_)
                sigma[c,:,:] += x_i / np.sum(gamma[c,:])
        sigma = sigma + addedNoise

    return alpha, mu, sigma


def load_imgs(dir):
    files = glob.glob('{}/*.png'.format(dir))
    imgs = [ski.img_as_float(ski.io.imread(fname)) for fname in files]

    return imgs


def get_patches(imgs, W, K, hop, n=100000000, rand_sel=False):
    """
    get patches from list of imgs
    """

    # init
    X = np.empty((0, K))

    # patch space list
    mmnn_list = []

    for img in imgs:

        # make patches
        X_img = view_as_windows(img, (W, W), step=hop)
        MM, NN, _, _ = X_img.shape

        # get patch size
        n_patches = MM * NN

        # concatenate patches
        X = np.concatenate((X, np.reshape(X_img, (n_patches, K))))

        # add patch dimensions for each file
        mmnn_list.append((MM, NN))

    # return if n is larger than actual samples
    if n > X.shape[0]:
        return X, mmnn_list

    # random or linear selection
    if rand_sel:
        # random selection of patches
        sel = np.random.choice(X.shape[0], n)

    else:
        # linear selection
        sel = np.arange(n)

    return X[sel], mmnn_list


def denoise():
    # TODO: Find appropiate parameters
    C = 8  # Number of mixture components
    W = 5  # Window size
    K = W**2  # Number of pixels in each patch

    train_imgs = load_imgs("../ignore/train_set")
    val_imgs = load_imgs("../ignore/valid_set")
    test_imgs = np.load("../ignore/test_set.npy", allow_pickle=True).item()


    # --
    # patches for training

    # hop size of patching
    hop = W

    # training patches
    X, mmnn_list = get_patches(train_imgs, W, K, hop, n=1000, rand_sel=True)

    # (N, K) patches
    print("X: ", X.shape)


    # test patches
    #ski.io.imshow(np.reshape(X_val[0], (W, W)))
    #plt.show()


    # --
    # training

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X[0:10000], C=C, max_iter=10)
     
    # The Wiener filter requires the precision matrix which is the inverted covariance matrix
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6) 

    
    # -- 
    # add noise to images

    # init noisy imgs lists
    train_noisy_imgs = []
    val_noisy_imgs = []

    # noisy training imgs
    for img in train_imgs:
        train_noisy_imgs.append(get_noisy_img(img))

    # noisy validation imgs
    for img in val_imgs:
        val_noisy_imgs.append(get_noisy_img(img))


    # --
    # create patches for denoising -> F

    # hop size -> should be windows size
    hop = 1
    #hop = 1

    # training patches
    F, mmnn_list = get_patches(val_noisy_imgs, W, K, hop)

    # (N, K) patches
    N, K = F.shape

    print("F pre: ", F.shape)

    # --
    # reconstruction shapes

    # amount of img files
    n_imgs = len(val_noisy_imgs)

    # amount of patches per image
    n_patches = N // n_imgs

    # reshape patches witch additional file space dimension
    F = np.reshape(F, (n_imgs, n_patches, K))

    # some prints
    print("n_patches: ", N / n_imgs)
    print("noisy val set:: ", len(val_noisy_imgs), val_noisy_imgs[0].shape)
    print("F reshaped: ", F.shape)
    print("patch space: MM, NN: ", mmnn_list)


    # --
    # Wiener filter params

    # params
    lamb = 1
    alpha = 0.5
    maxiter = 2

    # zero mean average matrix
    E = get_e_matrix(K)

    # for testing
    image_sel = slice(0, 1, 1)

    F = F[image_sel]
    print("F_test: ", F.shape)

    # Initialize with the noisy image patches
    U = F.copy()  

    # --
    # Wiener filtering

    print("---Wiener filtering---")
    # for each image
    for i, clean_img in enumerate(val_imgs[image_sel]):

        print("--image: {}".format(i))

        # iterations
        for iter in range(0, maxiter):

            # wiener filter
            U[i] = alpha * U[i] + (1 - alpha) * wiener_filter(U[i], F[i], E, gmm['precisions'], gmm['mu'], gmm['alpha'], lamb)

            print("Ui: ", U[i].shape)
            # reconstruction
            u = reconstruct_average(U[i].reshape(mmnn_list[i][0], mmnn_list[i][1], W, W))

            print("u: ", u.shape)

            # plot iteration
            plt.figure(5, figsize=(10,5))
            plt.subplot(121)
            plt.imshow(clean_img, cmap="gray")
            plt.subplot(122)
            plt.imshow(u, cmap="gray")
            plt.show()

            # PSNR
            psnr_denoised = compute_psnr(u, clean_img)
            print("Iter: {} - PSNR: {}".format(iter, psnr_denoised))

        
        psnr_noisy = compute_psnr(val_noisy_imgs[i], clean_img)
        psnr_denoised = compute_psnr(u, clean_img)

        print("PSNR noisy: {} - PSNR denoised: {}".format(psnr_noisy, psnr_denoised))


if __name__ == "__main__":
    denoise()