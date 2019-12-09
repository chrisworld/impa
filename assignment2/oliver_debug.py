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
    X = np.einsum('nk, nkj->nj', F, E[None,:,:])
    z = -1./2 * np.einsum('cnp, cnpq, cnq -> cn',(X-means[:,None,:]), precisions[:,None,:,:], (X-means[:,None,:]))


    sign, logdet = LA.slogdet(LA.inv(precisions))

    # calculate log of weighted likelihood
    k = z + (np.log(weights) - (K/2. * np.log(2*np.pi) + 1./2 * logdet ))[:,np.newaxis]

    # determine component with highest log likelihood
    k_max = np.argmax(k, axis=0)

    U = w_filter(U, F, E, precisions, means, weights, lamb, k_max)

    return U

@numba.njit
def w_filter(U, F, E, precisions, means, weights, lamb, k_max):
    #wiener filter
    N, K = U.shape
    for i in range(0,N):
        A = np.linalg.inv(E.T@precisions[k_max[i]]@E + lamb*np.eye(K))
        b = E.T@(precisions[k_max[i]]@means[k_max[i]])
        U[i,:] = A@(b+lamb*F[i,:])
    return U
    

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
        z_k = np.max(z, axis=1) 
        #z_k = np.max(z)
        #----------------------------------------------------------
        
        _, logdet = LA.slogdet(sigma+np.eye(K)*1e-6)
        # pre calculate log of c for all components
        log_c = np.log(alpha)[:,np.newaxis] - (K/2. * np.log(2*np.pi) + 1./2 * logdet )[:,np.newaxis]
        #c = alpha * 1./np.sqrt((2*np.pi)**K * sign * np.exp(logdet))
        #gamma = np.exp(z + log_c - (z_k+ np.log(np.sum(np.exp(log_c + z-z_k), axis=0))))
        gamma = np.exp(z + log_c - (z_k[:, np.newaxis] + \
            np.log(np.sum(np.exp(log_c + z-z_k[:,np.newaxis]), axis=0))))

        # calculation of new alpha, mu and cov
        #----------------------------------------------------------
        alpha = np.einsum('cn->c',gamma)/N
        mu = np.einsum('cn,nk -> ck', gamma, X)/gamma.sum(1)[:,None]
        sigma = np.einsum('cn,cnp,cnq -> cqp', gamma, X-mu[:,None,:], \
            X-mu[:,None,:])/gamma.sum(axis=1)[:,None,None]

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
    C = 3  # Number of mixture components
    W = 9  # Window size
    K = W**2  # Number of pixels in each patch

    train_imgs = load_imgs("../ignore/train_set")
    val_imgs = load_imgs("../ignore/valid_set")
    test_imgs = np.load("../ignore/test_set.npy", allow_pickle=True).item()

    MM, NN, w, _ = view_as_windows(train_imgs[0], (W,W), step=1).shape
    # --
    # patches for training

    # hop size of patching
    hop = 1

    # training patches
    X, mmnn_list = get_patches(train_imgs, W, K, hop, n=10000, rand_sel=True)


    # --
    # training

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X, C=C, max_iter=10)
     
    # The Wiener filter requires the precision matrix which is the inverted covariance matrix
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6) 
    plot(gmm['mu'][0], gmm['precisions'][0], W)

    
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
    lamb = 100
    alpha = 0.001
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