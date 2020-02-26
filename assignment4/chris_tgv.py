import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp


def _make_nabla(M, N):
    row = np.arange(0, M * N)
    dat = np.ones(M * N)
    col = np.arange(0, M * N).reshape(M, N)
    col_xp = np.hstack([col[:, 1:], col[:, -1:]])
    col_yp = np.vstack([col[1:, :], col[-1:, :]])

    nabla_x = scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(M * N, M * N)) - \
              scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

    nabla_y = scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(M * N, M * N)) - \
              scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

    nabla = scipy.sparse.vstack([nabla_x, nabla_y])

    return nabla, nabla_x, nabla_y


def compute_Wi(W, i):
    """
    Used for calculation of the dataterm projection

    can be used for confidences or set to zero if datapoint is not available
    @param W:
    @param i: index of the observation
    @return:
    """
    Wi = -np.sum(W[:, :, :i], axis=-1) + np.sum(W[:, :, i:], axis=-1)
    return Wi


def prox_sum_l1(u, f, tau, Wis):
    """
    Used for calculation of the dataterm projection

    compute pi with pi = \bar x + tau * W_i
    @param u: MN
    @param tau: scalar
    @param Wis: MN x K
    @param f: MN x K
    """
    pis = u[..., np.newaxis] + tau * Wis

    var = np.concatenate((f, pis), axis=-1)

    prox = np.median(var, axis=-1)

    return prox


def make_K(M, N):
    """
    @param M:
    @param N:
    @return: the K operator as described in Equation (5)
    """

    _, nabla_x, nabla_y = _make_nabla(M,N)    
    I = -1 * sp.identity(M*N, format='coo')
    #K = sp.coo_matrix((6*M*N, 3*M*N))
    K = scipy.sparse.bmat([[nabla_x, I, None], 
                           [nabla_y, None, I], 
                           [None, nabla_x, None],
                           [None, nabla_y, None],
                           [None, None, nabla_x],
                           [None, None, nabla_y]])
    
    return K


def proj_ball(Y, lamb):
    """
    Projection to a ball as described in Equation (6)
    @param Y: either 2xMN or 4xMN
    @param lamb: scalar hyperparameter lambda
    @return: projection result either 2xMN or 4xMN
    """
    return Y / (np.max((1, np.linalg.norm(Y) / lamb)))


def compute_accX(x, y, X=1, mask=None):
    """
    accuracy calculation
    """

    # set mask to all ones if not defined
    if mask is None:
        mask = np.ones(x.shape)

    # return accuracy meassure
    return np.sum(mask * (np.abs(x - y) <= X)) / np.sum(mask == 1)


def tgv2_pd(f, alpha, maxit):
    """
    @param f: the K observations of shape MxNxK
    @param alpha: tuple containing alpha1 and alpha2
    @param maxit: maximum number of iterations
    @return: tuple of u with shape MxN and v with shape 2xMxN
    """
    M, N, K = f.shape
    f = f.reshape(M*N, K)

    # make operators
    k = make_K(M,N)

    # Used for calculation of the dataterm projection
    W = np.ones((M, N, K))
    Wis = np.asarray([compute_Wi(W, i) for i in range(K)])
    Wis = Wis.transpose(1, 2, 0)
    Wis = Wis.reshape(M * N, K)

    # Lipschitz constant of K
    L = np.sqrt(12)

    # initialize primal variables
    u = np.zeros(M * N)
    v = np.zeros(2 * M * N)

    # initialize dual variables
    p = np.zeros(2 * M * N)
    q = np.zeros(4 * M * N)

    # concatenate
    u_bold = np.concatenate((u.ravel(), v.ravel()))
    p_bold = np.concatenate((p.ravel(), q.ravel()))

    # primal and dual step size
    tau = 1 / 24
    sigma = 1 / 24

    for it in range(0, maxit):
        # TODO calculate iterates as described in Equation (4)
        # To calculate the data term projection you can use:
        # prox_sum_l1(x, f, tau, Wis)
        # where x is the parameter of the projection function i.e. u^(n+(1/2))
        
        u_bold_n = u_bold

        u_bold = u_bold_n - tau * k.T @ p_bold

        u = prox_sum_l1(u_bold[:M*N], f, tau, Wis)
        v = u_bold[M*N:]

        p_bold = p_bold + sigma * k @ (2 * u_bold - u_bold_n)

        p = proj_ball(p_bold[:2*M*N].reshape(2, M*N), alpha[0])
        q = proj_ball(p_bold[2*M*N:].reshape(4, M*N), alpha[1])


    return u, v


def plot_data(f, gt):
    """
    plot sample data and ground truth
    """

    # plot obervations
    for obs in range(f.shape[2]):

        plt.figure()
        plt.imshow(f[:,:, obs], cmap='gray')
        plt.title("Observation {}".format(obs))
        plt.colorbar()
        plt.show()

    # plot ground truth
    plt.figure()
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth")
    plt.colorbar()
    plt.show()


def plot_result(u_tgv, alpha):
    """
    plot the result
    """
    plt.figure()
    plt.imshow(u_tgv, cmap='gray')
    plt.title("TGV alpha=[{}, {}]".format(alpha[0], alpha[1]))
    plt.colorbar()
    


def main():

    # path to img data and save location
    data_path = '../ignore/ass4_data/'
    
    # Load Observations
    samples = np.array([np.load(data_path + 'observation{}.npy'.format(i)) for i in range(0,9)])
    f = samples.transpose(1,2,0)
    print("samples f: ", f.shape)

    # shape of things
    M, N, K = f.shape

    # load ground truth
    gt = np.load(data_path + 'gt.npy')

    # max iterations
    maxit = 10
    
    # hyper params -> find good sets
    alpha_set = [(0.3, 0.5), (0.1, 1.0), (1.0, 0.1)]

    for alpha in alpha_set:

        # Perform TGV-Fusion
        u_tgv, v_tgv = tgv2_pd(f, alpha=alpha, maxit=maxit)  

        # Plot result
        plot_result(u_tgv.reshape(M, N), alpha)

        # Calculate Accuracy
        acc = compute_accX(u_tgv.reshape(M, N), gt, X=1)

        # print message
        print("maxit=[{}], alpha=[{}, {}], Acc: [{}]".format(maxit, alpha[0], alpha[1], acc))

    plt.show()
    # --
    # plot data
    #plot_data(f, gt)




if __name__== "__main__":
    main()

