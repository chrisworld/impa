import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

# 3d plot
from mpl_toolkits.mplot3d import Axes3D


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
    
    # TODO
    return K


def proj_ball(Y, lamb):
    """
    Projection to a ball as described in Equation (6)
    @param Y: either 2xMN or 4xMN
    @param lamb: scalar hyperparameter lambda
    @return: projection result either 2xMN or 4xMN
    """
    # TODO
    return Y/(np.maximum(1, (1/lamb * np.linalg.norm(Y, axis=0))))

def L2_1norm(X):
    #Y = np.sum(np.square(np.sum(np.power(X,2), 1)),0)
    return np.linalg.norm(np.linalg.norm(X,2, axis=0),1)
    
def calc_energy(u_, alpha, f, nabla, nabla_tilde, M,N):
    """
    calculate the energy of the TGV regularized fusion task
    @param u: vector containing of (u_, v_).T
    @param alpha: vector of two regularization parameters alpha1 and alpha2
    @param f: K observations of the same scene
    """
    # seperate disparity map u and gradient map v
    u = u_[:M*N]
    v = u_[M*N:]
    # get regularization parameters
    a1 = alpha[0]
    a2 = alpha[1]
        
    E = a1 * L2_1norm(((nabla @ u) - v).reshape(2, M*N)) + \
        a2 * L2_1norm((nabla_tilde @ v).reshape(4, M*N)) + \
            np.sum(np.linalg.norm(u[:,np.newaxis] - f, ord=1, axis=0))

    return E

def compute_accX(x, y, X=1):
    M,N = y.shape
    Z = M*N 

    acc = 1.0 /Z * np.sum(np.abs(x - y)<=X)
    # TODO
    return acc

def plot_result(u_tgv, alpha):
    """
    plot the result
    """
    plt.figure()
    plt.imshow(u_tgv, cmap='gray')
    plt.title("TGV alpha=[{}, {}]".format(alpha[0], alpha[1]))
    plt.colorbar()

    # 3D plot

    # shape of things
    M, N = u_tgv.shape
    x = np.arange(0, M)
    y = np.arange(0, N)

    # meshgrid
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u_tgv.T, cmap='gray', linewidth=0, antialiased=False)
    plt.draw()


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

    # nablas
    nabla,_,_ = _make_nabla(M,N)
    # probably not correct, but lol.
    nabla_tilde = sp.bmat([ [nabla, None], 
                            [None, nabla]])


    # initializing primal points
    # setting up vectors u_ MN and v_ 2MN with zeros
    u_ = np.zeros((M*N))
    v_ = np.zeros((2*M*N))
    #combine them to u
    u = np.concatenate((u_,v_)) 

    # initializing dual points
    # set up p_ 2MN and q_ 4MN with zeros
    p_ = np.zeros((2*M*N))
    q_ = np.zeros((4*M*N))
    # combining them to p
    p = np.concatenate((p_,q_))

    sigma = tau = 1.0 / (2* L**2)

    # primal and dual step size
    #tau = 0.0  # TODO
    #sigma = 0.0  # TODO
    res = np.zeros((maxit))

    for it in range(0, maxit):
        # TODO calculate iterates as described in Equation (4)
        # To calculate the data term projection you can use:
        # prox_sum_l1(x, f, tau, Wis)
        # where x is the parameter of the projection function i.e. u^(n+(1/2))
        u_n  = u - tau*k.T * p       
        u_ = prox_sum_l1(u_n[:M*N], f, tau, Wis)
        v_ = u_n[M*N:]
        u_n = np.concatenate((u_,v_)) 
        # half step of p
        p_n = p + sigma*k * (2*u_n - u)
        p_ = proj_ball(np.reshape(p_n[:2*M*N], (2,M*N)), alpha[0])
        q_ = proj_ball(np.reshape(p_n[2*M*N:], (4,M*N)), alpha[1])
        p = np.concatenate((p_.ravel(),q_.ravel()))
        u = u_n
        res[it] = calc_energy(u, alpha, f, nabla, nabla_tilde, M, N)

    #return res, u
    return res, u


# Load Observations
samples = np.array([np.load('../ignore/ass4_data/observation{}.npy'.format(i)) for i in range(0,9)])
f = samples.transpose(1,2,0)
M, N, K = f.shape
alphaset = [(0.1, 10), (10, 0.1), (0.1, 0.1)]
#alphaset = [(1.5, 1.1)]

for alpha in alphaset:
# Perform TGV-Fusion
    res, v = tgv2_pd(f, alpha=alpha, maxit=500)  # TODO: set appropriate parameters
    u = v[:M*N].reshape(M,N)

    gt = np.load('../ignore/ass4_data/gt.npy')
    acc= compute_accX(u, gt)
    print(acc)

    plt.figure()
    plt.imshow(u, cmap='gray')
    #plt.title("TGV alpha=[{}, {}]".format(alpha[0], alpha[1]))
    plt.colorbar()
    plt.show()

    #plot_result(u, alpha)

    plt.figure()
    plt.plot(res)
    plt.title('Energy over Iterations', fontsize=16)
    plt.ylabel('Energy in A.U.', fontsize=14)
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.show()


# Plot result
# TODO

# Calculate Accuracy
# TODO