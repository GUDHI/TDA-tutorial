import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import gudhi as gd

def sample_circle(n):
    theta = 2 * np.pi * np.random.rand(n)
    x, y = np.cos(theta), np.sin(theta)
    return np.array([x, y]).T


def sample_torus(n, r1, r2):
    theta1 = 2 * np.pi * np.random.rand(n)
    theta2 = 2 * np.pi * np.random.rand(n)

    x = (r1 + r2 * np.cos(theta2)) * np.cos(theta1)
    y = (r1 + r2 * np.cos(theta2)) * np.sin(theta1)
    z = r2 * np.sin(theta2)

    X = np.array([x, y, z]).T

    return X


def sample_noise(N, ndim, scale=1., type="uniform"):
    '''
    noise sample uniforme on [-scale, scale]^2 or gaussien N(0, scale^2)
    '''
    if type=="uniform":
        X_noise = scale * (2 * np.random.rand(N, ndim) - 1)
    elif type=="gaussian":
        X_noise = scale * np.random.randn(N, ndim)
    return X_noise


def add_noise(X_noise, X):
    return np.concatenate([X, X_noise])


# Compute persistence diagram using the Cech filtration
def alphacomplex(X, hdim=1):
    ac = gd.AlphaComplex(points=X)
    st = ac.create_simplex_tree()
    pers = st.persistence(min_persistence=0.0001)
    h1 = st.persistence_intervals_in_dimension(hdim)
    return np.sqrt(np.array(h1))  # we must sqrt to get Cech dgm


def expected_dgm(X, n, k, hdim=1, replace=True):
    '''
    Subsample n points in a point cloud X, k times, and return the k computed persistence diagrams.
    '''
    N = len(X)  # total nb of points
    samples = [np.random.choice(N, n, replace=replace) for _ in range(k)]
    Xs = [X[sample] for sample in samples]
    diags = [alphacomplex(X, hdim=hdim) for X in Xs]
    return diags


# Discretization utils to plot expected PD in a nice way.

def discretize_dgm(diag, m=0, M=1, res=30, expo=2., filt=None, sigma=1.):
    '''
    Discretize one diagram, returns a 2D-histogram of size (res x res), the grid representing [m, M]^2.
    '''
    if diag.size:
        h2d = np.histogram2d(diag[:,0], diag[:,1], bins=res, range=[[m, M],[m, M]])[0]
        h2d = np.flip(h2d.T, axis=0)
        if filt=="gaussian":
            h2d = gaussian_filter(h2d, sigma=sigma)
        return h2d
    else:
        return np.zeros((res, res))

def tohist(diags, m=0, M=1, res=30, expo=2.,filt=None, sigma=1.):
    '''
    Take a list of observed diagrams 'diags' and return an histogram
    which is an estimator of E(Dgm(X_n)) for some integer n
    '''

    output = [discretize_dgm(diag, m=m, M=M, res=res, expo=expo, filt=filt, sigma=sigma)
             for diag in diags]
    return np.mean(output, axis=0)


def plot3d(X, X_noise):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', color='blue')
    ax.scatter(X_noise[:,0], X_noise[:,1], X_noise[:,2], marker='x', color='red')

def plot2d(X, X_noise):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], marker='x', color='blue')
    ax.scatter(X_noise[:,0], X_noise[:,1], marker='x', color='red')
    ax.set_aspect("equal")


def plot_object(X, X_noise):
    if X.ndim == 2:
        plot2d(X, X_noise)
    elif X.ndim == 3:
        plot3d(X, X_noise)
    else:
        print("Unknown number of dimension. Input should be a 2D or a 3D point cloud.")


def plot_hist(h, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(h, cmap='hot_r', interpolation='bilinear')
    L = h.shape[0]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Births", fontsize=18)
    ax.set_ylabel("Deaths", fontsize=18)
    ax.add_patch(Polygon([[L,-1], [L,L], [-1,L]], fill=True, color='lightgrey'))
    ax.set_title("Expected PD", fontsize=24)

def plot_dgm(dgm_true, dgm_tot, m, M, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y = dgm_true[:,0], dgm_true[:,1]
    ax.scatter(x, y, marker='o', color='blue', label='true dgm')
    ax.scatter(dgm_tot[:,0], dgm_tot[:,1], marker='x', color='red', label='total dgm')
    ax.set_xlim(m,M)
    ax.set_ylim(m,M)
    ax.add_patch(Polygon([[m, m], [M, m], [M, M]], fill=True, color='lightgrey'))
    ax.set_aspect("equal")
    ax.set_xlabel("Births", fontsize=18)
    ax.set_ylabel("Deaths", fontsize=18)

    ax.legend()
    ax.set_title("Diagrams", fontsize=24)



