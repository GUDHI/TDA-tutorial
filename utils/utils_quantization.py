import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd


def _fparab(a,b,x):
    return (x - a)**2 / (2 * b) + (b / 2)


def _rotate(t):
    res = np.dot(t, np.sqrt(2)/2 * np.array([[1, 1],[-1, 1]]))
    return res


def _parab(c):
    a, b = c
    def P(x):
        return _fparab(a,b,x) # (x**2 - 2 * a * x + b**2 + a**2)/(2 * b)
    return P


def plot_partition_codebook(cs):
    xmin, xmax = -7, 3
    ymin, ymax = -2, 8

    x = np.linspace(-10, 10, 100)
    ys = np.array([_parab(c)(x) for c in cs])
    miny = np.min(ys, axis=0)

    r_cs = _rotate(cs)

    vor = Voronoi(r_cs)

    fig, ax = plt.subplots(figsize=(6,6))

    voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False)
    ax.scatter(r_cs[:,0], r_cs[:,1], marker='o', c='b')

    ax.annotate('$c_j$', r_cs[0]+[0.2,0.2], c='blue', fontsize=24)
    #ax.annotate('$c_2$', r_cs[2]+[0.2,0.2], c='green', fontsize=24)
    #ax.annotate('$V_j(\mathbf{c})$', r_cs[1]+[-0.2,1.9], c='blue', fontsize=24)
    #ax.annotate('$V_{k+1}(\mathbf{c})$', r_cs[1]+[2.,-2], c='black', fontsize=24, rotation=45)

    tmp = np.zeros((len(x), 2))
    tmp[:,0] = x
    tmp[:,1] = miny
    r_parab = _rotate(tmp)
    ax.plot(r_parab[:,0], r_parab[:,1], linestyle='dashed', color='black', linewidth=3)

    ax.set_aspect('equal')
    ax.fill_between(r_parab[:60,0],y1=r_parab[:60,0],y2=r_parab[:60,1], color='white', alpha=1, zorder=3)
    ax.fill_betweenx(r_parab[59:,1],x1=r_parab[59:,1],x2=r_parab[59:,0], color='white', alpha=1, zorder=3)
    ax.add_patch(mpatches.Polygon([[0,0], [0,r_parab[59,1]], [r_parab[59,1], r_parab[59,1]]], fill=True, color='white', alpha=1, zorder=3))
    ax.add_patch(mpatches.Polygon([[ymin,ymin], [xmax,ymin], [xmax, xmax]], fill=True, color='lightgrey', alpha=1,zorder=3))
    ax.plot([min(xmin,ymin), max(xmax,ymax)], [min(xmin,ymin), max(xmax,ymax)], color='k', linewidth=3,zorder=3)
    ax.annotate('$\partial \Omega$', [-1.3, -1.9], fontsize=24)
    #ax.annotate('$N(\mathbf{c})$', [-6.5,0], fontsize=24, rotation=45)

    ax.set_axis_off()
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin, xmax)


def plot_dgm(dgm, box=None, ax=None, color="blue", label='diagram', alpha=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if dgm.size:
        x, y = dgm[:, 0], dgm[:, 1]
        ax.scatter(x, y, marker='o', color=color, label=label, alpha=alpha)
    if box is None:
        if dgm.size:
            m, M = np.min(dgm) - 0.1, np.max(dgm) + 0.1
        else:
            m, M = 0, 1
    else:
        m, M = box
    ax.set_xlim(m, M)
    ax.set_ylim(m, M)
    ax.add_patch(mpatches.Polygon([[m, m], [M, m], [M, M]], fill=True, color='lightgrey'))
    ax.set_aspect("equal")
    ax.set_xlabel("Birth", fontsize=24)
    ax.set_ylabel("Death", fontsize=24)
    # ax.legend()


########################
### Experiment utils ###
########################


def _sample_torus(n, r1, r2, radius_eps, ambiant_eps):
    # Sample points uniformly on a torus of big radius r1 and small radius r2
    theta1 = 2 * np.pi * np.random.rand(n)
    theta2 = 2 * np.pi * np.random.rand(n)

    r1 = r1 + radius_eps * (2 * np.random.rand() - 1)
    r2 = r2 + radius_eps * (2 * np.random.rand() - 1)

    x = (r1 + r2 * np.cos(theta2)) * np.cos(theta1)
    y = (r1 + r2 * np.cos(theta2)) * np.sin(theta1)
    z = r2 * np.sin(theta2)

    X = np.array([x, y, z]).T

    X = X + ambiant_eps * (2 * np.random.rand(n,3) - 1)

    return X


def _compute_pd(X, hdim=1, min_persistence=0.0001, mode="alpha", rotate=False):
    if mode == "alpha":
        ac = gd.AlphaComplex(points=X)
        st = ac.create_simplex_tree()
    elif mode == "rips":
        ac = gd.RipsComplex(points=X)
        st = ac.create_simplex_tree(max_dimension=2)
    pers = st.persistence(min_persistence=min_persistence)
    h1 = st.persistence_intervals_in_dimension(hdim)
    if mode == "alpha":
        h1 = np.sqrt(np.array(h1))
        if rotate:
            h1[:, 1] = h1[:, 1] - h1[:, 0]
        return h1
    else:
        return np.array(h1) / 2  # to make it comparable with Cech


def build_dataset(K, params):
    average_nb_pts = params['nb_points']
    ns = np.random.poisson(lam=average_nb_pts, size=K)
    r1, r2 = params['r1'], params['r2']
    radius_eps = params['radius_eps']
    ambiant_eps = params['ambiant_eps']
    Xs = [_sample_torus(n, r1, r2, radius_eps, ambiant_eps) for n in ns]
    diags = [_compute_pd(X) for X in Xs]
    return Xs, diags


##############################
### Quantization algorithm ###
##############################




def _dist_to_diag(X, internal_p):
    return ((X[:, 1] - X[:, 0]) * 2 ** (1. / internal_p - 1))


def _build_dist_matrix(X, Y, order=2., internal_p=2):
    '''
    :param X: (n x 2) numpy.array encoding the (points of the) first diagram.
    :param Y: (m x 2) numpy.array encoding the second diagram.
    :param order: exponent for the Wasserstein metric.
    :param internal_p: Ground metric (i.e. norm L^p).
    :returns: (n+1) x (m+1) np.array encoding the cost matrix C.
                For 0 <= i < n, 0 <= j < m, C[i,j] encodes the distance between X[i] and Y[j],
                while C[i, m] (resp. C[n, j]) encodes the distance (to the p) between X[i] (resp Y[j])
                and its orthogonal projection onto the diagonal.
                note also that C[n, m] = 0  (it costs nothing to move from the diagonal to the diagonal).
    '''
    Cxd = _dist_to_diag(X, internal_p=internal_p)**order #((X[:, 1] - X[:,0]) * 2 ** (1./internal_p - 1))**order
    Cdy = _dist_to_diag(Y, internal_p=internal_p)**order #((Y[:, 1] - Y[:,0]) * 2 ** (1./internal_p - 1))**order
    if np.isinf(internal_p):
        C = sc.cdist(X, Y, metric='chebyshev') ** order
    else:
        C = sc.cdist(X, Y, metric='minkowski', p=internal_p) ** order

    Cf = np.hstack((C, Cxd[:, None]))
    Cdy = np.append(Cdy, 0)

    Cf = np.vstack((Cf, Cdy[None, :]))

    return Cf


def _get_cells(X, c, withdiag, internal_p):
    """
    X size (n x 2)
    c size (k x 2)
    withdiag: boolean
    returns: list of size k or (k+1) s.t list[j] corresponds to the points in X which are close to c[j].
                with the convention c[k] <=> the diagonal.
    """
    M = _build_dist_matrix(X, c, internal_p=internal_p) # Note: Order is useless here
    if withdiag:
        a = np.argmin(M[:-1, :], axis=1)
    else:
        a = np.argmin(M[:-1, :-1], axis=1)

    k = len(c)

    cells = [X[a == j] for j in range(k)]

    if withdiag:
        cells.append(X[a == k])  # this is the (k+1)-th centroid corresponding to the diagonal

    return cells


def _get_cost_Rk(X, c, withdiag, order, internal_p):
    cells = _get_cells(X, c, withdiag, internal_p=internal_p)
    k = len(c)

    cost = 0

    if order == np.infty and withdiag:
        for cells_j, c_j in zip(cells, c):
            if len(cells_j) == 0:
                pass
            else:
                cost_j = np.max(np.linalg.norm(cells_j - c_j, ord=internal_p, axis=1))
                cost = max(cost, cost_j)
        if len(cells[k]) == 0:
            pass
        else:
            dists_diag = _dist_to_diag(cells[k], internal_p=internal_p) #**order
            cost_diag = np.max(dists_diag)  # ** (1. / order)
            cost = max(cost, cost_diag)
        return cost

    for cells_j, c_j in zip(cells, c):
        cost_j = np.linalg.norm(np.linalg.norm(cells_j - c_j, ord=internal_p, axis=1), ord=order)**order
        cost += cost_j

    if withdiag:
        dists_to_diag = dist_to_diag(cells[k], internal_p=internal_p)**order
        cost_diag = np.sum(dists_to_diag) #** (1. / order)
        cost += cost_diag

    return cost #** (1./order)


def _from_batch(Xs, batches_indices):
    X_batch = np.concatenate([Xs[i] for i in batches_indices if Xs[i].ndim==2])
    return X_batch


def init_c(list_diags, k, internal_p=2):
    dgm = list_diags[0]
    w = _dist_to_diag(dgm, internal_p)
    s = np.argsort(w)
    c0 = dgm[s[-k:]]
    return c0


def quantization(Xs, batch_size, c0, withdiag, order=2., internal_p=2.):
    #np.random.shuffle(Xs)
    k = len(c0)
    c_current = c0.copy()
    n = len(Xs)
    batches = np.arange(0, n, dtype=int).reshape(int(n / batch_size), 2, int(batch_size / 2))

    nb_step = len(batches)

    positions = [c_current.copy()]

    for t in range(nb_step):
        X_bar_t_1 = _from_batch(Xs, batches[t, 0])
        X_bar_t_2 = _from_batch(Xs, batches[t, 1])

        cells_1 = _get_cells(X_bar_t_1, c_current, withdiag=withdiag, internal_p=internal_p)
        cells_2 = _get_cells(X_bar_t_2, c_current, withdiag=withdiag, internal_p=internal_p)

        s1, s2 = len(batches[t, 0]), len(batches[t, 1])
        if order == 2.:
            for j in range(k):
                lc1 = len(cells_1[j])
                if lc1 > 0:
                    grad = np.sum(c_current[j] - cells_2[j], axis=0) / s2
                    c_current[j] = c_current[j] - grad / ((t + 1) * len(cells_1[j]) / s1)
        else:
            raise NotImplemented('Order %s is not available yet. Only order=2. is valid in this notebook' %order)
        positions.append(c_current.copy())

    return positions


###########################
### Plot quantiz result ###
###########################
def plot_result_quantiz(diags, c_final_diag, c_final_vanilla, c0):
    low, high = -.2, 3.2

    fig, ax2 = plt.subplots(figsize=(6,6))

    for pd in diags:
        ax2.scatter(pd[:,0], pd[:,1], marker='o', c='orange', alpha=0.1)
    ax2.add_patch(mpatches.Polygon([[low,low], [high,low], [high,high]], fill=True, color='lightgrey'))

    ax2.scatter(c_final_diag[:,0], c_final_diag[:,1], marker='^', color='red', s=100,
                label='$\mathbf{c}^{\mathrm{output}}$')

    ax2.scatter(c_final_vanilla[:,0], c_final_vanilla[:,1], marker='o', color='blue', label='Output w/out diag cell')

    ax2.scatter(c0[:,0], c0[:,1], marker='x', color='black', label='initial position')
    ax2.legend(fontsize=12)
    ax2.set_xlim(low, high)
    ax2.set_ylim(low, high)
    ax2.set_xlabel('Birth',fontsize=24)
    ax2.set_ylabel('Death',fontsize=24)
    ax2.set_aspect('equal')

