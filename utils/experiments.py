import os.path
import itertools
import h5py
import numpy              as np
import matplotlib.pyplot  as plt
import pandas             as pd
import tensorflow         as tf
import tensorflow_addons  as tfa
import gudhi              as gd

from scipy.sparse           import csgraph
from scipy.io               import loadmat
from scipy.linalg           import eigh
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
from tensorflow             import random_uniform_initializer as rui
from gudhi.representations  import PerslayModel


def get_parameters(dataset):
    if dataset == "MUTAG" or dataset == "PROTEINS":
        dataset_parameters = {"data_type": "graph", "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "COX2" or dataset == "DHFR" or dataset == "NCI1" or dataset == "NCI109" or dataset == "IMDB-BINARY" or dataset == "IMDB-MULTI":
        dataset_parameters = {"data_type": "graph", "filt_names": ["Ord0_0.1-hks", "Rel1_0.1-hks", "Ext0_0.1-hks", "Ext1_0.1-hks", "Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "ORBIT5K" or dataset == "ORBIT100K":
        dataset_parameters = {"data_type": "orbit", "filt_names": ["Alpha0", "Alpha1"]}
    return dataset_parameters

def get_model(dataset):

    if dataset == "MUTAG":

        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (10, 10)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "Image"
        plp["image_size"]     = (20, 20)
        plp["image_bnds"]     = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = rui(3., 3.)
        plp["layer_train"]    = True
        plp["perm_op"]        = "sum"
        perslay_parameters    = [plp for _ in range(4)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            for i in range(4):
                fmodel = tf.keras.Sequential([tf.keras.layers.Conv2D(10, 2, input_shape=(21,21,1)), tf.keras.layers.Flatten()])
                perslay_parameters[i]["final_model"] = fmodel
            rho = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(16039,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    elif dataset == "PROTEINS":

        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (10, 10)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "Image"
        plp["image_size"]     = (15, 15)
        plp["image_bnds"]     = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = rui(3., 3.)
        plp["layer_train"]    = True
        plp["perm_op"]        = "sum"
        perslay_parameters    = [plp for _ in range(4)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            for i in range(4):
                fmodel = tf.keras.Sequential([tf.keras.layers.Conv2D(10, 2, input_shape=(16,16,1)), tf.keras.layers.Flatten()])
                perslay_parameters[i]["final_model"] = fmodel
            rho = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(9039,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    elif dataset == "NCI1" or dataset == "NCI109":

        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (10, 10)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "PermutationEquivariant"
        plp["lpeq"]           = [(25, None), (25, "max")]
        plp["layer_train"]    = True
        plp["perm_op"]        = "sum"
        plp["final_model"]    = "identity"
        perslay_parameters    = [plp for _ in range(8)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            rho = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(239,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    elif dataset == "IMDB-MULTI" or dataset == "IMDB-BINARY":

        nlab = 2 if dataset == "IMDB-BINARY" else 3
        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (20, 20)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "Image"
        plp["image_size"]     = (20, 20)
        plp["image_bnds"]     = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = rui(3., 3.)
        plp["layer_train"]    = True
        plp["perm_op"]        = "sum"
        perslay_parameters    = [plp for _ in range(8)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            for i in range(8):
                fmodel = tf.keras.Sequential([tf.keras.layers.Conv2D(10, 2, input_shape=(21,21,1)), tf.keras.layers.Flatten()])
                perslay_parameters[i]["final_model"] = fmodel
            rho = tf.keras.Sequential([tf.keras.layers.Dense(nlab, activation="sigmoid", input_shape=(32039,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    elif dataset == "COX2" or dataset == "DHFR":

        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (10, 10)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "Image"
        plp["image_size"]     = (20, 20)
        plp["image_bnds"]     = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = rui(3., 3.)
        plp["layer_train"]    = True
        plp["perm_op"]        = "sum"
        perslay_parameters    = [plp for _ in range(8)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            for i in range(8):
                fmodel = tf.keras.Sequential([tf.keras.layers.Conv2D(10, 2, input_shape=(21,21,1)), tf.keras.layers.Flatten()])
                perslay_parameters[i]["final_model"] = fmodel
            rho = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(32039,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    elif dataset == "ORBIT5K" or dataset == "ORBIT100K":

        plp = {}
        plp["pweight"]        = "grid"
        plp["pweight_init"]   = rui(1., 1.)
        plp["pweight_size"]   = (10, 10)
        plp["pweight_bnds"]   = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"]  = True
        plp["layer"]          = "PermutationEquivariant"
        plp["lpeq"]           = [(25, None), (25, "max")]
        plp["lweight_init"]   = rui(0.,1.)
        plp["lbias_init"]     = rui(0.,1.)
        plp["lgamma_init"]    = rui(0.,1.)
        plp["layer_train"]    = True
        plp["perm_op"]        = "topk"
        plp["keep"]           = 5
        plp["final_model"]    = "identity"
        perslay_parameters    = [plp for _ in range(2)]

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            rho = tf.keras.Sequential([tf.keras.layers.Dense(5, activation="sigmoid", input_shape=(250,))])
            model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=1., staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

    return model, optimizer, loss, metrics

def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)

def generate_orbit(num_pts_per_orbit, param):
    X = np.zeros([num_pts_per_orbit, 2])
    xcur, ycur = np.random.rand(), np.random.rand()
    for idx in range(num_pts_per_orbit):
        xcur = (xcur + param * ycur * (1. - ycur)) % 1
        ycur = (ycur + param * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]
    return X

def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):        
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0,2])
    dgmRel1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0,2])
    dgmExt0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0,2])
    dgmExt1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0,2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def generate_diagrams_and_features(dataset, path_dataset=""):

    dataset_parameters = get_parameters(dataset)
    dataset_type = dataset_parameters["data_type"]

    if "REDDIT" in dataset:
        print("Unfortunately, REDDIT data are not available yet for memory issues.\n")
        print("Moreover, the link we used to download the data,")
        print("http://www.mit.edu/~pinary/kdd/datasets.tar.gz")
        print("is down at the commit time (May 23rd).")
        print("We will update this repository when we figure out a workaround.")
        return

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5", "w")
    list_filtrations = dataset_parameters["filt_names"]
    [diag_file.create_group(str(filtration)) for filtration in dataset_parameters["filt_names"]]
    
    if dataset_type == "graph":

        list_hks_times = np.unique([filtration.split("_")[1] for filtration in list_filtrations])

        # preprocessing
        pad_size = 1
        for graph_name in os.listdir(path_dataset + "mat/"):
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            pad_size = np.max((A.shape[0], pad_size))

        feature_names = ["eval"+str(i) for i in range(pad_size)] + [name+"-percent"+str(i) for name, i in itertools.product([f for f in list_hks_times if "hks" in f], 10*np.arange(11))]
        features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))), columns=["label"] + feature_names)

        for idx, graph_name in enumerate((os.listdir(path_dataset + "mat/"))):

            name = graph_name.split("_")
            gid = int(name[name.index("gid") + 1]) - 1
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            num_vertices = A.shape[0]
            label = int(name[name.index("lb") + 1])

            L = csgraph.laplacian(A, normed=True)
            egvals, egvectors = eigh(L)
            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for fhks in list_hks_times:
                hks_time = float(fhks.split("-")[0])
                filtration_val = hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val)
                diag_file["Ord0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmOrd0)
                diag_file["Ext0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt0)
                diag_file["Rel1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmRel1)
                diag_file["Ext1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt1)
                graph_features.append(np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time), 10 * np.arange(11)))
            features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)
        features["label"] = features["label"].astype(int)

    elif dataset_type == "orbit":

        labs = []
        count = 0
        num_diag_per_param = 1000 if "5K" in dataset else 20000
        for lab, r in enumerate([2.5, 3.5, 4.0, 4.1, 4.3]):
            print("Generating", num_diag_per_param, "orbits and diagrams for r = ", r, "...")
            for dg in range(num_diag_per_param):
                X = generate_orbit(num_pts_per_orbit=1000, param=r)
                alpha_complex = gd.AlphaComplex(points=X)
                st = alpha_complex.create_simplex_tree(max_alpha_square=1e50)
                st.persistence()
                diag_file["Alpha0"].create_dataset(name=str(count), data=np.array(st.persistence_intervals_in_dimension(0)))
                diag_file["Alpha1"].create_dataset(name=str(count), data=np.array(st.persistence_intervals_in_dimension(1)))
                orbit_label = {"label": lab, "pcid": count}
                labs.append(orbit_label)
                count += 1
        labels = pd.DataFrame(labs)
        labels.set_index("pcid")
        features = labels[["label"]]

    features.to_csv(path_dataset + dataset + ".csv")

    return diag_file.close()

def load_data(dataset, path_dataset="", filtrations=[], verbose=False):

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys()) if len(filtrations) == 0 else filtrations

    diags_dict = dict()
    if len(filts) == 0:
        filts = diagfile.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diagfile[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diagfile[filtration][str(diag)]))
        diags_dict[filtration] = list_dgm

    # Extract features and encode labels with integers
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    F = np.array(feat)[:, 1:]  # 1: removes the labels
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diags_dict, F, L

def visualize_diagrams(diags_dict, ilist=(0, 10, 20, 30, 40, 50)):
    filts = diags_dict.keys()
    n, m = len(filts), len(ilist)
    fig, axs = plt.subplots(n, m, figsize=(m*n / 2, n*m / 2))
    for (i, filtration) in enumerate(filts):
        for (j, idx) in enumerate(ilist):
            xs, ys = diags_dict[filtration][idx][:, 0], diags_dict[filtration][idx][:, 1]
            axs[i, j].scatter(xs, ys)
            axs[i, j].plot([0, 1], [0, 1])
            axs[i, j].axis([0, 1, 0, 1])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    # axis plot
    cols = ["idx = " + str(i) for i in ilist]
    rows = filts
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large")
    plt.show()
    return

def evaluate_model(L, F, D, train_sub, test_sub, model, optimizer, loss, metrics, num_epochs, batch_size=128, verbose=1, plots=False):

    num_pts, num_labels, num_features, num_filt = L.shape[0], L.shape[1], F.shape[1], len(D)

    train_num_pts, test_num_pts = len(train_sub), len(test_sub)
    label_train, label_test = L[train_sub, :], L[test_sub, :]
    feats_train, feats_test = F[train_sub, :], F[test_sub, :]
    diags_train, diags_test = [D[dt][train_sub, :] for dt in range(num_filt)], [D[dt][test_sub, :] for dt in range(num_filt)]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(x=[diags_train, feats_train], y=label_train, validation_data=([diags_test, feats_test], label_test), epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
    train_results = model.evaluate([diags_train, feats_train], label_train, verbose=verbose)
    test_results = model.evaluate([diags_test,  feats_test],  label_test, verbose=verbose)
    
    if plots:
        ltrain, ltest = history.history["categorical_accuracy"], history.history["val_categorical_accuracy"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(ltrain), color="blue", label="train acc")
        ax.plot(np.array(ltest),  color="red",  label="test acc")
        ax.set_ylim(top=1.)
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel("classif. accuracy")
        ax.set_title("Evolution of train/test accuracy")
        plt.show()

    return history.history, train_results, test_results

