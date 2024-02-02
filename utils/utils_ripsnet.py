import numpy as np
from tqdm import tqdm

####################################
###  Creation of point clouds    ###
####################################

def create_circle(N_points, r, x_0, y_0):
    X = []
    for i in range(N_points):
        theta = np.random.uniform() * 2 * np.pi
        X.append([(r * np.cos(theta)) + x_0, (r * np.sin(theta) + y_0)])
    return np.array(X)


def create_1_circle_clean(N_points):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    return create_circle(N_points, r, x_0, y_0)


def create_2_circle_clean(N_points):
    r1 = 5
    r2 = 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r1 + r2:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle1 = create_circle(N_points // 2, r1, x_0, y_0)
    circle2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = [0] * N_points
    X[:N_points // 2] = circle1
    X[N_points // 2:] = circle2
    np.random.shuffle(X)
    return np.array(X)


def create_3_circle_clean(N_points):
    r0 = 5
    r1 = 3
    r2 = 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    circle0 = create_circle(N_points // 3, r0, x_0, y_0)
    circle1 = create_circle(N_points // 3, r1, x_1, y_1)
    circle2 = create_circle(N_points // 3, r2, x_2, y_2)

    # Handler in case N_points mod 3 != 0.
    true_N_points = 3 * (N_points // 3)

    X = [[0,0]] * true_N_points
    X[:true_N_points // 3] = circle0
    X[true_N_points // 3:2 * true_N_points // 3] = circle1
    X[2 * true_N_points // 3:] = circle2
    np.random.shuffle(X)
    return np.array(X)


def create_1_circle_noisy(N_points, N_noise):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    X = create_circle(N_points, r, x_0, y_0)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(x_0 - r, x_0 + r),
                      np.random.uniform(y_0 - r, y_0 + r)])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X


def create_2_circle_noisy(N_points, N_noise):
    r1 = 5
    r2 = 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r1 + r2):
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle1 = create_circle(N_points // 2, r1, x_0, y_0)
    circle2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = [0] * N_points
    X[:N_points // 2] = circle1
    X[N_points // 2:] = circle2
    np.random.shuffle(X)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(min(x_0 - r1, x_1 - r2), max(x_0 + r1, x_1 + r2)),
                      np.random.uniform(min(y_0 - r1, y_1 - r2), max(y_0 + r1, y_1 + r2))])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X


def create_3_circle_noisy(N_points, N_noise):
    r0 = 5
    r1 = 3
    r2 = 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle0 = create_circle(N_points // 3, r0, x_0, y_0)
    circle1 = create_circle(N_points // 3, r1, x_1, y_1)
    circle2 = create_circle(N_points // 3, r2, x_2, y_2)

    true_N_points = 3 * (N_points // 3)
    X = [[0,0]] * true_N_points
    X[:true_N_points // 3] = circle0
    X[true_N_points // 3:2 * true_N_points // 3] = circle1
    X[2 * true_N_points // 3:] = circle2

    np.random.shuffle(X)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(np.min([x_0 - r0, x_1 - r1, x_2 - r2]), np.max([x_0 + r0, x_1 + r1, x_2 + r2])),
                      np.random.uniform(np.min([y_0 - r0, y_1 - r1, y_2 - r2]), np.max([y_0 + r0, y_1 + r1, y_2 + r2]))])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X


def create_multiple_circles(N_sets_train, N_points, noisy=False, N_noise=0):

    data_train, PD_train = [[] for _ in range(N_sets_train)], []
    label_train = np.zeros((N_sets_train,))

    if not noisy:
        for i in tqdm(range(N_sets_train // 3)):
            data_train[i] = create_1_circle_clean(N_points)
            label_train[i] = 1
        for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
            data_train[i] = create_2_circle_clean(N_points)
            label_train[i] = 2
        for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
            data_train[i] = create_3_circle_clean(N_points)
            label_train[i] = 3
    else:
        for i in tqdm(range(N_sets_train // 3)):
            data_train[i] = create_1_circle_noisy(N_points, N_noise)
            label_train[i] = 1
        for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
            data_train[i] = create_2_circle_noisy(N_points, N_noise)
            label_train[i] = 2
        for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
            data_train[i] = create_3_circle_noisy(N_points, N_noise)
            label_train[i] = 3

    shuffler = np.random.permutation(len(data_train))
    label_train = label_train[shuffler]
    data_train = [data_train[p] for p in shuffler]

    return data_train, label_train