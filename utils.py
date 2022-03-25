import numpy as np
import random
import os
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def dataloader(mode):
    file = open("data/two_spiral_{}_data.txt".format(mode), 'r')
    res = []
    for line in file.readlines():
        res.append([np.array([float(line.split()[0]), float(line.split()[1])]), int(line.split()[2])])
    file.close()
    return np.array(res)

def plot_decision_boundary(model, X, y, filename):
    # Set min and max values and give it some padding
    x_min, x_max = -7, 7
    y_min, y_max = -7, 7
    h = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for i in data:
        tmp = model(i)
        Z.append(tmp)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.savefig(filename)

def partition_data(data, mode):
    x, y = data[:, 0], data[:, 1]
    x_one, y_one = x[np.where(y == 1)], y[np.where(y == 1)]
    x_zero, y_zero = x[np.where(y == 0)], y[np.where(y == 0)]
    one_set, zero_set = [[], []], [[], []]

    if mode == 'random':
        for i, j in enumerate(np.random.choice(2, len(x_one))):
            one_set[j].append((x_one[i], y_one[i]))
        for i, j in enumerate(np.random.choice(2, len(x_zero))):
            zero_set[j].append((x_zero[i], y_zero[i]))
    elif mode == 'yaxis':
        for i in range(len(x_one)):
            if x_one[i][0] >= -0.3:
                one_set[1].append((x_one[i], y_one[i]))
            elif x_one[i][0] <= 0.3:
                one_set[0].append((x_one[i], y_one[i]))
        for i in range(len(x_zero)):
            if x_zero[i][0] >= -0.3:
                zero_set[1].append((x_zero[i], y_zero[i]))
            elif x_zero[i][0] <= 0.3:
                zero_set[0].append((x_zero[i], y_zero[i]))
    elif mode == 'radius':
        for i in range(len(x_one)):
            if x_one[i][0] ** 2 + x_one[i][1] ** 2 >= 8:
                one_set[1].append((x_one[i], y_one[i]))
            elif x_one[i][0] ** 2 + x_one[i][1] ** 2 <= 10:
                one_set[0].append((x_one[i], y_one[i]))
        for i in range(len(x_zero)):
            if x_zero[i][0] ** 2 + x_zero[i][1] ** 2 >= 8:
                zero_set[1].append((x_zero[i], y_zero[i]))
            elif x_zero[i][0] ** 2 + x_zero[i][1] ** 2 <= 10:
                zero_set[0].append((x_zero[i], y_zero[i]))

    return one_set, zero_set