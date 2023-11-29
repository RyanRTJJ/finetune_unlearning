import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from toy_model import *

def generate_data(N, k, S):
    """
    @param N:       (int) number of samples
    @param k:       (int) number of features
    @param S:       (float) sparsity: % chance that any feature x_i == 0.
    """
    data = np.random.uniform(-1.0, 1.0, size=(N, k))
    weight = np.random.uniform(0.0, 1.0, (N, k))
    data = np.where(weight < S, 0., data)
    return data

def get_indices_within_angle(arr, vec, angle):
    dot_product = np.dot(arr, vec)
    arr_magnitude = np.linalg.norm(arr, axis=1)
    vec_magnitude = np.linalg.norm(vec)
    cos_angle = dot_product / (arr_magnitude * vec_magnitude)
    indices = np.argwhere(np.arccos(cos_angle) <= np.radians(angle))
    return indices

def plot_data(data, pathological_idxs, colors=('black', 'yellow'), fig=None):
    """
    Desc :          Plots the top 2 principle components of data. 
                    If data is lower than 3D, also plots a 3D plot of the data.

                    If fig is provided, will simply scatter data onto the same fig.
    """
    non_pathological_idxs = np.setdiff1d(np.arange(data.shape[0]), pathological_idxs)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    if fig == None:
        if data.shape[1] > 3:
            fig = plt.figure(figsize=(2, 2))
        else:
            fig = plt.figure(figsize=(4, 2))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
    else:
        if data.shape[1] <= 3:
            ax1 = fig.get_axes()[0]
            ax2 = fig.get_axes()[1]

    if data.shape[1] <= 3:
        if data.shape[1] == 2:
            third_axis = np.array([0] * data.shape[0])
        else:
            third_axis = data[:,2]

        ax1.scatter(data[non_pathological_idxs, 0], data[non_pathological_idxs, 1], third_axis[non_pathological_idxs], color=colors[0], alpha=0.5)
        ax1.scatter(data[pathological_idxs, 0], data[pathological_idxs, 1], third_axis[pathological_idxs], color=colors[1], alpha=0.5)

        ax2.scatter(data_pca[non_pathological_idxs, 0], data_pca[non_pathological_idxs, 1], color=colors[0], alpha=0.5)
        ax2.scatter(data_pca[pathological_idxs, 0], data_pca[pathological_idxs, 1], color=colors[1], alpha=0.5)
    else:
        plt.scatter(data_pca[non_pathological_idxs, 0], data_pca[non_pathological_idxs, 1], color=colors[0], alpha=0.5)
        plt.scatter(data_pca[pathological_idxs, 0], data_pca[pathological_idxs, 1], color=colors[1], alpha=0.5)
    
    return fig

if __name__ == '__main__':
    # args
    N = 1000
    k = 2
    sparsity = 0.75
    bottleneck_dim = 2
    # np.random.seed(420)
    tensorboard_logdir = "train_outs/debug"
    num_epochs = 300

    data = generate_data(N, k, sparsity)
    pathological_vector = np.array([1] * k)
    pathological_idxs = get_indices_within_angle(data, pathological_vector, 45)

    # plot_data(data, pathological_idxs)

    model = ToyModel(k, bottleneck_dim)
    model.init_weights()
    train(model, data, num_epochs, tensorboard_logdir)