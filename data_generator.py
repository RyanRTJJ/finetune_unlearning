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
    data = np.random.uniform(0.0, 1.0, size=(N, k))
    weight = np.random.uniform(0.0, 1.0, (N, k))
    data = np.where(weight < S, 1., data)
    return data

def get_indices_within_angle(arr, vec, angle):
    dot_product = np.dot(arr, vec)
    arr_magnitude = np.linalg.norm(arr, axis=1)
    vec_magnitude = np.linalg.norm(vec)
    cos_angle = dot_product / (arr_magnitude * vec_magnitude)
    indices = np.argwhere(np.arccos(cos_angle) <= np.radians(angle))
    return indices

def plot_data(data, pathological_idxs, colors=('black', 'yellow'), alpha=0.2, fig=None, PCA_M=None, PCA_mu=None):
    """
    Desc :          Plots the top 2 principle components of data. 
                    If data is lower than 3D, also plots a 3D plot of the data.

                    If fig is provided, will simply scatter data onto the same fig.
    
    @param data:                    (np.ndarray) of datapoints we would like to plot. Could be original; could be reconstructed.
    @param pathological_idxs:       (np.ndarray) of indices that are near the pathological vector's direction
    @param colors:                  (Tuple[str]) colors to plot [0]: non-pathological, [1]: pathological
    @param fig:                     (plt.fig)
    @param PCA_M:                   (np.ndarray) a PCA rotation matrix. If given, will not do pca.fit_transform, but will
                                    simply apply PCA_M to the data.
    @param PCA_mu:                  (np.ndarray) a mean vector to subtract from every data point.
    """
    non_pathological_idxs = np.setdiff1d(np.arange(data.shape[0]), pathological_idxs)

    if not isinstance(PCA_M, np.ndarray):
        pca = PCA()
        PCA_mu = np.mean(data, axis=0)
        data_normalized = data - PCA_mu
        data_pca = pca.fit_transform(data_normalized)
        PCA_M = pca.components_
    else:
        data_pca = (data - PCA_mu) @ PCA_M.T

    # Initialize the figure object
    if fig == None:
        if data.shape[1] > 3:
            fig = plt.figure(figsize=(3, 3))
        else:
            fig = plt.figure(figsize=(6, 3))
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

        ax1.scatter(data[non_pathological_idxs, 0], data[non_pathological_idxs, 1], third_axis[non_pathological_idxs], color=colors[0], alpha=alpha, s=3)
        ax1.scatter(data[pathological_idxs, 0], data[pathological_idxs, 1], third_axis[pathological_idxs], color=colors[1], alpha=alpha, s=3)

        ax2.scatter(data_pca[non_pathological_idxs, 0], data_pca[non_pathological_idxs, 1], color=colors[0], alpha=alpha, s=3)
        ax2.scatter(data_pca[pathological_idxs, 0], data_pca[pathological_idxs, 1], color=colors[1], alpha=alpha, s=3)
    else:
        plt.scatter(data_pca[non_pathological_idxs, 0], data_pca[non_pathological_idxs, 1], color=colors[0], alpha=alpha, s=3)
        plt.scatter(data_pca[pathological_idxs, 0], data_pca[pathological_idxs, 1], color=colors[1], alpha=alpha, s=3)
    
    return fig, PCA_M, PCA_mu