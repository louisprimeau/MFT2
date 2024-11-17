import numpy as np
from numpy import pi
import os
import kwant
import matplotlib.pyplot as plt
import quasinewton
import scipy
import itertools
import time
import copy

def make_k_path(points, N):
    points = [p.reshape(1, 2) for p in points]
    distances = [np.linalg.norm(a - b) for a, b in zip(points[1:], points[:-1])]
    segments = []
    for a, b, d in zip(points[1:], points[:-1], distances):
        t = np.linspace(0, 1, int(N * d / sum(distances)) + 1)[:-1].reshape(-1, 1)
        segments.append(a * t + b * (1 - t))
    return np.concatenate(segments + [np.array(points[-1]).reshape(1, 2)], axis=0)

def get_bands(syst, ks):
    bands = []
    for k in ks:
        H = syst.hamiltonian_submatrix(params={"k_x": k[0], "k_y": k[1]})
        bands.append(np.linalg.eigvalsh(H).reshape(1, -1))
    return np.concatenate(bands, axis=0)

def band_plot(fig, ax, syst, point_names, N):
    labelled_points = ['G', 'X', 'Y', 'R']
    G = np.array([0.0, 0.0]) * 2 * pi
    R = np.array([0.5, 0.5]) * 2 * pi
    X = np.array([0.5, 0.0]) * 2 * pi
    Y = np.array([0.0, 0.5]) * 2 * pi

    points = []
    for point in point_names:
        if point == 'G': points.append(G)
        elif point == 'X': points.append(X)
        elif point == 'Y': points.append(Y)
        elif point == 'R': points.append(R)
        else: raise NotImplementedError
    
    ks = make_k_path(points, N)
    xpos = [0] + list(np.cumsum([np.linalg.norm(a - b) for a, b in zip(points[1:], points[:-1])]))
    bands = get_bands(syst, ks)

    for band_index in range(bands.shape[1]):
        ax.plot(np.linspace(0, xpos[-1], ks.shape[0]), bands[:, band_index], color='r')

    #ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(point_names)

    return fig, ax
