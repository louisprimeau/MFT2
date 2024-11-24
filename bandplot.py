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

def band_plot(fig, ax, syst, point_names, N, **kwargs):
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
        ax.plot(np.linspace(0, xpos[-1], ks.shape[0]), bands[:, band_index], **kwargs)

    #ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(point_names)

    return fig, ax

def bandfold(old_syst, n_x, n_y, lattice_vectors_old, x_coordinates_old, old_lat):

    n_sublattices_old = lattice_vectors_old.shape[1]
    lattice_vector_x = lattice_vectors_old[:, 0] * n_x
    lattice_vector_y = lattice_vectors_old[:, 1] * n_y
    n_sublattices = x_coordinates_old.shape[1] * n_x * n_y
    lattice_vectors = np.vstack((lattice_vector_x, lattice_vector_y))
    x_coordinates = []
    for i in range(n_x):
        for j in range(n_y):
            for vec in x_coordinates_old:
                x_coordinates.append(vec + i * (lattice_vector_x / n_x) + j * (lattice_vector_y / n_y))
    x_coordinates = np.array(x_coordinates)
    n_orbs = x_coordinates.shape[0] * old_syst[list(old_syst.sites())[0]].shape[0]
    lat = kwant.lattice.general(lattice_vectors, norbs=n_orbs, basis=x_coordinates)
    sym = kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1)))
    new_syst = kwant.Builder(sym)

    supercell_hoppings = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    # in unit cell
    for s in range(n_x * n_y * n_sublattices_old):
        l_s = s % 2
        W_s = (s // 2) % n_x
        L_s = ((s // 2) - W_s) // n_x 
        W_s += n_x; L_s += n_y
        new_syst[lat.sublattices[s](0,0)] = old_syst[old_lat.sublattices[l_s](W_s, L_s)]

    for s in range(n_x * n_y * n_sublattices_old):
        for t in range(n_x * n_y * n_sublattices_old):
            if s == t: continue
            l_s = s % 2; W_s = (s // 2) % n_x; L_s = ((s // 2) - W_s) // n_x
            W_s += n_x; L_s += n_y
            l_t = t % 2; W_t = (t // 2) % n_x; L_t = ((t // 2) - W_t) // n_x
            W_t += n_x; L_t += n_y
            try:
                new_syst[lat.sublattices[s](0,0), lat.sublattices[t](0,0)] = old_syst[old_lat.sublattices[l_s](W_s,L_s), old_lat.sublattices[l_t](W_t,L_t)]
            except KeyError:
                continue
            
    for s in range(n_x * n_y * n_sublattices_old):
        for t in range(n_x * n_y * n_sublattices_old):
            for h_x, h_y in supercell_hoppings:
                l_s = s % 2; W_s = (s // 2) % n_x; L_s = ((s // 2) - W_s) // n_x
                W_s += n_x; L_s += n_y
                l_t = t % 2; W_t = (t // 2) % n_x; L_t = ((t // 2) - W_t) // n_x
                W_t += (h_x+1 ) * n_x; L_t += (h_y + 1) * n_y
                try:
                    new_syst[lat.sublattices[s](0,0), lat.sublattices[t](h_x,h_y)] = old_syst[old_lat.sublattices[l_s](W_s,L_s), old_lat.sublattices[l_t](W_t,L_t)]
                except KeyError:
                    continue
    
    return new_syst
