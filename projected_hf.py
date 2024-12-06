import numpy as np
from numpy import pi
import os
import kwant
import matplotlib.pyplot as plt
import quasinewton
from bandplot import *
from vaspread import *
from kwanttools import *
import scipy
import itertools
import time
import copy

def add_interaction(syst, correlation, n_orbs, lattice_vectors, W_size, H_size, hopping_range=0):

    new_syst = copy.deepcopy(syst)
    sites = list(syst.sites())
    
    As, ds = compute_adjacency(syst, n_orbs, hopping_range, periods=(lattice_vectors[0,0] * W_size, lattice_vectors[1,1] * H_size))

    A = coarsen(As[0], n_orbs)
    for i, j in zip(*np.nonzero(A)):
        assert i == j
        interaction = onsite_interactions_120orbs(correlation[i * n_orbs:(i+1)* n_orbs, j * n_orbs:(j+1)*n_orbs])

        #new_syst[sites[i]] += projectors_Rs[:, :, onsite_index] @ interaction @ projectors_Rs[:, :, onsite_index]
        new_syst[sites[i]] = new_syst[sites[i]] + projectors_Rs[:, :, onsite_index] @ interaction @ projectors_Rs[:, :, onsite_index]
    return new_syst
    """
    for nn in range(1, hopping_range+1):
        A = coarsen(As[nn], n_orbs)
        for i, j in zip(*np.nonzero(A)):
            try:
                syst[(sites[i], sites[j])] += UL * -correlation[j * n_orbs:(j+1)* n_orbs, i * n_orbs:(i+1)*n_orbs].T / ds[nn]
            except KeyError:
                syst[(sites[i], sites[j])] = UL * -correlation[i * n_orbs:(i+1)* n_orbs, j * n_orbs:(j+1)*n_orbs].T / ds[nn]
        syst[sites[i]] += UL * np.diag(np.diag(correlation)[j*n_orbs:(j+1)*n_orbs]) / ds[nn]
    """
    return new_syst

def onsite_interactions_8orbs(correlation):

    interaction = np.zeros_like(correlation)

    interaction[0, 0] += U * correlation[4, 4]
    interaction[1, 1] += U * correlation[5, 5]
    interaction[2, 2] += U * correlation[6, 6]
    interaction[3, 3] += U * correlation[7, 7]

    interaction[4, 4] += U * correlation[0, 0]
    interaction[5, 5] += U * correlation[1, 1]
    interaction[6, 6] += U * correlation[2, 2]
    interaction[7, 7] += U * correlation[3, 3]

    interaction[0, 4] += -U * correlation[4, 0]
    interaction[1, 5] += -U * correlation[5, 1]
    interaction[2, 6] += -U * correlation[6, 2]
    interaction[3, 7] += -U * correlation[7, 3]

    interaction[4, 0] += U * correlation[0, 4]
    interaction[5, 1] += U * correlation[1, 5]
    interaction[6, 2] += U * correlation[2, 6]
    interaction[7, 3] += U * correlation[3, 7]

    return interaction


def onsite_interactions_120orbs(correlation):

    interaction = np.zeros_like(correlation)

    for i in range(0, 15, 1):
        s = i * 8
        interaction[s + 0, s + 0] += U * correlation[s + 4, s + 4]
        interaction[s + 1, s + 1] += U * correlation[s + 5, s + 5]
        interaction[s + 2, s + 2] += U * correlation[s + 6, s + 6]
        interaction[s + 3, s + 3] += U * correlation[s + 7, s + 7]

        interaction[s+4, 4+s] += U * correlation[s+0, 0+s]
        interaction[s+5, 5+s] += U * correlation[s+1, 1+s]
        interaction[s+6, 6+s] += U * correlation[s+2, 2+s]
        interaction[s+7, 7+s] += U * correlation[s+3, 3+s]

        interaction[s+0, 4+s] += -U * correlation[s+4, 0+s]
        interaction[s+1, 5+s] += -U * correlation[s+5, 1+s]
        interaction[s+2, 6+s] += -U * correlation[s+6, 2+s]
        interaction[s+3, 7+s] += -U * correlation[s+7, 3+s]

        interaction[s+4, 0+s] += U * correlation[s+0, 4+s]
        interaction[s+5, 1+s] += U * correlation[s+1, 5+s]
        interaction[s+6, 2+s] += U * correlation[s+2, 6+s]
        interaction[s+7, 3+s] += U * correlation[s+3, 7+s]

    return interaction



def compute_adjacency(syst, n_orbs, N, periods=None):
    positions = np.array([site.pos for site in syst.sites()]).repeat(n_orbs, axis=0)
    x, y = positions[:, 0], positions[:, 1]
    dmatrix = np.sqrt(np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)
    if periods is not None:
        x_period = periods[0]
        y_period = periods[1]
        for translation in itertools.product([-1, 0, 1], [-1, 0, 1]):
            dmatrix_new = np.sqrt(np.subtract.outer(x - x_period * translation[0], x)**2 + np.subtract.outer(y - y_period * translation[1], y)**2)
            dmatrix = np.minimum(dmatrix, dmatrix_new)
    distances = np.sort(np.unique(dmatrix.round(decimals=8)))
    As = []
    for i in range(0, N+1):
        As.append((np.abs(dmatrix - distances[i]) < 1e-6).astype(float))
    return As, distances[:N+1]

def coarsen(A, n):
    N, M = A.shape[0] // n, A.shape[1] // n
    return np.mean(A.reshape(N, n, M, n), axis=(1,3))

def compute_correlation(eigs, v, mu=0):
    mask = (eigs < mu).astype(np.complex128)
    all_correlation = v.conj() @ np.diag(mask) @ v.T
    return all_correlation

def compute_EF(eigs):
    return np.sort(eigs)[len(eigs)//2 + 2]

def cdw_density(syst, period):
    positions = np.array([site.pos for site in syst.sites()]).repeat(n_orbs, axis=0)
    filling = W * L * n_orbs // 2 + int(W * L / 15)
    oscillation = 1 + np.cos(2 * pi * positions[:, 0] / period)
    oscillation = oscillation * filling / np.sum(oscillation)
    return np.diag(oscillation)

def projectors(tbengine, N):
    k_x, k_y = np.meshgrid(np.linspace(-np.pi, np.pi, N+1)[:-1], np.linspace(-np.pi, np.pi, N+1)[:-1])
    ks = np.concatenate([k_x.reshape(-1, 1), k_y.reshape(-1, 1), np.zeros_like(k_x.reshape(-1, 1))], axis=1)
    Hk = tbengine.spool_hamiltonian(ks)
    print(Hk.shape)
    es, vs = np.linalg.eigh(Hk)
    projectors_k = np.zeros((es.shape[0], es.shape[1], es.shape[1]))

    which_bands = np.zeros((120,))
    which_bands[60] = 1
    which_bands[61] = 1


    print("done eigenvalue decomp.")
    projectors_k[:, np.arange(es.shape[1]), np.arange(es.shape[1])] = which_bands
    projectors_k = vs.conj() @ projectors_k @ vs.transpose(0, 2, 1)
    projectors_Rs = tbengine.project_hoppings(ks, np.einsum('ilm,imn,inp->ilp', projectors_k, Hk, projectors_k, optimize='greedy'))
    onsite_index = int(np.nonzero(np.all(tbengine.all_Rs == 0, axis=1))[0][0])
    print("done projection")
    return projectors_Rs, onsite_index

def objective_function(correlation):
    correlation = correlation.reshape(W*L*n_sl*n_orbs, W*L*n_sl*n_orbs)
    syst = add_interaction(bare_syst, correlation, n_orbs, lattice_vectors, W, L, hopping_range=1)
    H = syst.finalized().hamiltonian_submatrix(sparse=False)

    eigs, eigv = np.linalg.eigh(H)
    EF = compute_EF(eigs)
    new_correlation = compute_correlation(eigs, eigv, mu=EF).reshape(-1)
    return new_correlation

# HARTREE FOCK LOOP
W, L = 3, 3
U = 1.0 * 2

res = make_system_from_vasp_pg("wannier90_cdw_hr.dat",
                               "POSCAR-15.vasp",
                               W,
                               L,
                               periodic=True,
                               translationally_invariant=False,
                               gauge='periodic')
bare_syst, n_orbs, n_sl, lattice_vectors, x_coordinates, lat = res
tbengine = TBEngine("wannier90_cdw_hr.dat", "POSCAR-15.vasp")
print("fourier projection...")
projectors_Rs, onsite_index = projectors(tbengine, 20)
init_correlation = np.eye(n_sl * n_orbs * W * L).astype(np.complex128) * (W * L * n_orbs * n_sl // 2) / (n_orbs * W * L * n_sl)
print("starting optimization...")
correlation = quasinewton.quasi_newton(objective_function, init_correlation.reshape(-1)).reshape(W * L * n_sl * n_orbs, W * L * n_sl * n_orbs)

correlated_folded_syst = add_interaction(bare_syst, correlation, n_orbs, lattice_vectors, W, L)
#correlated_folded_syst = bandfold(correlated_syst, 15, 1, lattice_vectors, x_coordinates, lat)
bare_folded_syst = bare_syst


fig, ax = plt.subplots(1)

fig, ax = band_plot(fig, ax, kwant.wraparound.wraparound(correlated_folded_syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200, color='k')
fig, ax = band_plot(fig, ax, kwant.wraparound.wraparound(bare_folded_syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200, color='r')
