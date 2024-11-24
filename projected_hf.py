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

    #assert np.all(np.diag(correlation).real > 0)
    new_syst = copy.deepcopy(syst)
    sites = list(syst.sites())
    
    As, ds = compute_adjacency(syst, n_orbs, hopping_range, periods=(lattice_vectors[0,0] * W_size, lattice_vectors[1,1] * H_size))
    A = coarsen(As[0], n_orbs)
    for i, j in zip(*np.nonzero(A)):
        assert i == j
        new_syst[sites[i]] += onsite_interactions_2orbs(correlation[i * n_orbs:(i+1)* n_orbs, j * n_orbs:(j+1)*n_orbs])
    
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

def onsite_interactions_4orbs(correlation):
    
    interaction = np.zeros_like(correlation)

    # Hartree
    interaction[0, 0] += U * correlation[2, 2]
    interaction[1, 1] += U * correlation[3, 3]
    interaction[2, 2] += U * correlation[0, 0]
    interaction[3, 3] += U * correlation[1, 1]

    interaction[0, 2] += -U * correlation[2, 0]
    interaction[1, 3] += -U * correlation[3, 1]
    interaction[2, 0] += -U * correlation[0, 2]
    interaction[3, 1] += -U * correlation[1, 3]

    # Fock
    interaction[0, 0] += V * correlation[1, 1] + V * correlation[3, 3]
    interaction[0, 1] += - V * correlation[1, 0]
    interaction[0, 3] += - V * correlation[3, 0]

    interaction[1, 1] += V * correlation[0, 0] + V * correlation[2, 2]
    interaction[1, 0] += - V * correlation[0, 1]
    interaction[1, 2] += - V * correlation[2, 1]
    
    interaction[2, 2] += V * correlation[1, 1] + V * correlation[3, 3]
    interaction[2, 1] += - V * correlation[1, 2]
    interaction[2, 3] += - V * correlation[3, 2]

    interaction[3, 3] += V * correlation[0, 0] + V * correlation[2, 2]
    interaction[3, 0] += - V * correlation[0, 3]
    interaction[3, 2] += - V * correlation[2, 3]

    return interaction

def onsite_interactions_2orbs(correlation):
    interaction = np.zeros_like(correlation)

    interaction[0, 0] += U * correlation[1, 1]
    interaction[1, 1] += U * correlation[0, 0]

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
    return np.sort(eigs)[1]

def cdw_density(syst, period):
    positions = np.array([site.pos for site in syst.sites()]).repeat(n_orbs, axis=0)
    filling = W * L * n_orbs // 2 + int(W * L / 15)
    oscillation = 1 + np.cos(2 * pi * positions[:, 0] / period)
    oscillation = oscillation * filling / np.sum(oscillation)
    return np.diag(oscillation)

def projectors(es, vs):
    EF = 0.0
    vs_below = vs[:, es < EF]
    vs_above = vs[:, es > EF]
    P_below = vs_below
    P_above = vs_above
    return P_below, P_above

def objective_function(correlation):
    correlation = correlation.reshape(W*L*2*n_orbs, W*L*2*n_orbs)
    syst = add_interaction(bare_reduced_syst, correlation, n_orbs, lattice_vectors, W, L, hopping_range=1)
    H = syst.finalized().hamiltonian_submatrix(sparse=False)
    eigs, eigv = np.linalg.eigh(H)
    EF = compute_EF(eigs)
    new_correlation = compute_correlation(eigs, eigv, mu=EF).reshape(-1)
    return new_correlation

# HARTREE FOCK LOOP
W, L = 18, 18
U = 2.0 * 2
V = 3.0
UL = 0.0

bare_syst, n_orbs, lattice_vectors, x_coordinates, lat = make_system_from_vasp("wannier90_hr_soc.dat", "POSCAR-unit.vasp", W, L, periodic=True,  translationally_invariant=False)

H0 = bare_syst.finalized().hamiltonian_submatrix(sparse=False)
es, vs = np.linalg.eigh(H0)
P_below, P_above = projectors(es, vs)

es_below = es[es < 0.0]
es_above = es[es > 0.0]
H_above = P_above @ np.diag(es_above) @ P_above.conj().T
#H_above = H0[::2, ::2]
bare_reduced_syst = reduce_orbitals(bare_syst, H_above, 2)
n_orbs = 2

H0_reduced = bare_reduced_syst.finalized().hamiltonian_submatrix(sparse=False)

#H0 = bare_reduced_syst.finalized().hamiltonian_submatrix(sparse=False)

#init_correlation = np.eye(n_orbs * 2 * W * L).astype(np.complex128) * (W * L * 2* n_orbs // 2) / (n_orbs * W * L * 2)
#correlation = quasinewton.quasi_newton(objective_function, init_correlation.reshape(-1)).reshape(W * L * 2 * n_orbs, W * L * 2 * n_orbs)

#correlated_syst = add_interaction(bare_reduced_syst, correlation, n_orbs, lattice_vectors, W, L)
#H = correlated_syst.finalized().hamiltonian_submatrix(sparse=False)

E0 = np.linalg.eigvalsh(H0)
E0_reduced = np.linalg.eigvalsh(H0_reduced)

fig, ax = plt.subplots(1)
ax.scatter(np.arange(E0.shape[0]), E0, label='E0', marker='x')
ax.scatter(np.arange(E0_reduced.shape[0]) + E0_reduced.shape[0], E0_reduced, label='E0_r', marker='o')
#ax.scatter(np.arange(E.shape[0]), E, label='E', marker='+')
#ax.scatter(np.arange(E0.shape[0]), E0, label='E0')
ax.legend()

#folded_syst_old, _, _, _, _ = make_system_from_vasp("wannier90_cdw_hr.dat", "POSCAR-15.vasp", 3, 3, periodic=False,  translationally_invariant=True)

#correlated_folded_syst = bandfold(correlated_syst, 1, 1, lattice_vectors, x_coordinates, lat)
bare_folded_syst = bandfold(bare_syst, 1, 1, lattice_vectors, x_coordinates, lat)
bare_reduced_folded_syst = bandfold(bare_reduced_syst, 1, 1, lattice_vectors, x_coordinates, lat)

fig, ax = plt.subplots(1)
fig, ax = band_plot(fig, ax, kwant.wraparound.wraparound(bare_folded_syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200, color='r')
fig, ax = band_plot(fig, ax, kwant.wraparound.wraparound(bare_reduced_folded_syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200, color='b')
#fig, ax = band_plot(fig, ax, kwant.wraparound.wraparound(correlated_folded_syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200, color='k')

