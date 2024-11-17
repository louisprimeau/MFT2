import numpy as np
from numpy import pi
import os
import kwant
import matplotlib.pyplot as plt
import quasinewton
from bandplot import *
from vaspread import *
import scipy
import itertools
import time
import copy

def make_system_from_vasp(hopping_file_path, coordinate_file_path, W_size, H_size, periodic=True, translationally_invariant=False):

    n_orbs, hoppings = hopping_dict(hopping_file_path)
    x_coordinates, lattice_vectors = get_x_coordinates(coordinate_file_path)
    max_hopping = np.min(hoppings[(1, 0)])

    lattice_vectors = lattice_vectors[:2, :2]
    n_sublattices = lattice_vectors.shape[1]
    n_orbs = n_orbs // n_sublattices

    lat = kwant.lattice.general(lattice_vectors, norbs=n_orbs, basis=x_coordinates[:2, :2])

    if translationally_invariant:
        sym = kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1)))
        syst = kwant.Builder(sym)
    else:
        syst = kwant.Builder()
    
    for W, L, sl1, sl2 in itertools.prod(range(W_size),
                                         range(H_size),
                                         range(n_sublattices),
                                         range(n_sublattices)):

        syst[lat.sublattices[sl1](W,L)] = hoppings[(0,0)][sl1::2, sl2::2]

    for W, L, H, sl1, sl2 in itertools.prod(range(W_size),
                                            range(H_size),
                                            hoppings.keys(),
                                            range(n_sublattices),
                                            range(n_sublattices)):
        if H == (0,0): continue
        if W_size > W + H[0] >= 0 and H_size > L + H[1] >= 0:
            syst[lat.sublattices[sl1](W, L), lat.sublattices[sl2](W + H[0], L + H[1])] = hoppings[H][sl1::2, sl2::2]

    # periodic parts
    if periodic:
        assert not translationally_invariant
        for i, sl1, sl2 in itertools.prod(range(W), range(n_sublattices), range(n_sublattices)):
            syst[lat.sublattices[sl1](i, L-1), lat.sublattices[sl2](i, 0)] = hoppings[(0, 1)][sl1::2, sl2::2]
            syst[lat.sublattices[sl1](i, L-1), lat.sublattices[sl2](i, 0)] = hoppings[(0, -1)][sl1::2, sl2::2]

        for i, sl1, sl2 in itertools.prod(range(L), range(n_sublattices), range(n_sublattices)):
            syst[lat.sublattices[sl1](W-1, i), lat.sublattices[sl2](0, i)] = hoppings[(1, 0)][sl1::2, sl2::2]
            syst[lat.sublattices[sl1](W-1, i), lat.sublattices[sl2](0, i)] = hoppings[(-1, 0)][sl1::2, sl2::2]

        # PUT IN: diagonal hoppings ??????

    return syst, n_orbs, lattice_vectors, max_hopping

def add_interaction(syst, correlation, n_orbs, lattice_vectors, W_size, H_size, hopping_range=0):

    #assert np.all(np.diag(correlation).real > 0)
    syst = copy.deepcopy(syst)
    sites = list(syst.sites())
    
    As, ds = compute_adjacency(syst, n_orbs, hopping_range, periods=(lattice_vectors[0,0] * W_size, lattice_vectors[1,1] * H_size)) 
    A = coarsen(As[0], n_orbs)
    for i, j in zip(*np.nonzero(A)):
        assert i == j
        syst[sites[i]] += onsite_interactions(correlation[i * n_orbs:(i+1)* n_orbs, j * n_orbs:(j+1)*n_orbs])

    for nn in range(1, hopping_range+1):
        A = coarsen(As[nn], n_orbs)
        for i, j in zip(*np.nonzero(A)):
            try:
                syst[(sites[i], sites[j])] += UL * -correlation[j * n_orbs:(j+1)* n_orbs, i * n_orbs:(i+1)*n_orbs].T / ds[nn]
            except KeyError:
                syst[(sites[i], sites[j])] = UL * -correlation[i * n_orbs:(i+1)* n_orbs, j * n_orbs:(j+1)*n_orbs].T / ds[nn]
        syst[sites[i]] += UL * np.diag(np.diag(correlation)[j*n_orbs:(j+1)*n_orbs]) / ds[nn]

    return syst

def onsite_interactions(correlation):
    
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
    return np.sort(eigs)[W * L * 2 * n_orbs // 2 + 1]

def objective_function(correlation):
    #if not np.all(np.diag(correlation.reshape(W*L*n_orbs, W*L*n_orbs)) > 0):
    syst = add_interaction(bare_syst, correlation.reshape(W*L*2*n_orbs, W*L*2*n_orbs), n_orbs, lattice_vectors, W, L, hopping_range=1)

    syst = syst.finalized()
    H = syst.hamiltonian_submatrix(sparse=False)
    H0 = bare_syst.finalized().hamiltonian_submatrix(sparse=False)
    eigs, eigv = np.linalg.eigh(H)
    EF = compute_EF(eigs)
    new_correlation = compute_correlation(eigs, eigv, mu=EF).reshape(-1)
    return new_correlation

def cdw_density(syst, period):
    positions = np.array([site.pos for site in syst.sites()]).repeat(n_orbs, axis=0)
    filling = W * L * n_orbs // 2 + int(W * L / 15)
    oscillation = 1 + np.cos(2 * pi * positions[:, 0] / period)
    oscillation = oscillation * filling / np.sum(oscillation)
    return np.diag(oscillation)

# HARTREE FOCK LOOP
n_orbs = 120
W, L = 3, 3
U = 0.4 * 2
V = 0.0
UL = 0.0

correlation = np.zeros((n_orbs * W * L, n_orbs * W * L)).astype(np.complex128)
bare_syst, n_orbs, lattice_vectors, max_hopping = make_system_from_vasp("wannier90_cdw_hr.dat", "POSCAR-15.vasp", W, L, periodic=False,  translationally_invariant=True)

"""
breakpoint()

print("t1:", max_hopping)

positions = np.array([site.pos for site in bare_syst.sites()]).repeat(n_orbs, axis=0)
#init_correlation = cdw_density(bare_syst, lattice_vectors[0,0] * 15).astype(np.complex128)
init_correlation = np.eye(n_orbs * 2 * W * L).astype(np.complex128) * (W * L * 2* n_orbs // 2) / (n_orbs * W * L * 2)
correlation = quasinewton.quasi_newton(objective_function, init_correlation.reshape(-1)).reshape(W * L * 2 * n_orbs, W * L * 2 * n_orbs)

syst, _, _, _ = make_system_from_vasp("wannier90_hr_soc.dat", "POSCAR-unit.vasp", W, L, correlation, periodic=True,  translationally_invariant=False)
correlated_syst = add_interaction(syst, correlation, n_orbs, lattice_vectors, W, L)
correlated_eigs = np.linalg.eigvalsh(correlated_syst.finalized().hamiltonian_submatrix(sparse=False))
EF = compute_EF(correlated_eigs)

syst, _, _, _ = make_system_from_vasp("wannier90_hr_soc.dat", "POSCAR-unit.vasp", W, L, np.zeros([n_orbs*W*L, n_orbs*W*L]), periodic=True,  translationally_invariant=False)
eigs = np.linalg.eigvalsh(syst.finalized().hamiltonian_submatrix(sparse=False))

fig, ax = plt.subplots(1)

ax.scatter(list(range(len(correlated_eigs))), correlated_eigs)
ax.axhline(EF)
ax.scatter(list(range(len(eigs))), eigs)

fig, ax = plt.subplots(1)

#ax.plot(positions[:, 0], np.diag(correlation).real)
#ax.plot(positions[:, 0], np.diag(init_correlation).real)
for i in range(0, 8):
    ax.plot(np.diag(correlation)[i::8].real)
print([np.mean(np.diag(correlation)[i::8].real) for i in range(8)])
#ax.plot(np.diag(init_correlation).real)
"""

fig, ax = plt.subplots(1)
fig, ax = bandplot(fig, ax, kwant.wraparound.wraparound(syst, keep=None).finalized(), ['Y', 'G', 'X', 'R', 'G'], 200)
