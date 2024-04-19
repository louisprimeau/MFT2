import matplotlib.pyplot as plt
import kwant
import scipy.sparse as sp
import numpy as np


sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def DoS(sample_points, E, n=1000):
    sigma = 5*np.mean(np.abs(E[1:] - E[:-1]))
    dos = np.zeros(n)
    for e in E: dos += gaussian(sample_points, e, sigma)  
    return dos / E.size

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def compute_filling(correlation):
    return np.sum(correlation[:, :, 0, 0] + correlation[:, :, 1, 1])

def vec_diag(arr):
    return np.expand_dims(arr, axis=-2) * np.eye(arr.shape[-1])

def assign_density(correlation, density):
    for i in range(W):
        for j in range(L):
            c = i * L + j
            correlation[i, j] = density[2*c:2*c+2, 2*c:2*c+2]
    return correlation

def coarsen(A, n):
    N, M = A.shape[0] // n, A.shape[1] // n
    return np.mean(A.reshape(N, n, M, n), axis=(1,3))

def num_sites(W, L, n_orbs):
    syst = kwant.Builder()
    lat = kwant.lattice.kagome(a=1, norbs=n_orbs)
    a, b, c = lat.sublattices
    for i in range(W):
        for j in range(L):
            syst[a(i, j)] = np.eye(n_orbs)
            syst[b(i, j)] = np.eye(n_orbs)
            syst[c(i, j)] = np.eye(n_orbs)
    return len(list(syst.sites()))

def make_system(correlation, n_orbs):
    syst = kwant.Builder()
    lat = kwant.lattice.kagome(a=1, norbs=n_orbs)
    a, b, c = lat.sublattices
    for i in range(W):
        for j in range(L):
            syst[a(i, j)] = 4 * np.eye(n_orbs) #np.zeros((n_orbs, n_orbs))
            syst[b(i, j)] = 4 * np.eye(n_orbs) #np.zeros((n_orbs, n_orbs))
            syst[c(i, j)] = 4 * np.eye(n_orbs) #np.zeros((n_orbs, n_orbs))

    sites = list(syst.sites())
    nn0, nn1, nn2, nn3 = compute_adjacency(syst, n_orbs, 3)

    A = coarsen(nn1, n_orbs)
    for i, j in zip(*np.nonzero(A)):
        syst[(sites[i], sites[j])] = -np.eye(n_orbs) + 0.1 * correlation[i*n_orbs:(i+1)*n_orbs, j*n_orbs:(j+1)*n_orbs]

    A = coarsen(nn2, n_orbs)
    for i, j in zip(*np.nonzero(A)):
        syst[(sites[i], sites[j])] = 0.05 * correlation[i*n_orbs:(i+1)*n_orbs, j*n_orbs:(j+1)*n_orbs]

    return syst

def compute_adjacency(syst, n_orbs, N):
    positions = np.array([site.pos for site in syst.sites()]).repeat(n_orbs, axis=0)
    x, y = positions[:, 0], positions[:, 1]
    dmatrix = np.sqrt(np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)
    distances = np.sort(np.unique(dmatrix.round(decimals=8)))
    As = []
    for i in range(0, N+1):
        As.append((np.abs(dmatrix - distances[i]) < 1e-6).astype(float))
    return As

def compute_correlation(W, L, n_orbs, eigs, v, mu=0):
    mask = (eigs < mu)
    correlation = v @ np.diag(mask) @ v.conj().T
    return correlation

# Model Parameters
W, L = 20,20
filling, doping = 1/2, 0
U, t = 3.0, 1
n_orbs = 2

n_sites = num_sites(W, L, n_orbs)

# Initial Conditions
#correlation = np.tensordot(np.ones((W,L)), sigma_0, axes=0)
correlation = np.random.rand(*(n_sites * n_orbs, n_sites * n_orbs)) + \
    1j * np.random.rand(*(n_sites * n_orbs, n_sites * n_orbs)) 
correlation = (correlation + correlation.T.conj()) / 4
syst = make_system(correlation, n_orbs)
kwant.plot(syst, unit=0.1, site_size=0.7, hop_lw=0.5)
plt.show()

syst = syst.finalized()
H = syst.hamiltonian_submatrix(sparse=False)

for iteration in range(99):

    # Make system with new correlations
    new_H = make_system(correlation, n_orbs).finalized().hamiltonian_submatrix(sparse=False)

    # Mix with old system for stability
    H = 0.2 * H + 0.8 * new_H

    # solve
    eigs, eigv = np.linalg.eigh(H)

    # Calculate the fermi energy
    EF = np.sort(eigs)[int(eigs.shape[0]*filling) + doping]

    # recompute the correlation
    new_correlation = compute_correlation(W, L, n_orbs, eigs, eigv, mu=EF)

    # check for convergence
    print("iteration {}, max correlation: {:.8f}".format(iteration, np.max(np.abs(new_correlation - correlation))))
    print("max change:", np.max(np.abs(new_correlation - correlation)))
    if iteration > 100: breakpoint()
    print("EF:", EF)
    if np.max(np.abs(new_correlation - correlation)) < 1e-7:
        print("terminated at iteration {}".format(iteration))
        break
    
    correlation = new_correlation 
    #total_energy = np.sum(np.sort(eigs)[:int(eigs.shape[0]*filling)])
    #print('total_energy: {}'.format(total_energy))

    
# Compute and plot DoS
energies = eigs
erange = np.linspace(np.min(energies), np.max(energies), 1000)
dos = DoS(erange, energies, n=erange.shape[0])
fig, ax = plt.subplots(1)
ax.plot(erange - EF, dos)
ax.hist(eigs - EF, bins=50, density=True)
#ax.axvline(0, color='k')

"""
# In 1D:
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(correlation[:, :, 0, 0].real.reshape(-1), label='<n↑>')
ax1.plot(correlation[:, :, 1, 1].real.reshape(-1), label='<n↓>')
ax1.legend()
ax2.plot(np.abs(1/2 * (correlation[:, :, 0, 0] - correlation[:, :, 1, 1]).reshape(-1)), label='|<Sz>|')
ax2.plot(np.abs(1/2 * (correlation[:, :, 0, 1] + correlation[:, :, 0, 1]).reshape(-1)), label='|<Sx>|')
ax2.plot(np.abs(1/2 * (correlation[:, :, 0, 1] - correlation[:, :, 0, 1]).reshape(-1)), label='|<Sy>|')
ax2.legend()
"""
