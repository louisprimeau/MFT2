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

def make_system(correlation, periodic=True):
    syst = kwant.Builder()
    lat = kwant.lattice.square(a=1, norbs=2)
    #lat = kwant.lattice.triangular(a=1, norbs=2)

    for i in range(W):
        for j in range(L):
            C = correlation[i, j]
            syst[lat(i,j)] = U * np.array([[C[1,1],-C[1,0]],
                                           [-C[0,1], C[0,0]]])
            if j > 0:
                syst[lat(i,j), lat(i, j-1)] = -t * sigma_0 
            if i > 0:
                syst[lat(i,j), lat(i-1, j)] = -t * sigma_0
            #if i > 0 and j < L-1:
            #    syst[lat(i,j), lat(i-1, j+1)] = -t * sigma_0 # triangular lattice

    # make periodic (square lattice):
    if periodic:
        for i in range(W):
            syst[lat(i, L-1), lat(i, 0)] = -t * sigma_0
        for i in range(L):
            syst[lat(W-1, i), lat(0, i)] = -t * sigma_0
    return syst

def compute_correlation(eigs, v, mu=0):
    correlation = np.zeros((W, L, 2, 2), dtype='complex')
    mask = (eigs < mu)
    correlation[:, :, 0, 0] = np.sum(v[0::2, mask].conj() * v[0::2, mask], axis=1).reshape(W, L)
    correlation[:, :, 0, 1] = np.sum(v[0::2, mask].conj() * v[1::2, mask], axis=1).reshape(W, L)
    correlation[:, :, 1, 0] = np.sum(v[1::2, mask].conj() * v[0::2, mask], axis=1).reshape(W, L)
    correlation[:, :, 1, 1] = np.sum(v[1::2, mask].conj() * v[1::2, mask], axis=1).reshape(W, L)
    return correlation

# Model Parameters
W, L = 2, 2
filling, doping = 1/2, 0
U, t = 3.0, 1

# Initial Conditions
#correlation = np.tensordot(np.ones((W,L)), sigma_0, axes=0)
correlation = np.random.rand(*(W,L,2,2)) / 2 + 1j * np.random.rand(*(W,L,2,2)) / 2
syst = make_system(correlation).finalized()
H = syst.hamiltonian_submatrix(sparse=False)
for iteration in range(3500):
    new_H = make_system(correlation, periodic=True).finalized().hamiltonian_submatrix(sparse=False)
    H = 0.4 * H + 0.6 * new_H
    eigs, eigv = np.linalg.eigh(H)
    EF = np.sort(eigs)[int(eigs.shape[0]*filling) + doping]

    new_correlation = compute_correlation(eigs, eigv, mu=EF)
    print("iteration {}, max correlation: {:.8f}".format(iteration, np.max(np.abs(new_correlation - correlation))))
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

# Compute spin and regular densities
n_up = np.abs(correlation[:, :, 0, 0])
n_down = np.abs(correlation[:, :, 1, 1])
S_z = np.abs(1/2 * (correlation[:, :, 0, 0] - correlation[:, :, 1, 1]))
S_x = np.abs(1/2 * (correlation[:, :, 0, 1] + correlation[:, :, 1, 0]))
S_y = np.abs(1/2 * (correlation[:, :, 0, 1] - correlation[:, :, 1, 0]))

# In 2D:
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
aa = ax1.matshow(n_up, vmin=0, vmax=1); ax1.set_title('<n↑>')
ax2.matshow(n_down, vmin=0, vmax=1); ax2.set_title('<n↓>')
ax3.matshow(S_z, vmin=0, vmax=1); ax3.set_title('<Sz>')
ax4.matshow(S_x, vmin=0, vmax=1); ax4.set_title('<Sx>')
ax5.matshow(S_y, vmin=0, vmax=1); ax5.set_title('<Sy>')
fig.colorbar(aa, ax=ax5)

plt.show()
