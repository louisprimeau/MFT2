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


def get_coordinates(poscar_file):
    
    with open(poscar_file) as f:
        lines = [line.rstrip() for line in f]
    vecs = []
    for line in lines[2:5]:
        vecs.append(np.array([float(a) for a in line.split()]))
    vecs = np.array(vecs)

    rs = []
    for line in lines[8:]:
        rs.append(np.array([float(a) for a in line.split()]))
    rs = np.array(rs)

    return rs, vecs

def hopping_dict(hopping_file_path):
    
    with open(hopping_file_path) as f:
        lines = [line.rstrip() for line in f]
        
    # ignore line[0]
    n_orbs = int("".join(lines[1].split()))#; assert n_orbitals == n_orbitals_site_file
    n_hoppings = int("".join(lines[2].split()))
    # ignore line[3]
    # rest of lines are hoppings...

    all_Ws, all_Ls = [], []
    for line in lines[4:]:
        W, L, _, _, _, _, _ = line.split()
        all_Ws.append(int(W))
        all_Ls.append(int(L))
    hoppings = {a:np.zeros([n_orbs, n_orbs], dtype=np.complex128) for a in set(zip(all_Ls, all_Ws))}

    for line in lines[4:]:
        W, L, _, orb_from, orb_to, t_real, t_imag = line.split()
        W, L, orb_from, orb_to = int(W), int(L), int(orb_from) - 1, int(orb_to) - 1
        t_real, t_imag = float(t_real), float(t_imag)
        hoppings[(W,L)][orb_from, orb_to] = t_real + 1j * t_imag

    return n_orbs, hoppings

def make_system_from_vasp(hopping_file_path, coordinate_file_path, W_size, H_size, periodic=True, translationally_invariant=False):

    n_orbs, hoppings = hopping_dict(hopping_file_path)
    x_coordinates, lattice_vectors = get_coordinates(coordinate_file_path)
    x_coordinates = x_coordinates[:2, :2]
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

    # in unit cell
    for W, L in itertools.product(range(W_size),
                                  range(H_size)):
        syst[lat.sublattices[0](W,L)] = hoppings[(0,0)][0::2, 0::2]
        syst[lat.sublattices[1](W,L)] = hoppings[(0,0)][1::2, 1::2]
        syst[lat.sublattices[0](W,L), lat.sublattices[1](W,L)] = hoppings[(0,0)][0::2, 1::2]
        syst[lat.sublattices[1](W,L), lat.sublattices[0](W,L)] = hoppings[(0,0)][1::2, 0::2]

    # out of unit cell
    for W, L, H, sl1, sl2 in itertools.product(range(W_size),
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
        for i, sl1, sl2 in itertools.product(range(W), range(n_sublattices), range(n_sublattices)):
            syst[lat.sublattices[sl1](i, L-1), lat.sublattices[sl2](i, 0)] = hoppings[(0, 1)][sl1::2, sl2::2]
            syst[lat.sublattices[sl1](i, L-1), lat.sublattices[sl2](i, 0)] = hoppings[(0, -1)][sl1::2, sl2::2]

        for i, sl1, sl2 in itertools.product(range(L), range(n_sublattices), range(n_sublattices)):
            syst[lat.sublattices[sl1](W-1, i), lat.sublattices[sl2](0, i)] = hoppings[(1, 0)][sl1::2, sl2::2]
            syst[lat.sublattices[sl1](W-1, i), lat.sublattices[sl2](0, i)] = hoppings[(-1, 0)][sl1::2, sl2::2]
        # PUT IN: diagonal hoppings ??????
        
    return syst, n_orbs, lattice_vectors, x_coordinates, lat


class TBEngine():

    def __init__(self, ham_file, coordinate_file, other_engine=None):
        
        all_ts, all_Rs, kscale, nwan = self.initialize(ham_file, coordinate_file)

        self.kscale = kscale
        self.nwan = nwan
        self.all_ts = all_ts
        self.all_Rs = all_Rs
        
    def initialize(self, ham_file, coordinate_file):
        # TODO: generalize to non square lattice
        with open(ham_file) as f:
            lines = [line.rstrip() for line in f]
        
        # ignore lines[0]
        n_orbs = int("".join(lines[1].split()))
        n_hoppings = int("".join(lines[2].split()))
        
        # ignore lines[3]
        # rest of lines are hoppings...
        unit_cell_pos, lat_vecs = get_coordinates(coordinate_file)

        def make_map(source_Rs, all_Rs):
            all_idx = []
            for i, row_source in enumerate(source_Rs):
                this_row_j = np.nonzero((row_source[0] == all_Rs[:, 0]) & (row_source[1] == all_Rs[:, 1]) & (row_source[2] == all_Rs[:, 2]))[0][0]
                all_idx.append(this_row_j)
            return all_idx

        Rs = []; ts = []; orbs_from = []; orbs_to = []
        for line in lines[4:]:
            X, Y, Z, orb_from, orb_to, t_real, t_imag = line.split()
            X, Y, Z, orb_from, orb_to = int(X), int(Y), int(Z), int(orb_from) - 1, int(orb_to) - 1
            Rs.append((X * lat_vecs[:, 0] + Y * lat_vecs[:, 1] + Z * lat_vecs[:, 2]) - unit_cell_pos[orb_from, :] + unit_cell_pos[orb_to, :])
            ts.append(float(t_real) + 1j * float(t_imag))
            orbs_from.append(int(orb_from))
            orbs_to.append(int(orb_to))

        Rs = np.array(Rs)
        ts = np.array(ts)
        orbs_from = np.array(orbs_from)
        orbs_to = np.array(orbs_to)
        all_Rs = np.unique(Rs, axis=0)
        allts = np.zeros((n_orbs, n_orbs, all_Rs.shape[0]), dtype=np.complex128)

        map_all = np.array(make_map(Rs, all_Rs))
        allts[orbs_from, orbs_to, map_all] = ts

        return allts, all_Rs, np.diag(lat_vecs), n_orbs

    def spool_hamiltonian(self, k_points, expR=None):
        if expR is None:
            ks_c = k_points
            expR = np.exp(1j * np.einsum('lj,ij->li', ks_c * self.kscale, self.all_Rs, optimize='greedy'))
        return np.einsum('lk,ijk->lij', expR, self.all_ts, optimize='greedy')
