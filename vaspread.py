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


def get_x_coordinates(poscar_file):
    
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
