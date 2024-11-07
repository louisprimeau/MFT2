import numpy as np
import os
import kwant

def make_system_from_vasp(file_path, periodic=True):

    with open(file_path) as f:
        lines = [line.rstrip() for line in file]
    # ignore line[0]
    n_orbitals = int("".join(lines[1].split()))
    n_hoppings = int("".join(lines[2].split()))
    # ignore line[3]
    # rest of lines are hoppings...
    syst = kwant.Builder()
    lat = kwant.lattice.square(a=1, norbs=n_orbitals)

    for line in lines[4:]:

        W, L, _, orb_from, orb_to, t_real, t_imag = line.split()
        W, L, orb_from, orb_to = int(W), int(L), int(orb_from), int(orb_to)
        t_real, t_imag = float(t_real), float(t_imag)
        
        syst[lat(W,L), lat(W, L)] = t_real + 1j * t_imag

    return syst




syst = make_system_from_vasp("wannier90_cdw_hr.dat")
