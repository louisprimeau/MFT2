import copy
import kwant
import numpy as np

def auxilliary_system(syst, operator):
    auxiliary_syst = copy.deepcopy(syst)
    n_orbs = bare_syst[list(bare_syst.sites())[0]].shape[0]
    sites = list(syst.sites())
    for i, s_a in enumerate(sites):
        for j, s_b in enumerate(sites):
            if s_a == s_b:
                auxiliary_syst[s_a] = operator[i*n_orbs:(i+1)*n_orbs, j*n_orbs:(j+1)*n_orbs]
            else:
                auxiliary_syst[s_a, s_b] = operator[i*n_orbs:(i+1)*n_orbs, j*n_orbs:(j+1)*n_orbs]
    return auxiliary_syst

def reduce_orbitals(syst, operator, n_orbs):
    new_syst = kwant.Builder()
    sites = list(syst.sites())
    
    for i, s_a in enumerate(sites):
        new_syst[s_a] = operator[i*n_orbs:(i+1)*n_orbs, i*n_orbs:(i+1)*n_orbs]
    
    for i, s_a in enumerate(sites):
        for j, s_b in enumerate(sites):
            if s_a == s_b: continue
            new_syst[s_a, s_b] = operator[i*n_orbs:(i+1)*n_orbs, j*n_orbs:(j+1)*n_orbs]
    return new_syst

def subtract_systems(systA, systB):
    new_syst = copy.deepcopy(systA)
    sites = list(systA.sites())
    for i, s_a in enumerate(sites):
        for j, s_b in enumerate(sites):
            if s_a == s_b:
                new_syst[s_a] = systA[s_a] - systB[s_a]
            else:
                if (s_a, s_b) in systA.hoppings() and (s_a, s_b) in systB.hoppings():
                    new_syst[s_a, s_b] = systA[s_a, s_b] - systB[s_a, s_b]
                elif (s_a, s_b) in systA.hoppings() and (s_a, s_b) not in systB.hoppings():
                    new_syst[s_a, s_b] = systA[s_a, s_b]
                elif (s_a, s_b) not in systA.hoppings() and (s_a, s_b) in systB.hoppings():
                    new_syst[s_a, s_b] = -systB[s_a, s_b]
                else:
                    continue     
    return new_syst

def add_systems(systA, systB):
    new_syst = copy.deepcopy(systA)
    sites = list(systA.sites())
    for i, s_a in enumerate(sites):
        for j, s_b in enumerate(sites):
            if s_a == s_b:
                new_syst[s_a] = systA[s_a] - systB[s_a]
            else:

                if (s_a, s_b) in systA.hoppings() and (s_a, s_b) in systB.hoppings():
                    new_syst[s_a, s_b] = systA[s_a, s_b] + systB[s_a, s_b]
                elif (s_a, s_b) in systA.hoppings() and (s_a, s_b) not in systB.hoppings():
                    new_syst[s_a, s_b] = systA[s_a, s_b]
                elif (s_a, s_b) not in systA.hoppings() and (s_a, s_b) in systB.hoppings():
                    new_syst[s_a, s_b] = systB[s_a, s_b]
                else:
                    continue
    return new_syst
