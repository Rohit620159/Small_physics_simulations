#importing required librairies
import numpy as np
import tqdm.notebook import tddm
from matplotlib import pyplot as plt

#Defining class - Ising model & its properties
class Ising:
    def __init__(                        #initializes
        itself, Nx: int, Ny: int, J: float, beta: float,                #defines size N of lattice , interaction energy J , inverse temp Beta
        start_type=None
    ):
        itself.Nx, itself.Ny = Nx, Ny                                   #size of grid
        itself.J, itself.beta = J, beta

        if start_type is None:                                   #starting state alignes- all spin up or down
            start_type = 'aligned'

        if start_type == 'aligned':
            itself.spins = np.ones([Nx, Ny], dtype=int) * np.random.choice([-1, 1])       #grid of spins of spins mostly alined using np ones
        else:
            itself.spins = np.random.choice([-1, 1], size=[Nx, Ny])          #random grid of spins - not aligned
        itself.energy = itself._energy()
        itself.magnetisation = itself.spins.sum()
         
