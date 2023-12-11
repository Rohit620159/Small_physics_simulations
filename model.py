#importing required librairies
import numpy as np
import tqdm.notebook import tddm
from matplotlib import pyplot as plt

#Defining class - Ising model & its properties
class 2dIsing:
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
        itself.magnetisation = itself.spins.sum()                             #initial energy and magnetisation
         
    def _energy(itself):                                       #total energy of system by iterating over all NEIGHBORS and adding J
        interaction_energy = -itself.J * (
            (itself.spins[:-1, :] * itself.spins[1:, :]).sum() +
            (itself.spins[:, :-1] * itself.spins[:, 1:]).sum() +
            (itself.spins[0, :] * itself.spins[-1, :]).sum() +
            (itself.spins[:, 0] * itself.spins[:, -1]).sum()
        )
        return interaction_energy
    def random_update(itself):
        i = np.random.randint(0, itself.Nx)
        j = np.random.randint(0, itself.Ny)
        deltaE = itself.delta_energy(i, j)      #change in energy due single flip
        if deltaE <= 0 or np.exp(-itself.beta * deltaE) > np.random.rand():        # criteria for accepting that change
            itself.spins[i, j] *= -1
            itself.energy += deltaE
            itself.magnetisation += 2 * itself.spins[i, j] 
#Parameters
N_trials = 5         #No of times to run the simulation for each temp 
N_equilibrate, N_compute = 100000, 100000    # steps to reach equilibrium then computing steps
Nx, Ny = 10, 10
N = Nx * Ny
J = 1
betas = np.geomspace(0.05, 2, 101)

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
axs = [ax for row in axs for ax in row]
for beta in tqdm(betas, leave=False):
    energies, magnetisations, heatcapacities, susceptibilities = [], [], [], []
    quantities = [energies, magnetisations, heatcapacities, susceptibilities]
    for trial in range(N_trials):
        system = 2dIsing(Nx, Ny, J, beta, 'hot')
        for i in range(N_equilibrate):
            system.random_update()
            
        energy = np.zeros([N_compute])
        magnetisation = np.zeros([N_compute])
        for i in range(N_compute):
            system.random_update()
            energy[i] = system.energy
            magnetisation[i] = system.magnetisation

        energies.append(np.mean(energy) / N)
        magnetisations.append(np.abs(np.mean(magnetisation)) / N)
        heatcapacities.append(beta ** 2 * (np.mean(energy ** 2) - np.mean(energy) ** 2) / N)
        susceptibilities.append(beta * (np.mean(magnetisation ** 2) - np.mean(magnetisation) ** 2) / N)

    for ax, quantity in zip(axs, quantities):
        ax.scatter([1/beta], [np.median(quantity)], color='k', s=5)



