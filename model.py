#importing required librairies
import numpy as np
import tqdm.notebook import tddm
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

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
# Define dictionaries for colors and markers
markers = {
    "Energy": "o",
    "Magnetisation": "-",
    "Heat capacity": "^",
    "Susceptibility": "s",
}

colors = {
    "Energy": "blue",
    "Magnetisation": "red",
    "Heat capacity": "green",
    "Susceptibility": "purple",
}
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
    # Plot results with different colors and markers
    cmap = get_cmap("viridis", len(quantities))
    for ax, quantity, i in zip(axs, quantities, range(len(quantities))):
        marker = markers[quantity]
        color = colors[quantity]

        ax.scatter([1/beta], [np.median(quantity)], color=color, marker=marker, s=5, label=quantity, alpha=0.7)
        ax.plot([1/beta], [np.median(quantity)], color=color, marker=marker, linestyle="-", alpha=0.7)

# Configure plot
axs[0].set_xscale('log')
axs[0].set_ylabel('Energy')
axs[1].set_ylabel('Magnetisation')
axs[2].set_ylabel('Heat capacity')
axs[3].set_ylabel('Susceptibility')
for ax in axs:
    ax.set_xlabel(r'$k_B T$')
    ax.legend()

plt.tight_layout()
plt.show()
# Save the plot as a file (choose your preferred format, e.g., PDF, PNG)
plt.savefig("2D_ising_model_plot.png", bbox_inches="tight")


