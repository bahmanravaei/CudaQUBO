# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:05:47 2023

@author: bahman
"""

import numpy as np
import matplotlib.pyplot as plt 

energyFilename = 'Energy.csv'
magnetFilename = 'Magnet.csv'


Energy = np.genfromtxt(energyFilename, delimiter=",")
Magnet = np.genfromtxt(magnetFilename, delimiter=",")


#Plot Result (Energy and Magnet) for when num_replicas == 1
#plt.plot(Energy)

#plt.plot(Magnet)

# Plot results (Energies)
num_replicas = Energy.shape[1]
temperatures = np.linspace(1, 4, num_replicas)
#num_replicas = 1
Energy = Energy.T
plt.figure(figsize=(12, 8))
for replica in range(num_replicas):
    plt.plot(Energy[replica, :], label=f'Temperature: {temperatures[replica]:.2f}')
    #plt.plot(Energy[replica, :])

plt.title('Replica Exchange Monte Carlo for the Ising Model')
plt.xlabel('MC Steps')
plt.ylabel('Energy')
plt.legend()
plt.show()
                   

# Plot results (Magnet)
Magnet = Magnet.T
plt.figure(figsize=(12, 8))
for replica in range(num_replicas):
    plt.plot(Magnet[replica, :], label=f'Temperature: {temperatures[replica]:.2f}')
    #plt.plot(Magnet[replica, :])

plt.title('Replica Exchange Monte Carlo for the Ising Model')
plt.xlabel('MC Steps')
plt.ylabel('Magnet')
plt.legend()
plt.show()


# Plot the intial lattice
InitlatticeFile = "Initlattice.csv"
Initlattice = np.genfromtxt(InitlatticeFile, delimiter=",")
plt.matshow(Initlattice)

# Plot the best lattice
latticeBestFile = "latticeBest.csv"
latticeBest = np.genfromtxt(latticeBestFile, delimiter=",")
plt.matshow(latticeBest)


# Plot the best lattice
latticeFinalFile = "latticeFinal.csv"
latticeFinal = np.genfromtxt(latticeFinalFile, delimiter=",")
plt.matshow(latticeFinal)



