import Support as sp
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import InverseDesign as invdes
import matplotlib.pyplot as plt

import pandas as pd
from scipy.io import savemat

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

global outputDirectory
outputDirectory = "data"
try:
    os.mkdir(f"{outputDirectory}")
except FileExistsError:
    print(f"{outputDirectory} Exists")

def writePopulationData(pop, fname):
    df = pd.DataFrame({
        "Fitness": pop.fitness,
        "Integral": pop.integral,
        "Mean Absolute Error": pop.meanAbsoluteError,
        "Mean Relative Error": pop.meanRelativeError,
        "Mean Squared Error": pop.meanSquaredError,
        "Root Mean Squared Error": pop.rootMeanSquaredError})
    df.to_csv(f"{fname}.csv")
    d = {
        "Fitness": pop.fitness,
        "Profile": pop.images,
        "Chromosome": pop.chromosomes,
        "Scattered Power": pop.scatteredPower,
        "Multipole Coefficients": pop.multipoles,
        "Integral": pop.integral,
        "Mean Absolute Error": pop.meanAbsoluteError,
        "Mean Relative Error": pop.meanRelativeError,
        "Mean Squared Error": pop.meanSquaredError,
        "Root Mean Squared Error": pop.rootMeanSquaredError}
    savemat(f"{fname}.mat", d)
    

def plotPopulationFitness(pop, fname):
    x = np.linspace(0, 100, 100)
    #plt.plot(x, invdes.scale(x), zorder = -1, c='black')
    plt.scatter(pop.integral, pop.fitness, zorder = 1)
    plt.xlabel("Integral")
    plt.ylabel("Fitness")
    plt.savefig(f"{fname}Fitness.png")
    plt.close()

def plotPopulationProfiles(pop, fname):
    n = pop.nProfiles
    nroot = int(np.round(np.sqrt(pop.nProfiles)))
    plt.figure(figsize=(15,15))
    for i in range(n):
        plt.subplot(nroot, nroot,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pop.images[i, :, :], cmap=plt.cm.binary)
        plt.xlabel(f"int: {int(np.round(pop.integral[i]))} - rmse: {np.round(pop.rootMeanSquaredError[i], 2)}")
    plt.show()
    plt.savefig(f"{fname}Profiles.png")
    plt.close()


def plotPopulationScattering(pop, fname):
    x = np.linspace(0, 1, 100)
    n = pop.nProfiles
    nroot = int(np.round(np.sqrt(pop.nProfiles)))
    plt.figure(figsize=(15,15))
    for i in range(n):
        plt.subplot(nroot, nroot,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.plot(pop.wavelengths, pop.objective, zorder = -1, color='black')
        plt.plot(pop.wavelengths, pop.scatteredPower[i, :], zorder = -1)
        plt.fill_between(pop.wavelengths, pop.objective, pop.scatteredPower[i, :], color='red', alpha = 0.2)
        plt.xlabel(f"int: {int(np.round(pop.integral[i]))} - rmse: {np.round(pop.rootMeanSquaredError[i], 2)}")
    plt.xlim(300, 800)
    #plt.xlabel("Wavelength (nm)")
    #plt.ylabel("Scattered Power")
    plt.savefig(f"{fname}ScatteredPower.png")
    plt.close()



nGenerations = 20
nProfiles = 49

mseEvolution = []
rmseEvolution = []
fitnessEvolution = []
intEvolution = []
N = [0]
i = 0
pop = invdes.Population(nProfiles, 30,"/media/work/evan/deep_learning_data/trained_models", mR = 0.1, cP = 1)

plt.imshow(pop.objectivePoly)
plt.savefig(f"{outputDirectory}/obj.png")
plt.close()


plotPopulationFitness(pop, f"{outputDirectory}/gen0")
plotPopulationProfiles(pop, f"{outputDirectory}/gen0")
plotPopulationScattering(pop, f"{outputDirectory}/gen0")
writePopulationData(pop, f"{outputDirectory}/gen0")


mseEvolution.append(np.mean(pop.meanSquaredError))
rmseEvolution.append(np.mean(pop.rootMeanSquaredError))
fitnessEvolution.append(np.mean(pop.fitness))
intEvolution.append(np.mean(pop.integral))

for n in range(nGenerations):
    x = (0.5-0.25)*np.random.rand() + 0.25
    print(f"Generation: {n+1}")
    pop.newGeneration(x)
    print(f"Maximum Fitness: {np.max(pop.fitness)}")
    print(f"Minium Fitness: {np.min(pop.fitness)}")
    print(f"Mean Fitness: {np.mean(pop.fitness)}")
    print(f"Median Fitness: {np.median(pop.fitness)}")
    plotPopulationFitness(pop, f"{outputDirectory}/gen{n+1}")
    plotPopulationProfiles(pop, f"{outputDirectory}/gen{n+1}")
    plotPopulationScattering(pop, f"{outputDirectory}/gen{n+1}")
    writePopulationData(pop, f"{outputDirectory}/gen{n+1}")
    N.append(n+1)
    mseEvolution.append(np.mean(pop.meanSquaredError))
    rmseEvolution.append(np.mean(pop.rootMeanSquaredError))
    fitnessEvolution.append(np.mean(pop.fitness))
    intEvolution.append(np.mean(pop.integral))



plt.scatter(N,fitnessEvolution)
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.savefig(f"{outputDirectory}/FitnessEvolution.png")
plt.close()

plt.scatter(N,intEvolution)
plt.xlabel("Generations")
plt.ylabel("Average Integral")
plt.savefig(f"{outputDirectory}/IntegralEvolution.png")
plt.close()

plt.scatter(N,mseEvolution)
plt.xlabel("Generations")
plt.ylabel("Average Root Mean Squared Error")
plt.savefig(f"{outputDirectory}/RmseEvolution.png")
plt.close()

plt.scatter(N,rmseEvolution)
plt.xlabel("Generations")
plt.ylabel("Average Mean Squared Error")
plt.savefig(f"{outputDirectory}/MseEvolution.png")
plt.close()

