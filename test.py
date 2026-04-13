import Support as sp
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import InverseDesign as invdes
import matplotlib.pyplot as plt
import sys

import pandas as pd
from scipy.io import savemat

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

global outputDirectory
outputDirectory = "data"
try:
    os.mkdir(f"{outputDirectory}")
except FileExistsError:
    print(f"Directory: {outputDirectory} Exists")

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
        "Chromosome": pop.chromosomes,
        "Scattered Power": pop.scatteredPower,
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
    plt.figure(figsize=(2*nroot,2*nroot))
    for i in range(n):
        plt.subplot(nroot, nroot,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pop.images[i, :, :], cmap=plt.cm.binary)
        plt.xlabel(f"fit: {np.round(pop.fitness[i], 2)} - rmse: {np.round(pop.rootMeanSquaredError[i], 2)}")
    plt.show()
    plt.savefig(f"{fname}Profiles.png")
    plt.close()


def plotPopulationScattering(pop, fname):
    x = np.linspace(0, 1, 100)
    n = pop.nProfiles
    nroot = int(np.round(np.sqrt(pop.nProfiles)))
    plt.figure(figsize=(2*nroot,2*nroot))
    for i in range(n):
        plt.subplot(nroot, nroot,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.plot(pop.wavelengths, pop.objective, zorder = -1, color='black')
        plt.plot(pop.wavelengths, pop.scatteredPower[i, :], zorder = -1)
        plt.fill_between(pop.wavelengths, pop.objective, pop.scatteredPower[i, :], color='red', alpha = 0.2)
        plt.xlabel(f"fit: {np.round(pop.fitness[i], 2)} - rmse: {np.round(pop.rootMeanSquaredError[i], 2)}")
    plt.xlim(300, 800)
    #plt.xlabel("Wavelength (nm)")
    #plt.ylabel("Scattered Power")
    plt.savefig(f"{fname}ScatteredPower.png")
    plt.close()




nGenerations = 150

generationalFitness = []
generationalIntegral = []
ft = 'rmse'
pop = invdes.Population(
    nVertices=24, fitnessType=ft,
    modelDirectory="/media/work/evan/deep_learning_data/trained_models",
    rMin=10, rMax=75, d=(180,180), s=12, p=12,
    mR = 0.20, cP = 1,
    l=[1,2,2], m=[1,1,2], f=['E', 'H'])

#Generate a new shape as an objective
#pop.defineObjective(profileType = 'pol', termsF=5)
#np.savetxt("objScatteredPower.txt",pop.objScatteredPower)

#set a scattered power spectrum as an objective
pop.defineObjective(spectrum=np.loadtxt("objScatteredPower.txt"))



pop.initialize(nT=20, nR=20, nC=20, nF=20, nP=20)
pop.displayParameters(outDir="data")
pop.writeLoss(iGen=0, outDir="data") 
pop.plotSelectPerformers(outputDirectory, "0")

"""

plt.imshow(pop.objImage, cmap=plt.cm.binary)
plt.savefig(f"objProfile.png")
plt.close()
plt.plot(pop.wavelengths, pop.objScatteredPower / np.max(pop.objScatteredPower), c='black')
plt.xlim(300, 800)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Scattered Power")
plt.title("Objective Scattered Power Spectrum")
plt.savefig(f"objScatteredPowerSpectrum.png")
plt.close()
"""





generationalFitness.append(pop.fitness)
if ft == 'mse':
    generationalIntegral.append(pop.meanSquaredError)
elif ft  == 'rmse':
    generationalIntegral.append(pop.rootMeanSquaredError)
elif ft == 'mre':
    generationalIntegral.append(pop.meanRelativeError)
elif ft == 'mae':
    generationalIntegral.append(pop.meanAbsoluteError)
elif ft == 'gap':
    generationalIntegral.append(pop.integral)
else:
    generationalIntegral.append(pop.integral)


print(pop.fitnessType)
for n in range(nGenerations):
    print(f"Generation {n + 1}")
    x = (0.5-0.25)*np.random.rand() + 0.25
    pop.newGeneration(x)
    pop.writeLoss(iGen=n+1, growthRate=x, outDir="data")
    pop.plotSelectPerformers(outputDirectory, f"{n+1}")
    generationalFitness.append(pop.fitness)

    #for i in range(pop.nProfiles):
    #    ProfilesList[k, :, :] = pop.images[i, : , :]
    #    k += 1


    if ft == 'mse':
        generationalIntegral.append(pop.meanSquaredError)
    elif ft == 'rmse':
        generationalIntegral.append(pop.rootMeanSquaredError)
    elif ft == 'mre':
        generationalIntegral.append(pop.meanRelativeError)
    elif ft == 'mae':
        generationalIntegral.append(pop.meanAbsoluteError)
    elif ft == 'gap':
        generationalIntegral.append(pop.integral)
    else:
        generationalIntegral.append(pop.integral)



#np.save(f"{outputDirectory}/allProfiles.npy", ProfilesList)

xmax = np.max(generationalIntegral)
xmin = np.min(generationalIntegral)



ymax = np.max(generationalFitness)
ymin = np.min(generationalFitness)

if ft == 'gap':
    xmax = min(xmax, 100)
else:
    xmax = min(xmax, 3)

ymax = min(ymax, 1)

for i in range(len(generationalFitness)):
    #plt.plot(np.linspace(xmin, xmax, 10), invdes.scale(np.linspace(xmin, xmax, 10)), zorder = -1, lw=1, c='black')
    plt.scatter(generationalIntegral[i], generationalFitness[i])
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.title(f"Generation {i} Fitness")
    if ft == 'mse':
        plt.xlabel("Mean Squared Error")
    elif ft  == 'rmse':
        plt.xlabel("Root Mean Squared Error")
    elif ft  == 'mre':
        plt.xlabel("Mean Relative Error")
    elif ft  == 'mae':
        plt.xlabel("Mean Absolute Error")
    elif ft == 'gap':
        plt.xlabel("Gap")
    else:
        plt.xlabel("")
    
    plt.ylabel("Fitness")
    plt.savefig(f"{outputDirectory}/fitnessGen{i}.png")
    plt.close()
