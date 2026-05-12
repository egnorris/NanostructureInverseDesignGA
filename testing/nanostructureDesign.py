import sys
import argparse
import os
import shutil
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

sys.path.append("../modules")

import Support
import runInverseDesign as run
import DataAnalysis

global entropyStudy
global fitnessStudy

global radialData
global fitnessData
global scatteringPowerData

def evolutionRun(s, br, parsedArgs):
    checkpoint = parsedArgs['seedCheckpoint']
    #reset random number generator
    np.random.seed(s)
    outDir =  parsedArgs['outDir']
    nGenerations = parsedArgs['numGenerations']
    birthRate= (br + 1)/parsedArgs['numBirthRate']
    birthRate = np.round(birthRate, 2)
    print(f"    Birth Rate: {birthRate}")
    genSave = parsedArgs['genSave']
    pop = run.setupPopulation(**parsedArgs)
    H, normH = DataAnalysis.getShannonEntropy(pop.chromosomes)
    entropyStudy[br, s, 0] = normH
    fitnessStudy[br, s, 0, 0] = pop.fitness[0]
    fitnessStudy[br, s, 0, 1] = pop.fitness[-1]
    fitnessStudy[br, s, 0, 2] = np.std(pop.fitness)
    fitnessStudy[br, s, 0, 3] = np.mean(pop.fitness)
    
    radialData[br, s, 0, :] = pop.polar
    fitnessData[br, s, 0, :] = pop.fitness
    scatteringPowerData[br, s, 0, :] = pop.scatteredPower

    if (s % checkpoint) == 0:
        DataAnalysis.plot6(f"Seed {s} - Birth Rate {birthRate} - Generation {0} Top Performers", outDir,pop)    
    for n in range(nGenerations):
        pop = run.updatePopulation(pop, birthRate=birthRate)
        H, normH = DataAnalysis.getShannonEntropy(pop.chromosomes)
        if ((n+1) % genSave) == 0:
            if (s % checkpoint) == 0:
                DataAnalysis.plot6(f"Seed {s} - Birth Rate {birthRate} - Generation {n+1} Top Performers",outDir,pop)
        entropyStudy[br, s, n+1] = normH
        fitnessStudy[br, s, n+1, 0] = pop.fitness[0]
        fitnessStudy[br, s, n+1, 1] = pop.fitness[-1]
        fitnessStudy[br, s, n+1, 2] = np.std(pop.fitness)
        fitnessStudy[br, s, n+1, 3] = np.mean(pop.fitness)
        radialData[br, s, n+1, :] = pop.polar
        fitnessData[br, s, n+1, :] = pop.fitness
        scatteringPowerData[br, s, n+1, :] = pop.scatteredPower


parser = argparse.ArgumentParser(description="Run Genetic Algorithm Inverse Design")
parser.add_argument('-outDir', type=str, required=False,
    default='output')
parser.add_argument('-numGenerations', type=int, required=False,
    default=30)
parser.add_argument('-numSeed', type=int, required=False,
    default=50)
parser.add_argument('-numBirthRate', type=int, required=False,
    default=5)
parser.add_argument('-genSave', type=int, required=False,
    default=5)
parser.add_argument('-seedCheckpoint', type=int, required=False,
    default=10)

parser.add_argument('-nV', type=int, required=False,
    default = Support.defaultKwargs["nV"])
parser.add_argument('-modelDir', type=str, required=False,
    default = Support.defaultKwargs["modelDir"])
parser.add_argument('-targetFile', type=str, required=False,
    default = Support.defaultKwargs["targetFile"])
parser.add_argument('-rMin', type=int, required=False,
    default = Support.defaultKwargs["rMin"])
parser.add_argument('-rMax', type=int, required=False,
    default = Support.defaultKwargs["rMax"])
parser.add_argument('-d', type=type(()), required=False,
    default = Support.defaultKwargs["d"])
parser.add_argument('-s', type=int, required=False,
    default = Support.defaultKwargs["s"])
parser.add_argument('-p', type=int, required=False,
    default = Support.defaultKwargs["p"])
parser.add_argument('-mR', type=float, required=False,
    default = Support.defaultKwargs["mR"])
parser.add_argument('-cP', type=int, required=False,
    default = Support.defaultKwargs["cP"])
parser.add_argument('-l', type=type([]), required=False,
    default = Support.defaultKwargs["l"])
parser.add_argument('-m', type=type([]), required=False,
    default = Support.defaultKwargs["m"])
parser.add_argument('-f', type=type([]), required=False,
    default = Support.defaultKwargs["f"])
parser.add_argument('-w0', type=int, required=False,
    default = Support.defaultKwargs["minWavelength"])
parser.add_argument('-w1', type=int, required=False,
    default = Support.defaultKwargs["maxWavelength"])
parser.add_argument('-saeF', type=int, required=False,
    default = Support.defaultKwargs["saeWeight"])
parser.add_argument('-sseF', type=int, required=False,
    default = Support.defaultKwargs["sseWeight"])
parser.add_argument('-nT', type=int, required=False,
    default = Support.defaultKwargs["nT"])
parser.add_argument('-nC', type=int, required=False,
    default = Support.defaultKwargs["nC"])
parser.add_argument('-nR', type=int, required=False,
    default = Support.defaultKwargs["nR"])
parser.add_argument('-nP', type=int, required=False,
    default = Support.defaultKwargs["nP"])
parser.add_argument('-nN', type=int, required=False,
    default = Support.defaultKwargs["nN"])
parsedArgs = parser.parse_args().__dict__



outDir = parsedArgs['outDir']

try:
    os.mkdir(f"{outDir}")
except FileExistsError:
    print(f"Directory: {outDir} Exists")

savemat(f"{outDir}/inputArgs.mat", parsedArgs)

nBirthRates = parsedArgs['numBirthRate']
nSeeds = parsedArgs['numSeed']
nGenerations = parsedArgs['numGenerations']
checkpoint = parsedArgs['seedCheckpoint']





nProfiles = parsedArgs['nT'] + parsedArgs['nC'] + parsedArgs['nR'] + parsedArgs['nP'] + parsedArgs['nN']

#reduced populaiton information for study plots at the end
entropyStudy = np.zeros((nBirthRates, nSeeds, nGenerations+1))
fitnessStudy = np.zeros((nBirthRates, nSeeds, nGenerations+1, 4))
#full population evolution data
radialData = np.zeros((nBirthRates, nSeeds, nGenerations+1, nProfiles, parsedArgs['nV'], 2))
fitnessData = np.zeros((nBirthRates, nSeeds, nGenerations+1, nProfiles))
scatteringPowerData = np.zeros((nBirthRates, nSeeds, nGenerations+1, nProfiles, 101))

finalFlag = False

try:
    a = loadmat(f"{outDir}/checkpoint/entropyStudy.checkpoint.mat")
    resume=a["checkpoint"][0][0]
    print("Checkpoint Data Detected!")
except FileNotFoundError:
    resume = -1

try:
    a = loadmat(f"{outDir}/entropyStudy.final.mat")
    print("Final Data detected!")
    finalFlag = True
except FileNotFoundError:
    try:
        os.mkdir(f"{outDir}/checkpoint")
    except FileExistsError:
        print(f"Directory: {outDir}/checkpoint Exists")
    finalFlag = False


if finalFlag == False:
    for s in range(nSeeds):
        if s < resume:
            """
            """
        elif (s == resume) and (resume >= 0):
            print(f"Loading Previously Collected Data")
            d0 = loadmat(f"{outDir}/checkpoint/entropyStudy.checkpoint.mat")
            d1 = loadmat(f"{outDir}/checkpoint/fitnessStudy.checkpoint.mat")
            d2 = loadmat(f"{outDir}/checkpoint/radialData.checkpoint.mat")
            d3 = loadmat(f"{outDir}/checkpoint/fitnessData.checkpoint.mat")
            d4 = loadmat(f"{outDir}/checkpoint/powerData.checkpoint.mat")
            entropyStudy = d0["dat"]
            fitnessStudy = d1["dat"]
            radialData = d2["dat"]
            fitnessData  = d3["dat"]
            scatteringPowerData = d4["dat"]
            print(f"Resuming from seed {resume}")

            
            
        else:
            print(f"Seed: {s}")
            for br in range(nBirthRates):
                
                    evolutionRun(s, br, parsedArgs)
                    

            print(f"saving data checkpoint: {s}")
            savemat(f"{outDir}/checkpoint/entropyStudy.checkpoint.mat",{"dat":entropyStudy, "checkpoint":s})
            savemat(f"{outDir}/checkpoint/fitnessStudy.checkpoint.mat",{"dat":fitnessStudy, "checkpoint":s})
            savemat(f"{outDir}/checkpoint/radialData.checkpoint.mat",{"dat":radialData, "checkpoint":s})
            savemat(f"{outDir}/checkpoint/fitnessData.checkpoint.mat",{"dat":fitnessData, "checkpoint":s})
            savemat(f"{outDir}/checkpoint/powerData.checkpoint.mat",{"dat":scatteringPowerData, "checkpoint":s})



    savemat(f"{outDir}/entropyStudy.final.mat", {"dat":entropyStudy})
    savemat(f"{outDir}/fitnessStudy.final.mat", {"dat":fitnessStudy})
    savemat(f"{outDir}/radialData.final.mat", {"dat":radialData})
    savemat(f"{outDir}/fitnessData.final.mat",{"dat":fitnessData})
    savemat(f"{outDir}/powerData.final.mat",{"dat":scatteringPowerData})


    shutil.rmtree(f"{outDir}/checkpoint/")

else:
    print("Loading Previously Collected Data - Completed Simulation")
    d0 = loadmat(f"{outDir}/entropyStudy.final.mat")
    d1 = loadmat(f"{outDir}/fitnessStudy.final.mat")
    d2 = loadmat(f"{outDir}/radialData.final.mat")
    d3 = loadmat(f"{outDir}/fitnessData.final.mat")
    d4 = loadmat(f"{outDir}/powerData.final.mat")
    entropyStudy = d0["dat"]
    fitnessStudy = d1["dat"]
    radialData = d2["dat"]
    fitnessData  = d3["dat"]
    scatteringPowerData = d4["dat"]



for birthRate in range(nBirthRates):
    bR = (birthRate + 1)/nBirthRates
    bR = np.round(bR, 2)
    s = np.std(entropyStudy[birthRate,:,:], axis=0)
    m = np.mean(entropyStudy[birthRate,:,:], axis=0)
    x = np.arange(nGenerations+1)
    #plt.plot(x, m, 'o-', label=f'Birth Rate: {bR}')
    #plt.fill_between(x, m-s, m+s, alpha = 0.2, zorder=-1, color='maroon')
    plt.errorbar(x,m, yerr=s, fmt='o-', markersize=4,capsize=2, label=f'Birth Rate: {bR}', alpha=0.5)
    plt.legend(loc='lower left')
    plt.xlabel("Generations")
    plt.ylabel("Entropy")
plt.title("Entropy Decay During Evolutionary Algorithm")
plt.savefig(f'{outDir}/Entropy-Decay-During-Evolutionary-Algorithm.png', dpi=300)
plt.close()


for birthRate in range(nBirthRates):
    bR = (birthRate + 1)/nBirthRates
    bR = np.round(bR, 2)
    m = np.mean(fitnessStudy[birthRate,:,:,3], axis=0)
    s = np.std(fitnessStudy[birthRate,:,:,3], axis=0)
    x = np.arange(nGenerations+1)

    #plt.plot(x, m, 'o-', label=f'Birth Rate: {bR}')
    #plt.fill_between(x, m-s, m+s, alpha = 0.2, zorder=-1, color='maroon')
    plt.errorbar(x,m, yerr=s, fmt='o-', markersize=4,capsize=2, label=f'Birth Rate: {bR}', alpha=0.5)
    plt.legend(loc='upper left')
    plt.ylim((0,1))
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
temp0 = []
temp1 = []
for n in range(nGenerations+1):
    temp0.append(np.max(fitnessStudy[:,:,n,0]))
    temp1.append(np.min(fitnessStudy[:,:,n,1]))
plt.fill_between(np.arange(nGenerations+1), temp0, temp1, alpha = 0.1, zorder=-3, color='black')
plt.title("Evolution of Average Fitness")
plt.savefig(f'{outDir}/Fitness-During-Evolutionary-Algorithm.png', dpi=300)
plt.close()