import sys
import argparse
import os
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

sys.path.append("../modules")

import Support
import runInverseDesign as run
import DataAnalysis
    
def evolutionRun(s, br, PopulationDict, EntropyStudy, parsedArgs):
    checkpoint = parsedArgs['seedCheckpoint']
    #reset random number generator
    np.random.seed(s)
    outDir =  parsedArgs['outDir']
    nGenerations = parsedArgs['numGenerations']
    birthRate= (br + 1)/parsedArgs['numBirthRate']
    print(f"    Birth Rate: {birthRate}")
    genSave = parsedArgs['genSave']
    pop = run.setupPopulation(**parsedArgs)
    H, normH = DataAnalysis.getShannonEntropy(pop.chromosomes)
    EntropyStudy[br, s, 0] = normH

    if (s % checkpoint) == 0:
        DataAnalysis.plot6(f"Seed {s} - Birth Rate {birthRate} - Generation {0} Top Performers", outDir,pop)
    populationDict[f'dat-BirthRate-{birthRate}-Seed-{s}-Generation{0}'] = DataAnalysis.packageDictionary(pop)
    
    for n in range(nGenerations):
        pop = run.updatePopulation(pop, birthRate=birthRate)
        H, normH = DataAnalysis.getShannonEntropy(pop.chromosomes)
        if ((n+1) % genSave) == 0:
            if (s % checkpoint) == 0:
                DataAnalysis.plot6(f"Seed {s} - Birth Rate {birthRate} - Generation {n+1} Top Performers",outDir,pop)
        populationDict[f'dat-BirthRate-{birthRate}-Seed-{s}-Generation{n+1}'] = DataAnalysis.packageDictionary(pop)
        EntropyStudy[br, s, n+1] = normH
    return PopulationDict, EntropyStudy


parser = argparse.ArgumentParser(description="Run Genetic Algorithm Inverse Design")
parser.add_argument('-outDir', type=str, required=False, default='.')
parser.add_argument('-numGenerations', type=int, required=False, default=10)
parser.add_argument('-numSeed', type=int, required=False, default=10)
parser.add_argument('-numBirthRate', type=int, required=False, default=5)
parser.add_argument('-genSave', type=int, required=False, default=5)
parser.add_argument('-seedCheckpoint', type=int, required=False, default=10)

parser.add_argument('-nV', type=int, required=False)
parser.add_argument('-modelDir', type=str, required=False)
parser.add_argument('-targetFile', type=str, required=False)
parser.add_argument('-rMin', type=int, required=False)
parser.add_argument('-rMax', type=int, required=False)
parser.add_argument('-d', type=type(()), required=False)
parser.add_argument('-s', type=int, required=False)
parser.add_argument('-p', type=int, required=False)
parser.add_argument('-mR', type=float, required=False)
parser.add_argument('-cP', type=int, required=False)
parser.add_argument('-l', type=type([]), required=False)
parser.add_argument('-m', type=type([]), required=False)
parser.add_argument('-f', type=type([]), required=False)
parser.add_argument('-w0', type=int, required=False)
parser.add_argument('-w1', type=int, required=False)
parser.add_argument('-saeF', type=int, required=False)
parser.add_argument('-sseF', type=int, required=False)
parser.add_argument('-nT', type=int, required=False)
parser.add_argument('-nC', type=int, required=False)
parser.add_argument('-nR', type=int, required=False)
parser.add_argument('-nP', type=int, required=False)
parser.add_argument('-nN', type=int, required=False)
parsedArgs = parser.parse_args().__dict__

outDir = parsedArgs['outDir']

try:
    os.mkdir(f"{outDir}")
except FileExistsError:
    print(f"Directory: {outDir} Exists")

try:
    os.mkdir(f"{outDir}/checkpoint")
except FileExistsError:
    print(f"Directory: {outDir}/checkpoint Exists")

nBirthRates = parsedArgs['numBirthRate']
nSeeds = parsedArgs['numSeed']
nGenerations = parsedArgs['numGenerations']
checkpoint = parsedArgs['seedCheckpoint']
global populationDict
global EntropyStudy

populationDict = {
    "birthRates": ((np.arange(nBirthRates)+1)/nBirthRates),
    "seeds": np.arange(nSeeds, step=checkpoint),
    }
print(populationDict.keys())
entropyStudy = np.zeros((nBirthRates, nSeeds, nGenerations+1))

a = loadmat(f"{outDir}/checkpoint/entropyData.checkpoint.mat")

print()
for s in range(nSeeds):
    
    if s < a["checkpoint"][0][0]:
        """
        """
    elif s == a["checkpoint"][0][0]:
        print(f'Resuming from checkpoint: {a["checkpoint"][0][0]}')
        entropyStudy = a["dat"]
        b = loadmat(f"{outDir}/checkpoint/populationData.checkpoint.mat")
        populationDict.update(b)
        
    else:
        print(f"Seed: {s}")
        for br in range(nBirthRates):
                P, entropyStudy = evolutionRun(s, br, populationDict, entropyStudy, parsedArgs)
                populationDict.update(P)
                

        populationDict["checkpoint"] = s
        print(f"saving data checkpoint: {s}")
        savemat(f"{outDir}/checkpoint/populationData.checkpoint.mat",populationDict)
        savemat(f"{outDir}/checkpoint/entropyData.checkpoint.mat",{"dat":entropyStudy, "checkpoint":s})


savemat(f"{outDir}/populationData.final.mat", populationDict)
savemat(f"{outDir}/entropyData.final.mat", {"dat":entropyStudy})

for birthRate in range(nBirthRates):
    bR = (birthRate + 1)/nBirthRates
    s = np.var(entropyStudy[birthRate,:,:], axis=0)
    m = np.mean(entropyStudy[birthRate,:,:], axis=0)
    x = np.arange(nGenerations+1)
    plt.plot(x, m, 'o-', label=f'Birth Rate: {bR}')
    plt.fill_between(x, m-s, m+s, alpha = 0.2, zorder=-1)
    plt.legend(loc='lower left')
    plt.xlabel("Generations")
    plt.ylabel("Entropy")
plt.title("Entropy Decay During Evolutionary Algorithm")
plt.savefig(f'{outDir}/Entropy-Decay-During-Evolutionary-Algorithm.png', dpi=300)
plt.close()