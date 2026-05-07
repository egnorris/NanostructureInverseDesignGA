import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("../modules")
import DataAnalysis as da
import InverseDesign as invdes
from scipy.io import savemat


#need to be parsed as either arguments or an input file
nTriangles = 10
nCircles = 10
nRectangles = 10
nPolygons = 10
nNgons = 10

nVertices = 24
minWavelength = 300
maxWavelength = 800
bPrecision = 12
saeWeight = 1
sseWeight = 1
sigma = 5
spectrumFileName = "input/objScatteredPower0.txt"
outDir = 'output'
domain = (180, 180)
rMin = 20
rMax = 80

nBirthRates = 5
nSeeds = 200
nGenerations = 50

w = np.linspace(300,800,101)
populationDict = {
    "birthRates": ((np.arange(nBirthRates)+1)/nBirthRates),
    "seeds": np.arange(nSeeds),
    "wavelengths": np.linspace(300,800,101),
    "evaluationWavelengths": [minWavelength, maxWavelength],
    "nVertices": nVertices,
    "radialRange": [rMin, rMax],
    "sigma": sigma,
    "domain": domain
    }


EntropyStudy = np.zeros((nBirthRates, nSeeds, nGenerations+1))

for birthRate in range(nBirthRates):
    bR = (birthRate + 1)/nBirthRates
    for seed in range(nSeeds):
        np.random.seed(seed)
        #####################################################################################################
            #Initialize Population Class
        #####################################################################################################
        pop = invdes.Population(
            nVertices=nVertices,
            modelDirectory="/media/work/evan/deep_learning_data/trained_models",
            rMin=rMin,
            rMax=rMax,
            domain=domain,
            sigma=sigma,
            precision=bPrecision,
            mutationRate=0.1,
            crossoverPoints=1,
            degree=[1,2,2],
            order=[1,1,2],
            fields=["E", "H"],
            minWavelength=minWavelength,
            maxWavelength=maxWavelength,
            saeWeight=saeWeight,
            sseWeight=sseWeight
            )
        #####################################################################################################
        #   Read the objective spectrum from file
        #####################################################################################################
        pop.readObjective(spectrumFileName=spectrumFileName)

        #####################################################################################################
        #   Create generation 0 population 
        #####################################################################################################
        pop.initialize(
            nT=nTriangles,
            nC=nCircles,
            nR=nRectangles,
            nP=nPolygons,
            nN=nNgons)
        #####################################################################################################
        #   Data Analysis - Generation 0
        #####################################################################################################
        H, normH = da.getShannonEntropy(pop.chromosomes)
        EntropyStudy[birthRate, seed, 0] = normH
        print(f"Seed: {seed}")
        print(f"Birth Rate: {bR}")
        print(f"Generation 0 - {pop.nProfiles} Profiles Generated")
        print(f"Entropy:                {H}")
        print(f"Entropy (Normalized):   {normH}")
        print(f"Maximum Fitness:        {np.round(np.max(pop.fitness),3)}")
        print(f"Average Fitness:        {np.round(np.mean(pop.fitness),3)}")
        print(f"Fitness Variance:       {np.round(np.var(pop.fitness),3)}")
        
        if (seed % 25) == 0:
            da.plot6(f"Seed {seed} - Birth Rate {bR} - Generation {0} Top Performers", outDir,pop)
        populationDict[f'dat-BirthRate-{bR}-Seed-{seed}-Generation{0}'] = da.packageDictionary(pop)
        

        for n in range(nGenerations):
            #####################################################################################################
            #   Create generation n+1 population 
            #####################################################################################################
            pop.update(birthRate=bR)
            #####################################################################################################
            #   Data Analysis - Generation n+1
            #####################################################################################################
            H, normH = da.getShannonEntropy(pop.chromosomes)
            EntropyStudy[birthRate, seed, n+1] = normH
            print(f"Seed: {seed}")
            print(f"Birth Rate: {bR}")
            print(f"Generation {n+1} - {pop.nGenerated*2} New Profiles Generated")
            print(f"Entropy:                {H}")
            print(f"Entropy (Normalized):   {normH}")
            print(f"Maximum Fitness:        {np.round(np.max(pop.fitness),3)}")
            print(f"Average Fitness:        {np.round(np.mean(pop.fitness),3)}")
            print(f"Fitness Variance:       {np.round(np.var(pop.fitness),3)}")
            if (seed % 25) == 0:
                if (n % 5) == 0:
                    da.plot6(f"Seed {seed} - Birth Rate {bR} - Generation {n+1} Top Performers", outDir,pop)
            populationDict[f'dat-BirthRate-{bR}-Seed-{seed}-Generation{n+1}'] = da.packageDictionary(pop)

savemat("output/PopulationData.mat", populationDict)


for birthRate in range(nBirthRates):
    bR = (birthRate + 1)/nBirthRates
    s = np.var(EntropyStudy[birthRate,:,:], axis=0)
    m = np.mean(EntropyStudy[birthRate,:,:], axis=0)
    x = np.arange(nGenerations+1)
    plt.plot(x, m, 'o-', label=f'Birth Rate: {bR}')
    plt.fill_between(x, m-s, m+s, alpha = 0.2, zorder=-1)
    plt.legend(loc='lower left')
    plt.xlabel("Generations")
    plt.ylabel("Entropy")
plt.title("Entropy Decay During Evolutionary Algorithm")
plt.savefig(f'{outDir}/Entropy-Decay-During-Evolutionary-Algorithm.png', dpi=300)
plt.close()