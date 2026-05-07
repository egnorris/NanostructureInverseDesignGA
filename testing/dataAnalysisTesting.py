import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../modules")
import DataAnalysis as da
import InverseDesign as invdes


#need to be parsed as either arguments or an input file
nTriangles = 10
nCircles = 10
nRectangles = 10
nPolygons = 10
nNgons = 10
nGenerations = 100
nVertices = 24
minWavelength = 300
maxWavelength = 800
bPrecision = 12
saeWeight = 1
sseWeight = 1
spectrumFileName = "input/objScatteredPower0.txt"
outDir = '.'


nBirthRates = 5
nSeeds = 10






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
            rMin=10,
            rMax=80,
            domain=(180,180),
            sigma=5,
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
        print(f"Seed: {seed}")
        print(f"Birth Rate: {bR}")
        print(f"Generation 0 - {pop.nProfiles} Profiles Generated")
        print(f"Entropy:                {H}")
        print(f"Entropy (Normalized):   {normH}")
        print(f"Maximum Fitness:        {np.round(np.max(pop.fitness),3)}")
        print(f"Average Fitness:        {np.round(np.mean(pop.fitness),3)}")
        print(f"Fitness Variance:       {np.round(np.var(pop.fitness),3)}")
        
        da.plot6("Generation 0 Top Performers", outDir,pop)

        for n in range(nGenerations):
            #####################################################################################################
            #   Create generation n+1 population 
            #####################################################################################################
            pop.update(birthRate=bR)
            #####################################################################################################
            #   Data Analysis - Generation n+1
            #####################################################################################################
            H, normH = da.getShannonEntropy(pop.chromosomes)
            print(f"Seed: {seed}")
            print(f"Birth Rate: {bR}")
            print(f"Generation {n+1} - {pop.nGenerated} New Profiles Generated")
            print(f"Entropy:                {H}")
            print(f"Entropy (Normalized):   {normH}")
            print(f"Maximum Fitness:        {np.round(np.max(pop.fitness),3)}")
            print(f"Average Fitness:        {np.round(np.mean(pop.fitness),3)}")
            print(f"Fitness Variance:       {np.round(np.var(pop.fitness),3)}")
            