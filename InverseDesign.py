import Support
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import itertools as iter
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords


def scale(x):
    return 1/x
        

class Population():
    def __init__(self, nProfiles, nVertices, path, objective=None, **kwargs):

        #required Arguments
        self.nVertices = nVertices
        self.nProfiles = nProfiles

        #Keyword Arguments
        #Profile Generation
        self.rMin = Support.getkwarg(kwargs, defaultKwargs["rMin"], keywords["rMin"])
        self.rMax = Support.getkwarg(kwargs, defaultKwargs["rMax"], keywords["rMax"])
        self.dom = Support.getkwarg(kwargs, defaultKwargs["d"], keywords["d"])
        self.s = Support.getkwarg(kwargs, defaultKwargs["s"], keywords["s"])
        self.p = Support.getkwarg(kwargs, defaultKwargs["p"], keywords["p"])
        #Genetic Operations
        self.mR = Support.getkwarg(kwargs, defaultKwargs["mR"], keywords["mR"])
        self.cP = Support.getkwarg(kwargs, defaultKwargs["cP"], keywords["cP"])
        #Deep Learning
        self.l = Support.getkwarg(kwargs, defaultKwargs["l"], keywords["l"])
        self.m = Support.getkwarg(kwargs, defaultKwargs["m"], keywords["m"])
        self.f = Support.getkwarg(kwargs, defaultKwargs["f"], keywords["f"])
        self.path = path

        #setup population arrays
        self.chromosomes = np.zeros(nProfiles)
        self.chromosomes = list(self.chromosomes)
        self.fitness = np.zeros(nProfiles)
        self.images = np.zeros((nProfiles, self.dom[0], self.dom[1]))
        self.wavelengths = np.linspace(300,800,101)
        self.multipoles = np.zeros((int(len(self.l)*len(self.f)), nProfiles, len(self.wavelengths)))
        self.scatteredPower = np.zeros((nProfiles, 101))

        #add external classes for use later
        self.profGen = pg.ProfileGeneration(self.nVertices,rMin=self.rMin,rMax=self.rMax,d=self.dom,s=self.s,p=self.p)
        self.dlModels = dl.DeepLearning(self.path,l=self.l,m=self.m,f=self.f)
        self.dlModels.loadModels()
        

        #setup objective
        if objective == None:
            temp = np.zeros((1, self.dom[0], self.dom[1]))
            self.profGen.generate('pol')
            self.objectivePoly = self.profGen.smoothedImage
            temp[0, :, :] = self.profGen.smoothedImage
            self.dlModels.getModelPrediction(temp)
            objective = self.dlModels.scatteredPowerPredictions

        self.objective = objective[0, :]


        #initialize population
        
        for k in range(self.nProfiles):
            if k < self.nProfiles/2:
                self.profGen.generate('tri')
            else:
                self.profGen.generate('cir')
            self.images[k, :, :] = self.profGen.smoothedImage
            self.chromosomes[k] = self.profGen.binaryPolygon
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.getFitness()


    def getResiduals(self):
        x = self.objective
        y = self.scatteredPower
        self.integral = np.zeros(self.nProfiles)
        self.meanAbsoluteError = np.zeros(self.nProfiles)
        self.meanRelativeError = np.zeros(self.nProfiles)
        self.meanSquaredError = np.zeros(self.nProfiles)
        self.rootMeanSquaredError = np.zeros(self.nProfiles)
        for n in range(self.nProfiles):
            distance = np.zeros(len(self.wavelengths))
            absoluteError = np.zeros(len(self.wavelengths))
            relativeError = np.zeros(len(self.wavelengths))
            meanSquaredError= np.zeros(len(self.wavelengths))
            rootMeanSquaredError = np.zeros(len(self.wavelengths))
            for w in range(len(self.wavelengths)):
                distance[w] = np.sqrt(np.abs(x[w]**2 - y[n,w]**2))
                absoluteError[w] = np.abs(x[w] - y[n,w])
                relativeError[w] = absoluteError[w]/x[w]
                meanSquaredError[w] = (x[w] - y[n,w])**2
                rootMeanSquaredError[w] = (x[w] - y[n,w])**2

            self.integral[n] = np.round(np.sum(distance), 5)
            self.meanAbsoluteError[n] = np.round(np.mean(absoluteError), 5)
            self.meanRelativeError[n] = np.round(np.mean(relativeError), 5)
            self.meanSquaredError[n] = np.round(np.mean(meanSquaredError), 5)
            self.rootMeanSquaredError[n] = np.round(np.sqrt(np.mean(rootMeanSquaredError)), 5)

    
    def getFitness(self):
        self.getResiduals()
        self.fitness = scale(self.integral)
        




    def roulette(self,f,i):
        p = f / np.sum(f)
        temp = 0
        x = np.random.rand()
        j = 0
        for k in range(len(p)):
            temp += p[k]
            
            if temp >= x:
                j = i[k]
                f = np.delete(f, k)
                i = np.delete(i, k)
                return (j, f, i)
    
    def selectParentPairs(self):
        f = self.fitness
        self.indices = np.linspace(0,len(f)-1,len(f), dtype=int)
        i = self.indices
        n = self.nGenerated
        self.parentPairs = np.zeros((n, 2), dtype=int)
        for k in range(n):
            j0, f, i = self.roulette(f, i)
            j1, f, i = self.roulette(f, i) 
            self.parentPairs[k, :] = [j0, j1]


    def selectRetainedPopulation(self):
        f = self.fitness
        self.indices = np.linspace(0,len(f)-1,len(f), dtype=int)
        i = self.indices
        n = self.nRetained
        self.retainedPopulation = np.zeros(n, dtype=int)
        for k in range(n):
            j0, f, i = self.roulette(f, i)
            self.retainedPopulation[k] = j0


    def newGeneration(self, growthRate = 1/3):
        newChromosomes = np.zeros(self.nProfiles)
        newChromosomes = list(newChromosomes)
        newImages = np.zeros((self.nProfiles, self.dom[0], self.dom[1]))

        

        n = 0
        self.nGenerated = int(self.nProfiles*growthRate)
        self.nRetained = int(self.nProfiles - 2*self.nGenerated)
        print(f'Growth Rate: {growthRate}')
        print(f'Generating {2*self.nGenerated} new chromosomes')
        print(f'{self.nRetained} old chromosomes will be retained')

        self.selectParentPairs()
        self.selectRetainedPopulation()

        for k in range(self.nGenerated):
            p0 = self.chromosomes[self.parentPairs[k, 0]]
            p1 = self.chromosomes[self.parentPairs[k, 1]]
            g = go.GeneticOperations(p0, p1)
            g.operate()
            self.profGen.decodeChromosome(g.c0Mutated)
            i0 = self.profGen.smoothedImage
            self.profGen.decodeChromosome(g.c1Mutated)
            i1 = self.profGen.smoothedImage

            newChromosomes[n] = g.c0Mutated
            newImages[n, :, :] = i0
            n += 1
            newChromosomes[n] = g.c1Mutated
            newImages[n, :, :] = i1
            n += 1

        for k in range(self.nRetained):
            newChromosomes[n] = self.chromosomes[self.retainedPopulation[k]]
            newImages[n, :, :] = self.images[self.retainedPopulation[k], :, :]
            n += 1
        
        self.chromosomes = newChromosomes
        self.images = newImages
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.getFitness()

            




    