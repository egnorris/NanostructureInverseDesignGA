import Support
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import itertools as iter
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"]="0"

global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords

class Population():
    def __init__(self, nVertices, modelDirectory, **kwargs):
        self.nVertices = nVertices
        self.trainedModelPath = modelDirectory
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
        #fitness
        self.lambdaMin = Support.getkwarg(kwargs, defaultKwargs["minWavelength"], keywords["minWavelength"])
        self.lambdaMax = Support.getkwarg(kwargs, defaultKwargs["maxWavelength"], keywords["maxWavelength"])
        self.saeWeight = Support.getkwarg(kwargs, defaultKwargs["saeWeight"], keywords["saeWeight"])
        self.sseWeight = Support.getkwarg(kwargs, defaultKwargs["sseWeight"], keywords["sseWeight"])

        self.profGen = pg.ProfileGeneration(self.nVertices,rMin=self.rMin,rMax=self.rMax,d=self.dom,s=self.s,p=self.p)
        self.dlModels = dl.DeepLearning(self.trainedModelPath,l=self.l,m=self.m,f=self.f)
        self.dlModels.loadModels()
    
    def readObjective(self, spectrumFileName):
        self.objScatteredPower = np.loadtxt(spectrumFileName)
        self.tss = np.var(self.objScatteredPower) * len(self.objScatteredPower)
    
    def __updateInitialPopulation(self, n):
            self.profGen.arrayConversion()
            self.profGen.encodePolygon()
            self.chromosomes[n]     = self.profGen.binaryPolygon
            self.polar[n, :, 0]     = self.profGen.r
            self.polar[n, :, 1]     = self.profGen.t
            self.cartesian[n, :, 0] = self.profGen.x
            self.cartesian[n, :, 1] = self.profGen.y
            self.images[n,:,:]      = self.profGen.smoothedImage
            self.serialNumber[n]    = self.ProfilesGenerated
            self.ProfilesGenerated  += 1
            return n + 1
    
    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def __fit(self, x, t=0, s=0.25):

        a = np.exp(np.log(s)/(-t+1))
        b = -np.log(s)/(-t+1)
        if x <= t:
            return (s/self.__sigmoid(t))*self.__sigmoid(x)
        else:
            return a*np.exp(b*x)
    
    def getFitness(self):
        k0= self.minWavelengthIdx
        k1= self.maxWavelengthIdx
        y= self.objScatteredPower[k0:k1]
        self.fitness= np.zeros(self.nProfiles)
        self.residual= np.zeros((self.nProfiles, len(y)))
        self.sumSquaredError= np.zeros(self.nProfiles)
        self.sumAbsoluteError= np.zeros(self.nProfiles)
        for n in range(self.nProfiles):
            fx = self.scatteredPower[n, k0:k1]
            self.residual[n,:] = y - fx
            self.sumSquaredError[n] = np.sum(self.residual[n,:]**2)
            self.sumAbsoluteError[n] = np.sum(np.abs(self.residual[n,:]))
            saeFitness = self.__fit(1 - self.sumAbsoluteError[n]/self.tss)
            sseFitness = self.__fit(1 - self.sumSquaredError[n]/self.tss)
            self.fitness[n] = saeFitness*self.saeWeight + sseFitness*self.sseWeight
            self.fitness[n] = self.fitness[n] / (self.saeWeight + self.sseWeight)
    
    def __sortChromosomes(self, sortIndices):
        ch = np.zeros(self.nProfiles)
        ch = list(ch)
        for k in range(len(self.chromosomes)):
            ch[k] = self.chromosomes[sortIndices[k]]
        self.chromosomes = ch
    
    def sortPopulation(self):
        sortIndices = np.argsort(self.fitness)
        sortIndices = np.flip(sortIndices)
        self.__sortChromosomes(sortIndices)
        self.polar              = self.polar[sortIndices, :, :]
        self.cartesian          = self.cartesian[sortIndices, :, :]
        self.images             = self.images[sortIndices, :, :]
        self.serialNumber       = self.serialNumber[sortIndices]
        self.multipoles         = self.multipoles[:, sortIndices, :]
        self.scatteredPower     = self.scatteredPower[sortIndices, :]
        self.fitness            = self.fitness[sortIndices]
        self.residual           = self.residual[sortIndices, :]
        self.sumSquaredError    = self.sumSquaredError[sortIndices]
        self.sumAbsoluteError   = self.sumAbsoluteError[sortIndices]
    
    def initialize(self, nT=0, nR=0, nC=0, nN=0, nP=0):
        nProfiles = nT + nR + nC + nN + nP
        #####################################################################################################
            #Initialize Population Arrays
        #####################################################################################################
        self.chromosomes        = np.zeros(nProfiles)
        self.chromosomes        = list(self.chromosomes)
        self.polar              = np.zeros((nProfiles, self.nVertices, 2))
        self.cartesian          = np.zeros((nProfiles, self.nVertices+1, 2))
        self.images             = np.zeros((nProfiles, self.dom[0], self.dom[1]))
        self.serialNumber       = np.zeros(nProfiles)
        self.wavelengths        = np.linspace(300,800,101)
        self.multipoles         = np.zeros((int(len(self.l)*len(self.f)), nProfiles, len(self.wavelengths)))
        self.scatteredPower     = np.zeros((nProfiles, 101))
        self.ProfilesGenerated  = 0
        self.nProfiles          = nProfiles
        self.minWavelengthIdx   = list(self.wavelengths).index(self.lambdaMin)
        self.maxWavelengthIdx   = list(self.wavelengths).index(self.lambdaMax)
        #####################################################################################################
            #Generate Initial Population Profiles
        #####################################################################################################
        n = 0
        for k in range(nT):
            #generate triangles
            self.profGen.triangleGenerator()
            n = self.__updateInitialPopulation(n)
        for k in range(nR):
            #generate rectangles
            self.profGen.rectangleGenerator()
            n = self.__updateInitialPopulation(n)
        for k in range(nC):
            #generate circles
            self.profGen.circleGenerator()
            n = self.__updateInitialPopulation(n)
        for k in range(nP):
            #generate randomized polygons
            self.profGen.polygonGenerator()
            n = self.__updateInitialPopulation(n)
        for k in range(nN):
            #generate n-sided polygons
            self.profGen.ngonGenerator(n=np.random.randint(3,10))
            n = self.__updateInitialPopulation(n)
        #####################################################################################################
            #Evaluate Initial Population Profiles
        #####################################################################################################
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.getFitness()
        self.sortPopulation()
    
    def roulette(self,f,i):
        p = f / np.sum(f)
        temp = 0
        x = np.random.rand()
        j = 0
        for k in range(len(p)):
            temp += p[k]
            
            if temp > x:
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

    def __updateNewPopulation(self, n):
            self.update_chromosomes[n]     = self.profGen.binaryPolygon
            self.update_polar[n, :, 0]     = self.profGen.r
            self.update_polar[n, :, 1]     = self.profGen.t
            self.update_cartesian[n, :, 0] = self.profGen.x
            self.update_cartesian[n, :, 1] = self.profGen.y
            self.update_images[n,:,:]      = self.profGen.smoothedImage
            self.update_serialNumber[n]    = self.ProfilesGenerated
            self.ProfilesGenerated  += 1
            return n + 1

    def update(self, birthRate = 1.0):

        #####################################################################################################
            #Initialize Next Generation Population Arrays
        #####################################################################################################
        nProfiles = self.nProfiles
        self.update_chromosomes         = np.zeros(nProfiles)
        self.update_polar               = np.zeros((nProfiles, self.nVertices, 2))
        self.update_cartesian           = np.zeros((nProfiles, self.nVertices+1, 2))
        self.update_images              = np.zeros((nProfiles, self.dom[0], self.dom[1]))
        self.update_serialNumber        = np.zeros(nProfiles)
        self.update_multipoles          = np.zeros((int(len(self.l)*len(self.f)), nProfiles, len(self.wavelengths)))
        self.update_scatteredPower      = np.zeros((nProfiles, 101))
        #####################################################################################################
            #Save the top performer from the previous generation for later
        #####################################################################################################
        top_chromosome      = self.chromosomes [0]
        top_polar           = self.polar[0,:,:]
        top_cartesian       = self.cartesian[0,:,:]
        top_image           = self.images[0,:,:]
        top_serialNumber    = self.serialNumber[0]
        #####################################################################################################
            #generate the children of the new population
        #####################################################################################################
        n = 0
        self.nGenerated = int(self.nProfiles*birthRate/2)
        self.selectParentPairs()
        for k in range(self.nGenerated):
            p0 = self.chromosomes[self.parentPairs[k, 0]]
            p1 = self.chromosomes[self.parentPairs[k, 1]]
            g = go.GeneticOperations(p0, p1, mR=self.mR, cP=self.cP)
            g.operate()
            self.profGen.decodeChromosome(g.c0)
            n = self.__updateNewPopulation(n)
            self.profGen.decodeChromosome(g.c1)
            n = self.__updateNewPopulation(n)
        #####################################################################################################
            #select members of previous population to retain
        #####################################################################################################
        self.nRetained = int(self.nProfiles - 2*self.nGenerated)
        self.selectRetainedPopulation()
        for k in range(self.nRetained):
            p = self.chromosomes[self.retainedPopulation[k]]
            #this step isn't necessary in general but allows the updatenewPopulation() function to be used
            self.profGen.decodeChromosome(p)
            n = self.__updateNewPopulation(n)
            #this isn't a new profile so remove it from the count and set the proper serial number
            self.ProfilesGenerated -= 1
            self.update_serialNumber[n-1] = self.serialNumber[k]
        #####################################################################################################
            #check if the previous top performer is included in the update
        #####################################################################################################
        if 0 in self.retainedPopulation:
            """
            """
        else:
            self.update_chromosomes[-1]     = self.chromosomes[0]
            self.update_polar[-1, :, :]     = self.polar[0, :, :]
            self.update_cartesian[-1, :, :] = self.cartesian[0, :, :]
            self.update_images[-1,:,:]      = self.images[0,:,:]
            self.update_serialNumber[-1]    = self.serialNumber[0]
        #####################################################################################################
            #Update the population 
        #####################################################################################################
        self.chromosomes     = self.update_chromosomes
        self.polar           = self.update_polar
        self.cartesian       = self.update_cartesian
        self.images          = self.update_images
        self.serialNumber    = self.update_serialNumber
        #####################################################################################################
            #Evaluate updated Population
        #####################################################################################################
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.getFitness()
        self.sortPopulation()