import Support
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import itertools as iter
import pandas as pd

global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords

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
            self.profGen.generate('tri')
            self.objectivePoly = self.profGen.smoothedImage
            temp[0, :, :] = self.profGen.smoothedImage
            self.dlModels.getModelPrediction(temp)
            objective = self.dlModels.scatteredPowerPredictions

        self.objective = objective[0, :]


        #initialize population
        
        for k in range(self.nProfiles):
            if k < 3*self.nProfiles/4:
                self.profGen.generate('cir')
            else:
                self.profGen.generate('tri')
            self.images[k, :, :] = self.profGen.smoothedImage
            self.chromosomes[k] = self.profGen.binaryPolygon
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.fitness = self.getFitness(self.images, self.scatteredPower)

        
    def getFitness(self, x, y):
        f = np.zeros(np.shape(x)[0])
        for i in range(np.shape(x)[0]):
            mse = 0
            for k in range(len(self.wavelengths)):
                mse += np.sqrt((self.objective[k] - y[i,k])**2)
            mse = mse / len(self.wavelengths)
            f[i] = 100*np.exp(-25*mse)
        return f


    