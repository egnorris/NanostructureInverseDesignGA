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

global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords


def scale(a,x):
    return np.exp(-a*x)
        

class Population():
    def __init__(self, nVertices, modelDirectory, fitnessType='gap', **kwargs):

        #required Arguments
        self.nVertices = nVertices
        self.fitnessType = fitnessType
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
        self.path = modelDirectory

    
        #add external classes for use later
        self.profGen = pg.ProfileGeneration(self.nVertices,rMin=self.rMin,rMax=self.rMax,d=self.dom,s=self.s,p=self.p)
        self.dlModels = dl.DeepLearning(self.path,l=self.l,m=self.m,f=self.f)
        self.dlModels.loadModels()
    
    def defineObjective(self, profileType=None, spectrum=None, termsF=None):
        if (profileType != None) and (spectrum == None):
            temp = np.zeros((1, self.dom[0], self.dom[1]))
            self.profGen.generate(profileType)
            if profileType == 'fou':
                if termsF == None:
                    self.profGen.fourierGenerator(np.random.randint(2,5))
                    self.profGen.arrayConversion()
                    self.profGen.encodePolygon()
                else:
                    self.profGen.fourierGenerator(termsF)
                    self.profGen.arrayConversion()
                    self.profGen.encodePolygon()

            temp[0, :, :] = self.profGen.smoothedImage
            self.dlModels.getModelPrediction(temp)
            self.objImage = self.profGen.smoothedImage
            self.objScatteredPower = self.dlModels.scatteredPowerPredictions[0, :]
        elif (profileType == None):
                self.objScatteredPower = spectrum
        
        else:
            raise Exception(f'Objective Spectrum cannot be defined')
 
    def __displayTotals(self):
        print("--------------------------------------------------------------------------")
        print(f"Initial Generation Makeup")
        print("--------------------------------------------------------------------------")
        print(f"{self.nT} Triangles")
        print(f"{self.nR} Rectangles")
        print(f"{self.nC} Circles")
        print(f"{self.nF} Fourier Series Polygons")
        print(f"{self.nF} Randomized Polygons")
        print(f"{self.nProfiles} Total Profiles")
        

    def initialize(self, nT=0, nR=0, nC=0, nF=0, nP=0, termsF = None):
        #initialize population
        nProfiles = nT + nR + nC + nF + nP
        if nProfiles < 16:
            raise Exception(f'Not enough profiles, nProfiles = nT + nR + nC + nF + nP should be at least 16')
        #setup population arrays
        self.chromosomes = np.zeros(nProfiles)
        self.chromosomes = list(self.chromosomes)
        self.fitness = np.zeros(nProfiles)
        self.images = np.zeros((nProfiles, self.dom[0], self.dom[1]))
        self.wavelengths = np.linspace(300,800,101)
        self.multipoles = np.zeros((int(len(self.l)*len(self.f)), nProfiles, len(self.wavelengths)))
        self.scatteredPower = np.zeros((nProfiles, 101))

        self.nProfiles = nProfiles
        self.nT = nT; self.nR = nR; self.nC = nC; self.nF = nF; self.nP = nP

        n = 0
        for k in range(nT):
            self.profGen.generate('tri')
            self.images[n, :, :] = self.profGen.smoothedImage
            self.chromosomes[n] = self.profGen.binaryPolygon
            n += 1
        for k in range(nR):
            self.profGen.generate('rec')
            self.images[n, :, :] = self.profGen.smoothedImage
            self.chromosomes[n] = self.profGen.binaryPolygon
            n += 1
        for k in range(nC):
            self.profGen.generate('cir')
            self.images[n, :, :] = self.profGen.smoothedImage
            self.chromosomes[n] = self.profGen.binaryPolygon
            n += 1
        for k in range(nF):
            if termsF == None:
                self.profGen.fourierGenerator(np.random.randint(2,4))
                self.profGen.arrayConversion()
                self.profGen.encodePolygon()
                self.images[n, :, :] = self.profGen.smoothedImage
                self.chromosomes[n] = self.profGen.binaryPolygon
            else:
                self.profGen.fourierGenerator(termsF)
                self.profGen.arrayConversion()
                self.profGen.encodePolygon()
                self.images[n, :, :] = self.profGen.smoothedImage
                self.chromosomes[n] = self.profGen.binaryPolygon

            n += 1
        for k in range(nP):
            self.profGen.generate('pol')
            self.images[n, :, :] = self.profGen.smoothedImage
            self.chromosomes[n] = self.profGen.binaryPolygon
            n += 1

            
        
        self.dlModels.getModelPrediction(self.images)
        self.multipoles = self.dlModels.multipolePredictions
        self.scatteredPower = self.dlModels.scatteredPowerPredictions
        self.getFitness()





    def getResiduals(self):
        x = self.objScatteredPower
        y = self.scatteredPower
        x = x / np.max(x)
        y = y / np.max(y)
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
                if np.isnan(distance[w]):
                    distance[w] = 1000
                if np.isnan(absoluteError[w]):
                    absoluteError[w] = 1000
                if np.isnan(relativeError[w]):
                    relativeError[w] = 1000
                if np.isnan(meanSquaredError[w]):
                    meanSquaredError[w] = 1000
                if np.isnan(rootMeanSquaredError[w]):
                    rootMeanSquaredError[w] = 1000

            self.integral[n] = np.round(np.sum(distance), 5)
            self.meanAbsoluteError[n] = np.round(np.mean(absoluteError), 5)
            self.meanRelativeError[n] = np.round(np.mean(relativeError), 5)
            self.meanSquaredError[n] = np.round(np.mean(meanSquaredError), 5)
            self.rootMeanSquaredError[n] = np.round(np.sqrt(np.mean(rootMeanSquaredError)), 5)

    
    def getFitness(self):
        self.getResiduals()

        if self.fitnessType == 'gap':
            self.fitness = scale(0.005,self.integral)
        elif self.fitnessType == 'mse':
            self.fitness = scale(75,self.meanSquaredError)
        elif self.fitnessType == 'rmse':
            self.fitness = scale(2,self.rootMeanSquaredError)
        elif self.fitnessType == 'mae':
            self.fitness = scale(2.5,self.meanAbsoluteError)
        elif self.fitnessType == 'mre':
            self.fitness = scale(3.5,self.meanRelativeError)
        else:
            self.fitness = scale(0.005,self.integral)


        




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

        self.selectParentPairs()
        self.selectRetainedPopulation()

        for k in range(self.nGenerated):
            p0 = self.chromosomes[self.parentPairs[k, 0]]
            p1 = self.chromosomes[self.parentPairs[k, 1]]
            g = go.GeneticOperations(p0, p1, mR=self.mR, cP=self.cP)
            g.operate()
            self.profGen.decodeChromosome(g.c0)
            i0 = self.profGen.smoothedImage
            self.profGen.decodeChromosome(g.c1)
            i1 = self.profGen.smoothedImage

            newChromosomes[n] = g.c0
            newImages[n, :, :] = i0
            n += 1
            newChromosomes[n] = g.c1
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

    def displayParameters(self, outDir=None):
        if outDir != None:
            with open(f"{outDir}/InverseDesignParameters.txt", "w") as f:
                f.write("=============================================================================\n")
                f.write("Inverse Design Details\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f"Initial Generation Makeup\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f"{self.nT} Triangles\n")
                f.write(f"{self.nR} Rectangles\n")
                f.write(f"{self.nC} Circles\n")
                f.write(f"{self.nF} Fourier Series Polygons\n")
                f.write(f"{self.nF} Randomized Polygons\n")
                f.write(f"{self.nProfiles} Total Profiles\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write("Profile Generation Parameters\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f"Profiles are generated with {self.nVertices} vertices\n")
                f.write(f"Profile vertices are allowed within a polar radius range of ({self.rMin}, {self.rMax})\n")
                f.write(f"Images rendered from profiles have a resolution of {self.dom[0]} x {self.dom[1]}\n")
                f.write(f"Images are smoothed by a gaussian filter with a standard deviation of {self.s}\n")
                f.write(f"Profiles are encoded to {self.p}-bit binary chromosomes\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write("Genetic Operation Parameters\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f"Parent chromosomes are split into {self.cP} crossover points\n")
                f.write(f"New chromosomes have {int(len(self.chromosomes[0])*self.mR)} mutations\n")
                f.write(f"Fitness is calculated from {self.fitnessType}\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write("Neural Network Parameters\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f"Trained neural networks are loaded from:\n  {self.path}\n")
                f.write("Predicted scattering behaviour defined by:\n")
                for j in range(len(self.f)):
                    for i in range(len(self.l)):
                            if self.l[i] == 1:
                                if self.f[j] == "E":
                                    f.write(f"    first order electric dipole\n")
                                elif self.f[j] == "H":
                                    f.write(f"    first order magnetic dipole\n")
                            elif self.l[i] == 2 and self.m[i] == 1:
                                if self.f[j] == "E":
                                    f.write(f"    first order electric quadrupole\n")
                                elif self.f[j] == "H":
                                    f.write(f"    first order magnetic quadrupole\n")
                            elif self.l[i] == 2 and self.m[i] == 2:
                                if self.f[j] == "E":
                                    f.write(f"    second order electric quadrupole\n")
                                elif self.f[j] == "H":
                                    f.write(f"    second order magnetic quadrupole\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write("Inverse Design Input Parameters\n")
                f.write("--------------------------------------------------------------------------\n")
                f.write(f'   modelDirectory: {self.path}\n')
                f.write(f'   nVertices:      {self.nVertices}\n')
                f.write(f'   rMin:           {self.rMin}\n')
                f.write(f'   rMax:           {self.rMax}\n')
                f.write(f'   nT:             {self.nT}\n')
                f.write(f'   nR:             {self.nR}\n')
                f.write(f'   nC:             {self.nC}\n')
                f.write(f'   nP:             {self.nP}\n')
                f.write(f'   nF:             {self.nF}\n')
                f.write(f'   cP:             {self.cP}\n')
                f.write(f'   mR:             {self.mR}\n')
                f.write(f'   d:              {self.dom}\n')
                f.write(f'   s:              {self.s}\n')
                f.write(f'   p:              {self.p}\n')
                f.write(f'   l:              {self.l}\n')
                f.write(f'   m:              {self.m}\n')
                f.write(f'   f:              {self.f}\n')
                f.write("=============================================================================")

                

        print("=============================================================================")
        print("Inverse Design Parameters")
        self.__displayTotals()
        print("--------------------------------------------------------------------------")
        print("Profile Generation Parameters")
        print("--------------------------------------------------------------------------")
        print(f"Profiles are generated with {self.nVertices} vertices")
        print(f"Profile vertices are allowed within a polar radius range of ({self.rMin}, {self.rMax})")
        print(f"Images rendered from profiles have a resolution of {self.dom[0]} x {self.dom[1]}")
        print(f"Images are smoothed by a gaussian filter with a standard deviation of {self.s}")
        print(f"Profiles are encoded to {self.p}-bit binary chromosomes")
        print("--------------------------------------------------------------------------")
        print("Genetic Operation Parameters")
        print("--------------------------------------------------------------------------")
        print(f"Parent chromosomes are split into {self.cP} crossover points")
        print(f"New chromosomes have {int(len(self.chromosomes[0])*self.mR)} mutations")
        print("--------------------------------------------------------------------------")
        print("Neural Network Parameters")
        print("--------------------------------------------------------------------------")
        print(f"Trained neural networks are loaded from:\n  {self.path}")
        print("Predicted scattering behaviour defined by:")
        for j in range(len(self.f)):
            for i in range(len(self.l)):
                    if self.l[i] == 1:
                        if self.f[j] == "E":
                            print(f"    first order electric dipole")
                        elif self.f[j] == "H":
                            print(f"    first order magnetic dipole")
                    elif self.l[i] == 2 and self.m[i] == 1:
                        if self.f[j] == "E":
                            print(f"    first order electric quadrupole")
                        elif self.f[j] == "H":
                            print(f"    first order magnetic quadrupole")
                    elif self.l[i] == 2 and self.m[i] == 2:
                        if self.f[j] == "E":
                            print(f"    second order electric quadrupole")
                        elif self.f[j] == "H":
                            print(f"    second order magnetic quadrupole")
        print("=============================================================================")
        print('\n')


    def writeLoss(self, iGen, growthRate=0, outDir=None):
        if outDir != None:
            if iGen == 0:
                with open(f"{outDir}/Fitness.txt", "w") as f:
                    f.write("=============================================\n")
                    f.write("Population Fitness\n")
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {0} - Birth Rate: n/a \n")
                    f.write("------------------------------------\n")
                    fit = self.fitness
                    fit = np.sort(fit)
                    fit = np.flip(fit)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(fit[k], 6)}\n')
            else:
                with open(f"{outDir}/Fitness.txt", "a") as f:
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {iGen} - Birth Rate: {np.round(2*growthRate, 2)}\n")
                    f.write("------------------------------------\n")
                    fit = self.fitness
                    fit = np.sort(fit)
                    fit = np.flip(fit)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(fit[k], 6)}\n')

            if iGen == 0:
                with open(f"{outDir}/MeanSquaredError.txt", "w") as f:
                    f.write("=============================================\n")
                    f.write("Population Mean Squared Error\n")
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {0} - Birth Rate: n/a \n")
                    f.write("------------------------------------\n")
                    mse = self.meanSquaredError
                    mse = np.sort(mse)                    
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mse[k], 6)}\n')
            else:
                with open(f"{outDir}/MeanSquaredError.txt", "a") as f:
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {iGen} - Birth Rate: {np.round(2*growthRate, 2)}\n")
                    f.write("------------------------------------\n")
                    mse = self.meanSquaredError
                    mse = np.sort(mse)                    
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mse[k], 6)}\n')

            if iGen == 0:
                with open(f"{outDir}/RootMeanSquaredError.txt", "w") as f:
                    f.write("=============================================\n")
                    f.write("Population Root Mean Squared Error\n")
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {0} - Birth Rate: n/a \n")
                    f.write("------------------------------------\n")
                    rmse = self.rootMeanSquaredError
                    rmse = np.sort(rmse)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(rmse[k], 6)}\n')
            else:
                with open(f"{outDir}/RootMeanSquaredError.txt", "a") as f:
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {iGen} - Birth Rate: {np.round(2*growthRate, 2)}\n")
                    f.write("------------------------------------\n")
                    rmse = self.rootMeanSquaredError
                    rmse = np.sort(rmse)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(rmse[k], 6)}\n')

            if iGen == 0:
                with open(f"{outDir}/MeanAbsoluteError.txt", "w") as f:
                    f.write("=============================================\n")
                    f.write("Population Mean Absolute Error\n")
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {0} - Birth Rate: n/a \n")
                    f.write("------------------------------------\n")
                    mae = self.meanAbsoluteError
                    mae = np.sort(mae)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mae[k], 6)}\n')
            else:
                with open(f"{outDir}/MeanAbsoluteError.txt", "a") as f:
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {iGen} - Birth Rate: {np.round(2*growthRate, 2)}\n")
                    f.write("------------------------------------\n")
                    mae = self.meanAbsoluteError
                    mae = np.sort(mae)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mae[k], 6)}\n')

            if iGen == 0:
                with open(f"{outDir}/MeanRelativeError.txt", "w") as f:
                    f.write("=============================================\n")
                    f.write("Population Mean Relative Error\n")
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {0} - Birth Rate: n/a \n")
                    f.write("------------------------------------\n")
                    mre = self.meanRelativeError
                    mre = np.sort(mre)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mre[k], 6)}\n')
            else:
                with open(f"{outDir}/MeanRelativeError.txt", "a") as f:
                    f.write("------------------------------------\n")
                    f.write(f"Generation: {iGen} - Birth Rate: {np.round(2*growthRate, 2)}\n")
                    f.write("------------------------------------\n")
                    mre = self.meanRelativeError
                    mre = np.sort(mre)
                    for k in range(len(self.fitness)):
                        f.write(f'{np.round(mre[k], 6)}\n')
        else:
            print("Need to define output directory!")

    def sortPopulation(self):
        sortIndices = np.argsort(self.fitness)
        sortIndices = np.flip(sortIndices)

        self.fitness = self.fitness[sortIndices]
        self.images = self.images[sortIndices, :, :]
        self.multipoles = self.multipoles[:, sortIndices, :]
        self.scatteredPower  = self.scatteredPower[sortIndices, :]

        #self.chromosomes isn't a numpy array so it should be sorted manually
        ch = np.zeros(self.nProfiles)
        ch= list(ch)
        for k in range(len(self.chromosomes)):
            ch[k] = self.chromosomes[sortIndices[k]]
        self.chromosomes = ch

        self.integral = self.integral[sortIndices]
        self.meanAbsoluteError = self.meanAbsoluteError[sortIndices]
        self.meanRelativeError = self.meanRelativeError[sortIndices]
        self.meanSquaredError = self.meanSquaredError[sortIndices]
        self.rootMeanSquaredError = self.rootMeanSquaredError[sortIndices]



    def plotSpectrum(self, ax, idx):
        y0 = self.scatteredPower[idx, :] / np.max(self.scatteredPower[idx, :])
        y1 = self.objScatteredPower / np.max(self.objScatteredPower)
        ax.set_ylim((0.5, 1.1))
        ax.plot(self.wavelengths, y0)
        ax.plot(self.wavelengths, y1, c='black')
        ax.fill_between(self.wavelengths, y0, y1,
         color='red', alpha=0.3, label=f'Fitness: {np.round(self.fitness[idx], 3)}')
        ax.legend()


    def plotSelectPerformers(self, outDir=None, fName=None):
        """
        create a plot of the top and bottom 6 performers
        """
        self.sortPopulation()

        #top 6
        f, axs = plt.subplots(3, 4, gridspec_kw={'width_ratios': [2, 4, 4, 2]}, figsize=(12, 6))
        for i in [0, 3]:
            for j in [0, 1, 2]:
                axs[j,i].set_yticks([])
                axs[j,i].set_xticks([])

        for j in range(np.shape(axs)[0]-1):
            for i in range(np.shape(axs)[1]):
                axs[j,i].set_xticks([])

        axs[0,0].imshow(self.images[0, :, :], cmap=plt.cm.binary)
        axs[0,3].imshow(self.images[1, :, :], cmap=plt.cm.binary)
        axs[1,0].imshow(self.images[2, :, :], cmap=plt.cm.binary)
        axs[1,3].imshow(self.images[3, :, :], cmap=plt.cm.binary)
        axs[2,0].imshow(self.images[4, :, :], cmap=plt.cm.binary)
        axs[2,3].imshow(self.images[5, :, :], cmap=plt.cm.binary)

        self.plotSpectrum(axs[0,1], 0)
        self.plotSpectrum(axs[0,2], 1)
        self.plotSpectrum(axs[1,1], 2)
        self.plotSpectrum(axs[1,2], 3)
        self.plotSpectrum(axs[2,1], 4)
        self.plotSpectrum(axs[2,2], 5)

        axs[2,1].set_xlabel("Wavelength (nm)")
        axs[2,2].set_xlabel("Wavelength (nm)")
        
        f.suptitle(f"Generation {fName} - Top 6 Performers")
        plt.tight_layout()
        plt.savefig(f"{outDir}/gen{fName}TopPerformers.png")
        plt.close()

        #bottom 6
        f, axs = plt.subplots(3, 4, gridspec_kw={'width_ratios': [2, 4, 4, 2]}, figsize=(12, 6))
        for i in [0, 3]:
            for j in [0, 1, 2]:
                axs[j,i].set_yticks([])
                axs[j,i].set_xticks([])

        for j in range(np.shape(axs)[0]-1):
            for i in range(np.shape(axs)[1]):
                axs[j,i].set_xticks([])

        axs[0,0].imshow(self.images[self.nProfiles-6, :, :], cmap=plt.cm.binary)
        axs[0,3].imshow(self.images[self.nProfiles-5, :, :], cmap=plt.cm.binary)
        axs[1,0].imshow(self.images[self.nProfiles-4, :, :], cmap=plt.cm.binary)
        axs[1,3].imshow(self.images[self.nProfiles-3, :, :], cmap=plt.cm.binary)
        axs[2,0].imshow(self.images[self.nProfiles-2, :, :], cmap=plt.cm.binary)
        axs[2,3].imshow(self.images[self.nProfiles-1, :, :], cmap=plt.cm.binary)

        self.plotSpectrum(axs[0,1], self.nProfiles-6)
        self.plotSpectrum(axs[0,2], self.nProfiles-5)
        self.plotSpectrum(axs[1,1], self.nProfiles-4)
        self.plotSpectrum(axs[1,2], self.nProfiles-3)
        self.plotSpectrum(axs[2,1], self.nProfiles-2)
        self.plotSpectrum(axs[2,2], self.nProfiles-1)

        axs[2,1].set_xlabel("Wavelength (nm)")
        axs[2,2].set_xlabel("Wavelength (nm)")
        
        f.suptitle(f"Generation {fName} - Bottom 6 Performers")
        plt.tight_layout()
        plt.savefig(f"{outDir}/gen{fName}BottomPerformers.png")
        plt.close()
