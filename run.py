import argparse
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from scipy.io import savemat
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import InverseDesign


def generateNewTarget(kwargs):
    pop = InverseDesign.Population(nVertices=kwargs['nV'], fitnessType='rmse',
        modelDirectory=kwargs['modelDirectory'],
        rMin=kwargs['rMin'], rMax=kwargs['rMax'], d=(180,180), s=5, p=kwargs['precision'],
        mR = 0.1, cP = 1,
        l=kwargs['l'], m=kwargs['m'], f=kwargs['f'])
    pop.defineObjective(profileType0 = kwargs['targetGeneration'], profileType1 = 'tri', termsF=3)

    plt.imshow(pop.objImage, cmap=plt.cm.binary)
    plt.savefig("targetProfile.png")
    plt.close()
    plt.plot(np.linspace(300,800,101), pop.objScatteredPower / np.max(pop.objScatteredPower), c='black')
    plt.xlim(300, 800)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Scattered Power")
    plt.title("Objective Scattered Power Spectrum")
    plt.savefig("targetScatteredPower.png")
    plt.close()
    np.savetxt("targetScatteredPower.txt",pop.objScatteredPower)

    return pop.objScatteredPower

def loadExistingTarget(kwargs):
    y = np.loadtxt(kwargs['targetPath'])
    return y 

def saveTopPerformers(kwargs, pop):
    pop.sortPopulation()
    for k in range(kwargs['numSave']):
        temp = {
            "Profile":  pop.images[k, :, :],
            "Fitness": pop.fitness[k],
            "Scattered Power": pop.scatteredPower[k, :]}

        topPerformersDict[f'G{0}-{k}'] = temp


parser = argparse.ArgumentParser(description="Run Genetic Algorithm Inverse Design")
#Required Parameters
parser.add_argument('-out', '--outputDir', type = str, required=True,
    help="Set directory to save output data to.")

parser.add_argument('-tp', '--targetPath', type=str, default=None,
    help="Enter the path of the target scattered power spectrum; supercedes --targetGeneration argument if valid")
parser.add_argument('-tg', '--targetGeneration', type=str, default=None,
    help="If a new target scattered power spectrum is desired define what type of shape to generate it from; \nAllowed Arguments: 'tri', 'cir', 'rec', 'pol'")

#Parameters with a default value
parser.add_argument('-ft', '--fitnessType', type = str, required=False, default='rmse',
    help="Set what type of error to use when calculating fitness; \nAllowed Arguments: 'rmse', 'mse', 'mre', 'mae', 'gap'; \nDefault: 'rmse'")
parser.add_argument('-ng', '--numGenerations', type = int, required=False, default=25,
    help="Define the number of generations to run the algorithm for; \nDefault: 25")
parser.add_argument('-ns', '--numSave', type=int, required=False, default = 6,
    help="Define the number of top performers to be saved from each generation to a .MAT binary; \n Default: 6")


parser.add_argument('-mD', '--modelDirectory', type=str, required=False, default = "/media/work/evan/deep_learning_data/trained_models",
    help="")

parser.add_argument('-nV', type=int, required=False, default = 24,
    help="")

parser.add_argument('-nT', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nR', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nC', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nP', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nF', type=int, required=False, default = 30,
    help="")

parser.add_argument('-r0', '--rMin', type=int, required=False, default = 10,
    help="")

parser.add_argument('-r1', '--rMax', type=int, required=False, default = 75,
    help="")

parser.add_argument('--smoothness', type=int, required=False, default = 5,
    help="")

parser.add_argument('-p','--precision', type=int, required=False, default = 12,
    help="")

parser.add_argument('-mR', '--mutationRate', type=float, required=False, default = 0.1,
    help="")

parser.add_argument('-cP', '--numCrossoverPoints', type=int, required=False, default = 1,
    help="")

parser.add_argument('-l' , required=False, default = [1,2,2],
    help="")

parser.add_argument('-m' , required=False, default = [1,1,2],
    help="")

parser.add_argument('-f' , required=False, default = ['E','H'],
    help="")

#Optional Parameters
parser.add_argument('-seed', '--seed', type = int, required=False, default=None,
    help="Define a seed for the random number generator if desired.")


kwargs = parser.parse_args().__dict__

generateNewTargetFlag = False

#check that the input arguments are valid

if (kwargs['targetPath'] == None) and (kwargs['targetGeneration'] == None):
    raise Exception('--targetPath/-tp or --targetGeneration/-tg must be set')

if kwargs['targetPath'] != None:
    if os.path.isfile(kwargs['targetPath']):
        ''
        temp = np.loadtxt(kwargs['targetPath'])
        if np.shape(temp)[0] != 101 and kwargs['targetGeneration'] == None:
            raise Exception(f"'{kwargs['targetPath']}', isn't properly formatted; should have length of 101 not {np.shape(temp)[0]};")
        elif kwargs['targetGeneration'] != None and np.shape(temp)[0] != 101:
            print(f"'{kwargs['targetPath']}', isn't properly formatted; should have length of 101 not {np.shape(temp)[0]}; a new target will be generated")
            generateNewTargetFlag = True
        elif np.shape(temp)[0] == 101 :
            print(f"'{kwargs['targetPath']}' is valid, a new target will not be generated")
            
    elif kwargs['targetGeneration'] != None:
        print(f"Entered --targetPath/-tp, '{kwargs['targetPath']}', is invalid, a new target will be generated")
        generateNewTargetFlag = True
    else:
        raise Exception(f"Entered --targetPath/-tp, '{kwargs['targetPath']}', is invalid;")




print(kwargs)


if kwargs['targetGeneration'] != None:
    if kwargs['targetGeneration'] in ['tri', 'cir', 'rec', 'pol']:
        generateNewTargetFlag = True
        ''
    else:
        raise Exception(f"{kwargs['targetGeneration']} is not valid for --targetGeneration/-tg \nAllowed Arguments: 'tri', 'cir', 'rec', 'pol'")

if kwargs['fitnessType'] in ['rmse', 'mse', 'mre', 'mae', 'gap']:
    ''
else:
    raise Exception(f"{kwargs['fitnessType']} is not valid for --fitnessType/-ft \nAllowed Arguments: 'rmse', 'mse', 'mre', 'mae', 'gap'; \nDefault: 'rmse'")



if generateNewTargetFlag:
    targetSpectrum = generateNewTarget(kwargs)
else:
    targetSpectrum = loadExistingTarget(kwargs)

if kwargs['seed'] != None:
    print(f"Setting RNG seed: {kwargs['seed']}")
    np.random.seed(kwargs['seed'])

#Make new output directory if it doesn't already exist
outDir = kwargs['outputDir']
try:
    os.mkdir(f"{outDir}")
except FileExistsError:
    print(f"Directory: {outDir} Exists")

global topPerformersDict
topPerformersDict = {}


#Initialize Population
pop = InverseDesign.Population(
    nVertices=kwargs['nV'],
    fitnessType=kwargs['fitnessType'],
    modelDirectory=kwargs['modelDirectory'],
    lambdaMin=325,
    lambdaMax=700,
    rMin=kwargs['rMin'],
    rMax=kwargs['rMax'],
    d=(180,180),
    s=kwargs['smoothness'],
    p=kwargs['precision'],
    mR=kwargs['mutationRate'],
    cP=kwargs['numCrossoverPoints'],
    l=kwargs['l'], 
    m=kwargs['m'],
    f=kwargs['f'])
#Set the population objective
pop.defineObjective(spectrum=targetSpectrum)

#0th Generation 
pop.initialize(
    nT=kwargs['nT'],
    nR=kwargs['nR'],
    nC=kwargs['nC'],
    nP=kwargs['nP'],
    nF=kwargs['nF'])

saveTopPerformers(kwargs, pop)
pop.plotSelectPerformers(outDir, f"0")
print(pop.nProfiles)


nGenerations = kwargs['numGenerations']
for c in range(nGenerations):
    print(f"Generation {c+1}")
    birthRate = 0.45
    pop.newGeneration(birthRate)
    pop.writeLoss(iGen=c+1, growthRate=birthRate, outDir=outDir)
    pop.plotSelectPerformers(outDir, f"{c+1}")
    saveTopPerformers(kwargs, pop)
    
    
    


savemat(f"{outDir}/TopPerformers-{kwargs['fitnessType']}.mat", topPerformersDict)