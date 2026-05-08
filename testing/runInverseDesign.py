import sys
import argparse
import os

sys.path.append("../modules")
import InverseDesign
import Support

"""
streamline parsing keyword arguments by setting the 
default keyword arguments and the supported keywords
from ../modules/Support.py
"""
global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords

def kwargsHelp(key=None):
    if key == None:
        for key in keywords:
            print(f"The Default value of {key} is {defaultKwargs[key]}:")
            print(f"{key} has the supported alternate keywords:")
            for j in range(len(keywords[key])):
                print(f"    {keywords[key][j]}")
    else:
        print(f"The Default value of {key} is {defaultKwargs[key]}:")
        print(f"{key} has the supported alternate keywords:")
        for j in range(len(keywords[key])):
            print(f"    {keywords[key][j]}")
        
            


def setupPopulation(**kwargs):
    nV = Support.getkwarg(kwargs, defaultKwargs["nV"], keywords["nV"])
    modelDir = Support.getkwarg(kwargs, defaultKwargs["modelDir"], keywords["modelDir"])
    spectrumFileName=Support.getkwarg(kwargs, defaultKwargs["targetFile"], keywords["targetFile"])
    pop = InverseDesign.Population(nVertices=nV,modelDirectory=modelDir,**kwargs)
    pop.readObjective(spectrumFileName=spectrumFileName)
    pop.initialize(
        nT=Support.getkwarg(kwargs, defaultKwargs["nT"], keywords["nT"]),
        nC=Support.getkwarg(kwargs, defaultKwargs["nC"], keywords["nC"]),
        nR=Support.getkwarg(kwargs, defaultKwargs["nR"], keywords["nR"]),
        nP=Support.getkwarg(kwargs, defaultKwargs["nP"], keywords["nP"]),
        nN=Support.getkwarg(kwargs, defaultKwargs["nN"], keywords["nN"])
    )
    return pop

def updatePopulation(pop, birthRate):
    pop.update(birthRate=birthRate)
    return pop
    


if __name__ == '__main__':
    pop = setupPopulation()
    pop = updatePopulation(pop, birthRate=1.0)
    pop = updatePopulation(pop, birthRate=1.0)
