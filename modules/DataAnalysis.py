import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt

def stateList2Array(stateList):
    """
    stateList2Array()
        Convert a list of binary strings representing current microstates into a numpy
            array of binary integers
        Input:
            stateList: list of strings representing each of the current microstates as a
                binary string i.e. '000'
        Output:
            stateArray: numpy array of binary integers representing all current microstates 
                formatted as (nStates, nBits)
    """
    temp = [list(s) for s in stateList]
    return np.array(temp, dtype=int)

def getBitProbabilities(stateArray):
    """
    getBitProbabilities()
        Calculate the probability of having each a value of 1 for 
            each bit in a microstate
        Input:
            stateArray: numpy array of binary integers representing all current microstates 
                formatted as (nStates, nBits)

        Output:
            bitProbabilities: an array of the probabilites of each bit having a value of 1
    """
    nStates = np.shape(stateArray)[0]
    return np.sum(stateArray, axis=0)/nStates

def getShannonEntropy(stateList):
    """
    getShannonEntropy()
        calculate the shannon entropy from the current set of microstates:

        Input:
            stateList: list of strings representing each of the current microstates as a
                binary string i.e. '000'
        Output:
            H: shannon entropy
            normH: normalized shannon entropy, normalized by the maximum shannon entropy of nBits/2  
    """
    stateArray = stateList2Array(stateList)
    nBits = np.shape(stateArray)[1]
    bitProbabilities = getBitProbabilities(stateArray)
    H = -np.nansum(np.dot(bitProbabilities,np.log(bitProbabilities)))
    normH = 2*H/nBits
    return np.round(H, 3), np.round(normH, 3)


