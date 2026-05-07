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
            normH: normalized shannon entropy, normalized by the maximum shannon entropy of N/2  
    """
    stateArray = stateList2Array(stateList)
    bitProbabilities = getBitProbabilities(stateArray)
    #shannon entropy is calculated with a base 2 logarithm and np.log() is the natural logarithm
    log2p = (np.log(bitProbabilities)/np.log(2))
    H = -np.nansum(bitProbabilities*log2p)
    normH = 2*H/len(stateList)
    return np.round(H, 3), np.round(normH, 3)