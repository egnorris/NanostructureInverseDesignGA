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



def plotSpectrum(ax, idx, population):
    k0= population.minWavelengthIdx
    k1= population.maxWavelengthIdx
    y0 = population.scatteredPower[idx, :]
    y1 = population.objScatteredPower
    r = population.residual[idx, :]
    wav = population.wavelengths
    wavTrunc = wav[k0:k1]
    y0Trunc = y0[k0:k1]
    y1Trunc = y1[k0:k1]
    ax.set_xlim((300,800))
    ax.set_ylim((0, np.min([1.1, 1.5*np.max(y1)])))

    ax.plot(wav, y0, c='black', lw=1, zorder = 1)
    ax.plot(wav, y1, c='tab:blue', lw=1, zorder = 1)
    ax.scatter(wav, y1, c='tab:blue', s=10, zorder = 2)

    ax.fill_between(wavTrunc, y0Trunc, y0Trunc + np.abs(r)*np.sign(r),
        color='cornflowerblue', alpha=0.3, label=f'sae: {np.round(population.sumAbsoluteError[idx], 3)}', zorder = 0)
    ax.fill_between(wavTrunc, y0Trunc, y0Trunc + (r**2)*np.sign(r),
        color='red', alpha=0.3, label=f'sse: {np.round(population.sumSquaredError[idx], 3)}', zorder = 0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),ncols=2)


def plot6(fname, outDir, population, idxList=[0,1,2,3,4,5]):
    fig, axs = plt.subplots(3, 4, gridspec_kw={'width_ratios': [2, 4, 4, 2]}, figsize=(12, 6))
    for i in [0, 3]:
        for j in [0, 1, 2]:
            axs[j,i].set_yticks([])
            axs[j,i].set_xticks([])

    for j in range(np.shape(axs)[0]-1):
        for i in range(np.shape(axs)[1]):
            axs[j,i].set_xticks([])

    axs[0,0].imshow(population.images[idxList[0], :, :], cmap=plt.cm.binary)
    axs[0,3].imshow(population.images[idxList[1], :, :], cmap=plt.cm.binary)
    axs[1,0].imshow(population.images[idxList[2], :, :], cmap=plt.cm.binary)
    axs[1,3].imshow(population.images[idxList[3], :, :], cmap=plt.cm.binary)
    axs[2,0].imshow(population.images[idxList[4], :, :], cmap=plt.cm.binary)
    axs[2,3].imshow(population.images[idxList[5], :, :], cmap=plt.cm.binary)

    plotSpectrum(axs[0,1], idxList[0], population)
    plotSpectrum(axs[0,2], idxList[1], population)
    plotSpectrum(axs[1,1], idxList[2], population)
    plotSpectrum(axs[1,2], idxList[3], population)
    plotSpectrum(axs[2,1], idxList[4], population)
    plotSpectrum(axs[2,2], idxList[5], population)

    axs[2,1].set_xlabel("Wavelength (nm)")
    axs[2,2].set_xlabel("Wavelength (nm)")
    fig.suptitle(fname)
    plt.tight_layout()
    plt.savefig(f'{outDir}/{fname.replace(" ", "-")}.png',dpi=300)



def packageDictionary(population):
    #limit file size by only including data that can't be easily reconstructed
    temp = {
        "Fitness": population.fitness,
        "Residual": population.residual,
        "tss": population.tss,
        "Binary Chromosomes": population.chromosomes,
        "Polar Coordinates": population.polar,
        "scattered Power": population.scatteredPower,
        "objective": population.objScatteredPower,
        "serialNumber": population.serialNumber,
        }
    return temp

def unpackMAT():
    """
    """


