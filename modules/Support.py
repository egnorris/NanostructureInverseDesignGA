import numpy as np

defaultKwargs = {
  "rMin": 20,
  "rMax": 75,
  "d": (180,180),
  "s": 5,
  "p": 12,
  "mR":0.1,
  "cP":1,
  "l":[1,2,2],
  "m":[1,1,2],
  "f":["E", "H"],
  "minWavelength": 300,
  "maxWavelength": 800,
  "saeWeight": 1,
  "sseWeight": 1
  }

keywords = {
    "rMin":["rMin", "minR", "r0", "minimumR", "rMinimum", "minimumRadius"],
    "rMax":["rMax", "maxR", "r1", "maximumR", "rMaximum", "maximumRadius"],
    "d":["domain", "d", "dom"],
    "s":["smoothness", "s", "sigma"],
    "p":["precision", "p", "binaryPrecision"],
    "mR": ["mR", "mutationRate", "mutR", "mutRate"],
    "cP": ["cP", "crossoverPoints", "crossPoints"],
    "l": ["l", "degree"],
    "m": ["m", "order"],
    "f": ["f", "fields"],
    "minWavelength": ["wMin", "minW", "w0", "minimumW", "wMinimum", "minimumWavelength"],
    "maxWavelength": ["wMax", "maxW", "w1", "maximimW", "wMaximum", "maximumWavelength"],
    "saeWeight": ["saeWeight", "weightSae", "maeWeight", "weightMae", "fMae", "maeF", "fSae", "saeF"],
    "sseWeight": ["sseWeight", "weightSse", "mseWeight", "weightMse", "fMse", "mseF", "fSse", "sseF"]
    }

def getkwarg(kwargs, default, keywords):
  for i in range(len(keywords)):
    if f"{keywords[i]}" in kwargs:
      #print(f"{keywords[0]}: {kwargs[keywords[i]]}")
      return kwargs[keywords[i]]
  #print(f"Default {keywords[0]}: {default}")
  return default

def getCoding(rMin, rMax, p):
  return np.linspace(rMin, rMax, 2**p)

def closestRadius(r, rMin, rMax, p):
  radiusRange = getCoding(rMin, rMax, p)
  idx = np.argmin(np.abs(radiusRange - r))
  rMatch = radiusRange[idx]
  return rMatch

def binaryArray2Str(binaryArray):
  bStr = ''
  for i in range(len(binaryArray)):
    bStr = bStr + f'{binaryArray[i]}'
  return bStr