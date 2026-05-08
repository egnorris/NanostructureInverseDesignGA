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
  "sseWeight": 1,
  "nT": 10,
  "nC": 10,
  "nR": 10,
  "nP": 10,
  "nN": 10,
  "nV": 24,
  "modelDir":"/media/work/evan/deep_learning_data/trained_models",
  "targetFile": "input/objScatteredPower0.txt"

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
    "sseWeight": ["sseWeight", "weightSse", "mseWeight", "weightMse", "fMse", "mseF", "fSse", "sseF"],
    "nT": ["nT", "nTriangles", "numTriangles"],
    "nR": ["nR", "nRectangles", "numRectangles"],
    "nC": ["nC", "nCircles", "numCircles"],
    "nP": ["nP", "nPolygons", "numPolygons"],
    "nN": ["nN", "nNgons", "numNgons"],
    "nV": ["nV", 'nVertices'],
    "modelDir": ["modelDir"],
    "targetFile":["targetFile"]

    }

def getkwarg(kwargs, default, keywords):
  for i in range(len(keywords)):
    if f"{keywords[i]}" in kwargs:
      #print(f"{keywords[0]}: {kwargs[keywords[i]]}")
      kwargs[keywords[i]]
      if kwargs[keywords[i]] != None:
        return kwargs[keywords[i]]
      else:
        return default
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