import numpy as np

defaultKwargs = {"rMin": 20, "rMax": 75, "d": (180,180), "s": 5, "p": 12}

keywords = {
    "rMin":["rMin", "minR", "r0", "minimumR", "rMinimum", "minimumRadius"],
    "rMax":["rMax", "maxR", "r1", "maximumR", "rMaximum", "maximumRadius"],
    "d":["domain", "d", "dom"],
    "s":["smoothness", "s", "sigma"],
    "p":["precision", "p", "binaryPrecision"]
    }

def getkwarg(kwargs, default, keywords):
  for i in range(len(keywords)):
    if f"{keywords[i]}" in kwargs:
      print(f"{keywords[0]}: {kwargs[keywords[i]]}")
      return kwargs[keywords[i]]
  #print(f"Default {keywords[0]}: {default}")
  return default

def closestRadius(r, r0=40, r1=80, p=8):
  radiusRange = np.linspace(r0, r1, 2**p)
  idx = np.argmin(np.abs(radiusRange - r))
  rMatch = radiusRange[idx]
  #print(f"Current Radial Value: {r}")
  #print(f"New Radial Value: {rMatch}")
  return rMatch