import Support
import numpy as np
global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords
class GeneticOperations():
  def __init__(self, p0, p1, **kwargs):
    self.p0 = p0
    self.p1 = p1
    self.n = len(p0)
    self.mR = Support.getkwarg(kwargs, defaultKwargs["mR"], keywords["mR"])
    self.cP = Support.getkwarg(kwargs, defaultKwargs["cP"], keywords["cP"])
    """
      check that p0, p1 are formatted correctly, they should binary strings
        of the same length
    """
    if type(self.p0) != type(''):
      raise TypeError(f"Input p0 has invalid type, {type(self.p0)}")
    elif type(self.p1) != type(''):
      raise TypeError(f"Input p1 has invalid type, {type(self.p1)}")
    if p0.isdigit() == False:
      raise TypeError(f"p0 must be a numeric string")
    elif p1.isdigit() == False:
      raise TypeError(f"p1 must be a numeric string")
    if len(p0) != len(p1):
      raise TypeError(f"p0 and p1 must be the same length, {len(p0)} != {len(p1)}")
    for i in range(len(p0)):
      if ((p0[i] == '0') or (p0[i] == '1')) == False:
        raise TypeError(f"p0 must be a binary string, p0[{i}] = {p0[i]}")
      elif ((p1[i] == '0') or (p1[i] == '1')) == False:
        raise TypeError(f"p1 must be a binary string, p1[{i}] = {p0[i]}")
    """
      Check that the number of crossover points is valid, there should be at
        least one and not more n/2 - 1 (for example n = 8,  0 < cP < 4)
    """
    if self.cP >= self.n/2:
      raise(TypeError(f"Number crossover points should be less than the half the string length"))
    if (self.cP < 1 or self.cP > 3):
      raise(TypeError(f"Number crossover points should be between 1 and 3"))

  def crossover(self):
    """
    """
    p0 = self.p0
    p1 = self.p1
    crossoverPoints = []
    if self.cP == 1:
      crossoverPoints.append(int(self.n/2)) #middle
      crossoverPoints += np.random.randint(-(int(self.n/8)),int(self.n/8)+1, self.cP)
      c0 = p0[:crossoverPoints[0]]
      c0 += p1[crossoverPoints[0]:]

      c1 = p1[:crossoverPoints[0]]
      c1 += p0[crossoverPoints[0]:]
    elif self.cP == 2:
        crossoverPoints.append(int(self.n/4)) # first quartile
        crossoverPoints.append(int(3*self.n/4)) # third quartile
        crossoverPoints += np.random.randint(-(int(self.n/8)),int(self.n/8)+1, self.cP)
        c0 = p0[:crossoverPoints[0]]
        c0 += p1[crossoverPoints[0]:crossoverPoints[1]]
        c0 += p0[crossoverPoints[1]:]

        c1 = p1[:crossoverPoints[0]]
        c1 += p0[crossoverPoints[0]:crossoverPoints[1]]
        c1 += p1[crossoverPoints[1]:]
    else:
        crossoverPoints.append(int(self.n/4)) # first quartile
        crossoverPoints.append(int(self.n/2)) # middle
        crossoverPoints.append(int(3*self.n/4)) # third quartile
        crossoverPoints += np.random.randint(-(int(self.n/8)),int(self.n/8)+1, self.cP)
        c0 = p0[:crossoverPoints[0]]
        c0 += p1[crossoverPoints[0]:crossoverPoints[1]]
        c0 += p0[crossoverPoints[1]:crossoverPoints[2]]
        c0 += p1[crossoverPoints[2]:]

        c1 = p1[:crossoverPoints[0]]
        c1 += p0[crossoverPoints[0]:crossoverPoints[1]]
        c1 += p1[crossoverPoints[1]:crossoverPoints[2]]
        c1 += p0[crossoverPoints[2]:]

    self.crossoverPoints = crossoverPoints
    self.c0 = c0
    self.c1 = c1


  def mutate(self):
    """
    """
    #number of mutations that should occur
    m = int(self.n * self.mR)
    listIndices = np.linspace(0,self.n-1,self.n, dtype='int')
    np.random.shuffle(listIndices)
    
    c0Mut = np.array(list(self.c0), dtype='int')
    for i in range(m):
      #get index of mutation
      idx = listIndices[i]
      c0Mut[idx] = np.abs(int(c0Mut[idx])-1)

    np.random.shuffle(listIndices)
    c1Mut = np.array(list(self.c0), dtype='int')
    for i in range(m):
      #get index of mutation
      idx = listIndices[i]
      c1Mut[idx] = np.abs(int(c1Mut[idx])-1)
    
    self.c0Mutated = ''.join(map(str,c0Mut))
    self.c1Mutated = ''.join(map(str,c1Mut))

  def operate(self):
    self.crossover()
    self.mutate()