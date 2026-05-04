import Support
import numpy as np
import unittest
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

    self.c0 = self.c0Mutated
    self.c1 = self.c1Mutated

  def operate(self):
    self.crossover()
    self.mutate()






#unit Testing   
class TestGeneticOperations(unittest.TestCase):

  def testCrossover1(self):
    """
      One Point Crossover Validation Test 
    """
    p0 = '00001111'
    p1 = '01010101'
    go = GeneticOperations(p0, p1, mR=0.1, cP=1)
    go.crossover()
    cp = go.crossoverPoints
    self.assertTrue(go.c0 == p0[:cp[0]] + p1[cp[0]:])
    self.assertTrue(go.c1 == p1[:cp[0]] + p0[cp[0]:])

  def testCrossover2(self):
    """
      Two Point Crossover Validation Test 
    """
    p0 = '00001111'
    p1 = '01010101'
    go = GeneticOperations(p0, p1, mR=0.1, cP=2)
    go.crossover()
    cp = go.crossoverPoints
    self.assertTrue(go.c0 == p0[:cp[0]] + p1[cp[0]:cp[1]] + p0[cp[1]:])
    self.assertTrue(go.c1 == p1[:cp[0]] + p0[cp[0]:cp[1]] + p1[cp[1]:])


  def testCrossover3(self):
    """
      Three Point Crossover Validation Test 
    """
    p0 = '00001111'
    p1 = '01010101'
    go = GeneticOperations(p0, p1, mR=0.1, cP=3)
    go.crossover()
    cp = go.crossoverPoints
    self.assertTrue(go.c0 == p0[:cp[0]] + p1[cp[0]:cp[1]] + p0[cp[1]:cp[2]] + p1[cp[2]:])
    self.assertTrue(go.c1 == p1[:cp[0]] + p0[cp[0]:cp[1]] + p1[cp[1]:cp[2]] + p0[cp[2]:])


  def testMutation(self):
    """
      Mutation Test: mutated children should not match after mutation
    """
    p0 = '0000011111'
    p1 = '0101010101'
    go = GeneticOperations(p0, p1, mR=0.1, cP=1)
    go.crossover()
    c0 = go.c0
    c1 = go.c1
    go.mutate()
    self.assertFalse(c0 == go.c0)
    self.assertFalse(c1 == go.c1)


  def testCrossoverFormat(self):
    """
      Crossover Format Test: Crossover should result in children with the same length as the parents
    """
    for k in range(1,4):
        p0 = '00001111'
        p1 = '01010101'
        go = GeneticOperations(p0, p1, mR=0.1, cP=k)
        go.crossover()
        cp = go.crossoverPoints
        self.assertTrue(len(p0) == len(go.c0))
        self.assertTrue(len(p0) == len(go.c1))


  def testMutationaFormat(self):
    """
      Mutation Format Test: mutated children should retain their formatting
    """
    p0 = '00001111'
    p1 = '01010101'
    go = GeneticOperations(p0, p1, mR=0.1, cP=1)
    go.crossover()
    c0 = go.c0
    c1 = go.c1
    go.mutate()

    self.assertTrue(len(c0) == len(go.c0))
    self.assertTrue(len(c1) == len(go.c1))



if __name__ == '__main__':
    unittest.main()
    

    