import Support
import numpy as np
from skimage.measure import grid_points_in_poly
from skimage import filters
from shapely.geometry import LineString
import matplotlib.pyplot as plt

global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords

class ProfileGeneration():
  """
      Supporting Functions for Generating Profiles for Inverse Design
  """
  def __init__(self, nVertices, **kwargs):
      self.nVertices = nVertices
      self.rMin = Support.getkwarg(kwargs, defaultKwargs["rMin"], keywords["rMin"])
      self.rMax = Support.getkwarg(kwargs, defaultKwargs["rMax"], keywords["rMax"])
      self.s = Support.getkwarg(kwargs, defaultKwargs["s"], keywords["s"])
      self.dom = Support.getkwarg(kwargs, defaultKwargs["d"], keywords["d"])
      self.p = Support.getkwarg(kwargs, defaultKwargs["p"], keywords["p"])
      self.r = np.zeros(self.nVertices)
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      self.x = np.zeros(self.nVertices)
      self.y = np.zeros(self.nVertices)

  def fourierGenerator(self, nTerms):
      self.r = np.zeros(self.nVertices)
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      #setup random fourier series terms
      c = np.random.rand(nTerms)
      d = np.random.rand(nTerms)
      #fourier series terms should sum to rMax for scaling purposes
      c = self.rMax*(c / sum(c))
      d = self.rMax*(d / sum(d))
      for v in range(self.nVertices):
          for n in range(nTerms):
              self.r[v] += c[n]*np.sin(n*self.t[v])+d[n]*np.cos(n*self.t[v])

      #restrict r to the range [rMin, rMax]
      self.r = self.r + (self.r <= self.rMin)*(self.rMin-self.r)
      self.r = self.r + (self.r >= self.rMax)*(self.rMax-self.r)
      
  def circleGenerator(self):
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      self.fourierGenerator(1)
      self.r = np.random.randint(self.rMin, self.rMax)*(self.r / np.max(self.r))
      

  def rectangleGenerator(self, spanMin = 40, spanMax = 135):
      rectW = np.random.randint(spanMin, spanMax)
      rectH = np.random.randint(spanMin, spanMax)
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      self.r = np.zeros(self.nVertices)
      #calculate rays cast far from the origin
      x, y = 240*np.cos(self.t), 240*np.sin(self.t)
      for v in range(self.nVertices):
          if np.abs(y[v]/x[v]) > rectH/rectW:
              #The far ray intersects with the top or bottom of the rectangle
              self.r[v] = (rectH/2)/np.abs(np.sin(self.t[v]))
          else:
              #The far ray intersects with the side of the rectangle
              self.r[v] = (rectW/2)/np.abs(np.cos(self.t[v]))

      
  def triangleGenerator(self):
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      self.r = np.zeros(self.nVertices)

      #places three random vertices within the domain
      v0 = np.random.randint(-70,70, 2)
      v1 = np.random.randint(-70,70, 2)
      v2 = np.random.randint(-70,70, 2)
      x1, y1 = v0
      x2, y2 = v1
      x3, y3 = v2
      area = 0.5 * np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
      while area <= 4*np.pi*self.rMin*self.rMin:
          v0 = np.random.randint(-70,70, 2)
          v1 = np.random.randint(-70,70, 2)
          v2 = np.random.randint(-70,70, 2)
          x1, y1 = v0
          x2, y2 = v1
          x3, y3 = v2
          area = 0.5 * np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
      #find the center of the triangle
      xC = np.mean([v0[0], v1[0], v2[0]])
      yC = np.mean([v0[1], v1[1], v2[1]])
      #shift each vertice so that the triangle is centered at the origin
      vC0 = (v0[0]-xC, v0[1]-yC)
      vC1 = (v1[0]-xC, v1[1]-yC)
      vC2 = (v2[0]-xC, v2[1]-yC)
      
      for v in range(self.nVertices):
          ray = LineString([(0,0), (180*np.cos(self.t[v]), 180*np.sin(self.t[v]))])
          edge01 = LineString([(vC0[0], vC0[1]),(vC1[0], vC1[1])])
          edge12 = LineString([(vC1[0], vC1[1]),(vC2[0], vC2[1])])
          edge20 = LineString([(vC2[0], vC2[1]),(vC0[0], vC0[1])])

          edge01Int = ray.intersection(edge01)
          edge12Int = ray.intersection(edge12)
          edge20Int = ray.intersection(edge20)

          if np.shape(edge01Int.coords)[0] != 0:
              x, y = edge01Int.coords[0][0], edge01Int.coords[0][1]
          elif np.shape(edge12Int.coords)[0] != 0:
              x, y = edge12Int.coords[0][0], edge12Int.coords[0][1]
          elif np.shape(edge20Int.coords)[0] != 0:
              x, y = edge20Int.coords[0][0], edge20Int.coords[0][1]
          else:
              x, y = 0, 0
          
          self.r[v] = np.sqrt(x**2 + y**2)

  def polygonGenerator(self):
      self.t = np.linspace(0,2*np.pi, self.nVertices)
      self.r = np.random.randint(self.rMin, self.rMax, self.nVertices)
      self.r = self.r*1.0
      self.r[-1] = self.r[0]


  def polar2Cartesian(self):
      #find the radial values that are permitted by the encoding scheme
      self.__getCoding()
      for v in range(self.nVertices):
          self.r[v] = Support.closestRadius(self.r[v], self.rMin, self.rMax, self.p)
      self.x = self.r*np.cos(self.t) + self.dom[0]/2
      self.y = self.r*np.sin(self.t) + self.dom[1]/2
      xCenter, yCenter = np.mean(self.x), np.mean(self.y)
      a = np.arctan2(self.y - yCenter, self.x - xCenter)
      sortIdx = np.argsort(a)
      self.x, self.y = self.x[sortIdx], self.y[sortIdx]
      self.x = np.append(self.x, self.x[0]) - (xCenter - self.dom[0]/2)
      self.y = np.append(self.y, self.y[0]) - (yCenter - self.dom[1]/2)

  def arrayConversion(self):
      self.polar2Cartesian()
      self.cartPolygon = np.stack((self.y,self.x), axis=1)
      self.arrayImage = grid_points_in_poly(self.dom, self.cartPolygon).astype(int)
      self.smoothedImage = filters.gaussian(self.arrayImage, self.s)
      self.smoothedImage = np.round(self.smoothedImage / np.max(self.smoothedImage))
  
  def __getCoding(self):
    """
    create binary encoding and decoding key from allowed radius range
      and binary precision 
      (private)
    """
    self.c = np.linspace(self.rMin, self.rMax, 2**self.p)

  def __encodeRadius(self, v):
    """
    encode a specified radial value to binary using a binary encoding
      decoding key generated from __getCoding() 
      (private)
    """
    #create binary encoding and decoding key
    self.__getCoding()
    #find the index of the radial value from the binary key
    idx = list(self.c).index(v)
    #format that index to a binary string
    B = format(idx, "b")
    #append zeros to the front of the binary string to preserve precision
    for i in range(self.p - len(B)):
      B = '0' + B
    return B

  def __binary2Integer(self,b):
    """
    convert input binary string to an integer value
    """
    b = b[::-1]
    temp = 0
    for i in range(len(b)):
      temp += int(b[i])*(2**i)
    return temp

  def __decodeRadius(self, b):
    """
    decode binary encoded radial value using a binary encoding
      decoding key generated from __getCoding() 
      (private)
    """
    self.__getCoding()
    return self.c[self.__binary2Integer(b)]

  def encodePolygon(self):
    """
    """
    self.binaryPolygon = ''
    for v in range(self.nVertices):
      self.binaryPolygon += self.__encodeRadius(self.r[v])

  def decodePolygon(self):
    """
    """
    n = int(len(self.binaryPolygon) / self.p)
    self.decodedPolygon = np.zeros(n)
    for i in range(n):
      temp = self.binaryPolygon[self.p*i: (self.p*i+self.p)]
      self.decodedPolygon[i] = self.__decodeRadius(temp)

  def decodeChromosome(self, chromosome):
    self.binaryPolygon = chromosome
    self.decodePolygon()
    self.r = self.decodedPolygon
    self.arrayConversion()

  def generate(self, generator=None):
    if generator == None:
      x = np.random.randint(4)
      if x == 0:
        self.triangleGenerator()
      elif x == 1:
        self.rectangleGenerator()
      elif x == 2:
        self.circleGenerator()
      else:
        self.fourierGenerator(np.random.randint(2,4))

    elif generator == 'tri':
      self.triangleGenerator()
    elif generator == 'rec':
      self.rectangleGenerator()
    elif generator == 'cir':
      self.circleGenerator()
    elif generator == 'pol':
      self.polygonGenerator()
    else:
      self.fourierGenerator(np.random.randint(2,4))

    self.arrayConversion()
    self.encodePolygon()




    




  def testDecoder(self):
    #test polygonGenerator
    self.polygonGenerator()
    self.arrayConversion()
    self.encodePolygon()
    self.decodePolygon()
    if all(self.r == self.decodedPolygon):
      print("Polygon Generator Decoding is Okay")
    else:
      print("Problem with Polygon Generator Decoding")

    #test fourierGenerator
    self.fourierGenerator(10)
    self.arrayConversion()
    self.encodePolygon()
    self.decodePolygon()
    if all(self.r == self.decodedPolygon):
      print("Fourier Generator Decoding is Okay")
    else:
      print("Problem with Fourier Generator Decoding")

    #test fourierGenerator
    self.triangleGenerator()
    self.arrayConversion()
    self.encodePolygon()
    self.decodePolygon()
    if all(self.r == self.decodedPolygon):
      print("Triangle Generator Decoding is Okay")
    else:
      print("Problem with Triangle Generator Decoding")

    #test rectangleGenerator
    self.rectangleGenerator()
    self.arrayConversion()
    self.encodePolygon()
    self.decodePolygon()
    if all(self.r == self.decodedPolygon):
      print("Rectangle Generator Decoding is Okay")
    else:
      print("Problem with Rectangle Generator Decoding")

    #test circleGenerator
    self.rectangleGenerator()
    self.arrayConversion()
    self.encodePolygon()
    self.decodePolygon()
    if all(self.r == self.decodedPolygon):
      print("Circle Generator Decoding is Okay")
    else:
      print("Problem with Circle Generator Decoding")




def generateSampleProfiles(nProfiles, nVertices):
    dom = defaultKwargs["d"]
    sampleProfiles = np.zeros((nProfiles, dom[0], dom[1]))
    n = 0
    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.fourierGenerator(2)
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.fourierGenerator(3)
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.fourierGenerator(4)
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.triangleGenerator()
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.circleGenerator()
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(10):
        p = ProfileGeneration(nVertices)
        p.rectangleGenerator()
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    for i in range(40):
        p = ProfileGeneration(nVertices)
        p.polygonGenerator()
        p.arrayConversion()
        sampleProfiles[n, :, :] = p.smoothedImage
        n += 1

    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sampleProfiles[i, :, :], cmap=plt.cm.binary)
    plt.savefig("SampleProfiles.png")



def main():
    generateSampleProfiles(100, 32)

if __name__ == "__main__":
    main()