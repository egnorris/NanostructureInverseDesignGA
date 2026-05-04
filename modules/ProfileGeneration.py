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

    def polygonGenerator(self):
        self.t = np.linspace(0,2*np.pi, self.nVertices)
        self.r = np.random.randint(self.rMin, self.rMax, self.nVertices)
        self.r = self.r*1.0
        self.r[-1] = self.r[0]


    def __originCentering(self, v):
        #the center of a polygon is the mean of it's vertices
        xc, yc = np.mean(v[:-1,0]), np.mean(v[:-1,1])
        v[:,0], v[:,1] = [v[:,0] - xc,  v[:,1] - yc]
        return v


    def __dist(self, p0, p1):
        return np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2) 
    
    def __generateVertices(self,n):
        v = np.zeros((n+1, 2))
        t = np.random.rand(n)*2*np.pi
        sortIdx = np.argsort(t)
        t = t[sortIdx]
        for i in range(n):
            #generate a random allowed position
            r = np.random.randint(self.rMin, self.rMax)
            print(t)
            x = r*np.cos(t[i]) + self.dom[0]/2
            y = r*np.sin(t[i]) + self.dom[1]/2
            v[i, :] = [x,y]
        v[-1,:] = v[0, :]
        return v

    def __getEdges(self, n, v):
        e = np.zeros((n,4))
        l = np.zeros(n)
        for i in range(n):
            l[i] = self.__dist(v[i],v[i+1])
            e[i, 0] = v[i, 0]; e[i, 1] = v[i+1, 0]
            e[i, 2] = v[i, 1]; e[i, 3] = v[i+1, 1]
        return e, l

    def __placeVertices(self, n):
        vertex = self.__generateVertices(n)
        edge, edgeLength = self.__getEdges(n, vertex)
        while (np.sum(edgeLength) < 4*np.pi*self.rMin) or any(edgeLength > 3*np.min(edgeLength)):
            vertex = self.__generateVertices(n)
            vertex = self.__originCentering(vertex)
            edge, edgeLength = self.__getEdges(n, vertex)
        
        return vertex, edge, edgeLength

    def ___getIntersection(self, edge, angle):
        """
        """
        rayString = LineString([(0,0), (self.dom[0]*np.cos(angle), self.dom[1]*np.sin(angle))])
        for i in range(len(edge)):
            e = edge[i, :]
            edgeString = LineString([(e[0], e[2]), (e[1], e[3])])
            temp = rayString.intersection(edgeString)
            if np.shape(temp.coords)[0] != 0:
                return temp.coords[0]
            

    def ngonGenerator(self, n):
        """
        """
        ngonVertices, edge, edgeLength = self.__placeVertices(n)
        self.t = np.linspace(0,2*np.pi, self.nVertices)
        self.r = np.zeros(self.nVertices)

        self.ngonVertices = ngonVertices
        for i in range(self.nVertices):
            x, y = self.___getIntersection(edge, self.t[i])
            #x = x + self.dom[0]/2; y = y + self.dom[1]/2; 
            self.r[i] = np.sqrt(x**2 + y**2)

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


    