# ProfileGeneration.py
## About
This module primarily houses the ProfileGeneration() class which manages everything for the generation of a single geometry profile. Geometry profiles can be expressed in the following forms,

* Cartesian coordinate list
* polar coordinate list
* Binary encoded string
* Image array

## Radial Paramterization

In the context of the downstream genetic algorithm these geometry profiles describe different realization of parameter space. An example profile geometry is shown below:

<img src="/images/polygonParamterizationDiagram.png" alt="drawing" width="300"/>

each red dot indicates an individual element of a profile in polar coordinates, represented as an ordered pair, $(\theta_n,r_n)$ where $r_n$ is the point along the line cast from the center to the edge of the domain that intersects with the polygon, this realization of the polygon is refered to as the "radial parameterization" later on


## Binary Representation
in the genetic operations expect profiles to be in a binary representation, that conversion is handled in this module with encodePolygon() and the binary representation can be decoded back to the radial form with decodePolygon() 


```Python
def encodePolygon(self):
self.binaryPolygon = ''
for v in range(self.nVertices):
    self.binaryPolygon += self.__encodeRadius(self.r[v])
```

```Python
def decodePolygon(self):
n = int(len(self.binaryPolygon) / self.p)
self.decodedPolygon = np.zeros(n)
for i in range(n):
    temp = self.binaryPolygon[self.p*i: (self.p*i+self.p)]
    self.decodedPolygon[i] = self.__decodeRadius(temp)
```

these functions are supported by the following additional private functions

```Python
def __getCoding(self):
self.c = np.linspace(self.rMin, self.rMax, 2**self.p)
```

the coding is a key for associating radial values with a binary index value defined by the binary precision, this approach reduces hidden precision truncation when dealing with the floating point radial values and just overal is very simple to implement across many different optimization problems

```Python
def __binary2Integer(self,b):
b = b[::-1]
temp = 0
for i in range(len(b)):
    temp += int(b[i])*(2**i)
return temp
```

converting from binary numbers to integers is a well documented process, each binary element is multiplied by 2 raised to the power of it's digit placement, the sum of all of these digits in this process gives an integer index that can be fed into the coding

```Python
def __decodeRadius(self, b):
self.__getCoding()
return self.c[self.__binary2Integer(b)]
```

decoding a binary number to a radial value follows the previous code and just inputs the index to the coding to recover the allowed radial value

```Python
  def __encodeRadius(self, v):
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
```

encoding a radial value to a binary number is similarly simple, the index of the coding corresponding to the known radial value is found, that index is then formatted to a binary string. When formatting to a binary string the precision is limited to whatever the smallest value that can fit that integer but doesn't necessarily match the defined binary precision so additional '0' are added to the left hand side for precision matching.


```Python
for v in range(self.nVertices):
    self.binaryPolygon += self.__encodeRadius(self.r[v])
```
During encoding this process is repeated in a for-loop across every radial value in the list to give a single large binary string

```Python
n = int(len(self.binaryPolygon) / self.p)
self.decodedPolygon = np.zeros(n)
```

decoding is a bit more complicated, first from the length of the full binary polygon string and the binary precision the number of vertices is recovered.

```Python
for i in range(n):
    temp = self.binaryPolygon[self.p*i: (self.p*i+self.p)]
    self.decodedPolygon[i] = self.__decodeRadius(temp)
```

then each sub-string of the binary polygon string which are single binary radii are individual decoded and stored back inot a decoded polygon list. 


## Image Rendering
For interoperability with trained neural networks later on radial paramterization needs to be transformed into an image array, this is handled by arrayConversion() and polar2Cartesian()

```Python
def arrayConversion(self):
    self.polar2Cartesian()
    self.cartPolygon = np.stack((self.y,self.x), axis=1)
    self.arrayImage = grid_points_in_poly(self.dom, self.cartPolygon).astype(int)
    self.smoothedImage = filters.gaussian(self.arrayImage, self.s)
    self.smoothedImage = np.round(self.smoothedImage / np.max(self.smoothedImage))
```

``` Python
def polar2Cartesian(self):
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

```

Let's break down what's happening between these two functions, before the generated polar coordiantes can be converted to cartesian coordinates the individual radii need to comply with the coding discussed in the binary represenation section. So a radial value is replaced with the closest allowed radius

``` Python
self.__getCoding()
for v in range(self.nVertices):
    self.r[v] = Support.closestRadius(self.r[v], self.rMin, self.rMax, self.p)
```

from there the polar coordinate conversion follows directly with transformation from an origin at $(0,0)$ to an origin in the center of the domain

``` Python
self.x = self.r*np.cos(self.t) + self.dom[0]/2
self.y = self.r*np.sin(self.t) + self.dom[1]/2
```

this process could end at this point but an sorting of the vertices by angle is performed to guarantee that the polygon is always 'untangled' which is to say that no two polygon edges should intersect.

```Python
xCenter, yCenter = np.mean(self.x), np.mean(self.y)
a = np.arctan2(self.y - yCenter, self.x - xCenter)
sortIdx = np.argsort(a)
self.x, self.y = self.x[sortIdx], self.y[sortIdx]
```

the polygon should be closed so the first element of the list is appended to the end and all of center of mass of the polygon is shifted to the center of the domain

```Python
self.x = np.append(self.x, self.x[0]) - (xCenter - self.dom[0]/2)
self.y = np.append(self.y, self.y[0]) - (yCenter - self.dom[1]/2)
```

from the cartesian coordinates an image array is drawn with grid_points_in_poly() from scikit-image which takes a look at each polygon coordinate and builds an array with ones within the polygon perimeter and zeros outside the polygon perimeter

```Python
self.cartPolygon = np.stack((self.y,self.x), axis=1)
self.arrayImage = grid_points_in_poly(self.dom, self.cartPolygon).astype(int)
```

the image array is then smoothed by a gaussin filter to remove any small features

```Python
self.smoothedImage = filters.gaussian(self.arrayImage, self.s)
self.smoothedImage = np.round(self.smoothedImage / np.max(self.smoothedImage))
```

this smoothing results in the polygon edges acting more like splines to guide the nanostructure profile shape but still act as a reliable parameterization of the nanostructure. 

<img src="/images/polygonSmoothing.png" alt="drawing" width="900"/>


```python
import sys

sys.path.append("../modules")
import ProfileGeneration
```



