import ProfileGeneration as pg


def Encode(v, coding):
  BinaryPrecision = getPrecision(coding)
  idx = list(coding).index(v)
  B = format(idx, "b")
  for i in range(BinaryPrecision - len(B)):
    B = '0' + B
  return B

def Decode(b, coding):
  return coding[Binary2Int(b)]

def Binary2Int(B):
  B = B[::-1]
  temp = 0
  for i in range(len(B)):
    temp += int(B[i])*(2**i)

def getPrecision(coding):
  return int(np.log(len(coding)) / np.log(2))

def DecodeBinaryList(binaryList, coding):
  P = getPrecision(coding)
  N = int(len(binaryList) / P)
  DecodedList = np.zeros(N)
  for i in range(N):
    CurrentBinary = binaryList[P*i: (P*i + P)]
    DecodedList[i] = Decode(CurrentBinary, coding)
  return DecodedList

coding = np.linspace(self.r0, self.r1, 2**(self.p))

p = pg.ProfileGeneration(30)
p.fourierGenerator(2)
p.arrayConversion()

