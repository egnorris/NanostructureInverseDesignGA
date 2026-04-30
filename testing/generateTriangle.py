#Example of using ProfileGeneration.py to generate a triangular polygon
import sys
import matplotlib.pyplot as plt

sys.path.append("../modules")
import ProfileGeneration

#initialize profile generation class

sigma = 5       #Images will be smoothed by a gaussian blur with sigma=5
precision = 12  #Encoding will occur with a binary precision of 12
#Polygons generated will have 24 Vertices with radial values between 20 and 80
pg = ProfileGeneration.ProfileGeneration(nVertices=24,rMin=20,rMax=80,s=5,p=12)

#generate a triangular polygon
pg.triangleGenerator()
#convert radial paramters into a smoothed image
pg.arrayConversion()

#plot the different realizations of the triangle
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot(221,polar=True)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
ax1.set_rlim(0,90)
ax2.set_xlim(0,180); ax2.set_xticks([])
ax2.set_ylim(0,180); ax2.set_yticks([])
ax3.set_xticks([]); ax4.set_xticks([])
ax3.set_yticks([]); ax4.set_yticks([])
ax1.set_title("Radial Parameterization", fontsize=18)
ax2.set_title("Cartesian Coordinates", fontsize=18)
ax3.set_title("Array Image", fontsize=18)
ax4.set_title("Smoothed Image", fontsize=18)
ax1.plot(pg.t, pg.r, c='tab:red', zorder= 1)
ax1.scatter(pg.t, pg.r, c='tab:blue', s=10, zorder= 2)
ax2.plot(pg.x, pg.y, c='tab:red', zorder= 1)
ax2.scatter(pg.x, pg.y, c='tab:blue', s=10, zorder= 2)
ax3.imshow(pg.arrayImage, cmap='binary')
ax3.plot(pg.x, pg.y, c='tab:red', zorder= 1)
ax3.scatter(pg.x, pg.y, c='tab:blue', s=10, zorder= 2)
ax4.imshow(pg.smoothedImage, cmap='binary')
ax4.plot(pg.x, pg.y, c='tab:red', zorder= 1)
ax4.scatter(pg.x, pg.y, c='tab:blue', s=10, zorder= 2)
plt.savefig("triangleGeneration.png")
plt.close()

#verify that shape encoding/decoding is working properly
#encode polygon to chromosome
pg.encodePolygon()
originalTri = pg.r
encodedTri = pg.binaryPolygon
#decode chromosome back to polygon
pg.decodeChromosome(encodedTri)
decodedTri = pg.decodedPolygon
print(f"Encoded Chromosome:     {encodedTri}")
print(f"Original Radial values: {originalTri}")
print(f"Decoded Radial values:  {decodedTri}")
#check all of the triangle radii and and return false if any radii doesn't match
print(f"Encoding Successful?: {any(decodedTri == originalTri)}")




