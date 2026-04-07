import Support as sp
import numpy as np
import GeneticOperations as go
import ProfileGeneration as pg
import DeepLearning as dl
import InverseDesign as invdes
import matplotlib.pyplot as plt


pop = invdes.Population(100, 30,"/media/work/evan/deep_learning_data/trained_models", mR = 0.2, cP = 1)
print(np.mean(pop.fitness))

plt.imshow(pop.objectivePoly)
plt.savefig("obj.png")

plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pop.images[i, :, :], cmap=plt.cm.binary)
plt.show()
plt.savefig("gen0.png")
for n in range(10):
    pop.reproduction()
    print(np.mean(pop.fitness))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pop.images[i, :, :], cmap=plt.cm.binary)
    plt.show()
    plt.savefig(f"gen{n+1}.png")


"""
N = 100
Images = np.zeros((N, 180, 180))
BinaryPolygons = []


p = pg.ProfileGeneration(30)
for i in range(N):
    p.generate('tri')
    Images[i, :, :] = p.smoothedImage
    BinaryPolygons.append(p.binaryPolygon)
    print(np.shape(p.binaryPolygon))



for i in range(N):
    id0 = np.random.randint(N)
    id1 = np.random.randint(N)
    while id0 == id1:
        id1 = np.random.randint(N)
    p0 = BinaryPolygons[id0]
    p1 = BinaryPolygons[id1]
    g = go.GeneticOperations(p0, p1)
    g.operate()
    p.decodeChromosome(g.c0)
    Images[id0, :, :] = p.smoothedImage
    p.decodeChromosome(g.c1)
    Images[id1, :, :] = p.smoothedImage

 """

