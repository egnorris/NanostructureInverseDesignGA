import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../modules")
import InverseDesign as invdes


def calculateEntropy(pop):
    c = np.zeros(len(pop.chromosomes[0]))
    for i in range(len(pop.chromosomes)):
        ch = pop.chromosomes[i]
        for j in range(len(c)):
            c[j] += int(ch[j])
    p = c / pop.nProfiles   
    H = 0
    for j in range(len(c)):
        if p[j] == 0:
            H += 0
        else:
            H += (-1)*p[j]*np.log(p[j])
    return H

entr = np.zeros((10,100))

for k in range(np.shape(entr)[0]):
    np.random.seed(0)
    pop = invdes.Population(24, "/media/work/evan/deep_learning_data/trained_models")
    pop.readObjective("input/objScatteredPower0.txt")
    pop.initialize(nT = 10, nC = 10, nP = 10, nR = 10, nN = 10)
    print(f"Generation 0 Entropy: {calculateEntropy(pop)}")
    entr[k,0] = calculateEntropy(pop)
    for n in range(np.shape(entr)[1]-1):
        pop.update(birthRate=k/10)
        print(f"Generation {n+1} Entropy: {calculateEntropy(pop)}")
        entr[k,n+1] = calculateEntropy(pop)

for i in range(np.shape(entr)[0]):
    plt.scatter(entr[i,:], label=f'Birthrate: {i/10}')
plt.legend()
plt.xlabel("Generations")
plt.ylabel("Entropy")
plt.savefig("entropy.png")


