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

entr = np.zeros((5,50,100))
for s in range(np.shape(entr)[2]):
    for k in range(np.shape(entr)[0]):
        np.random.seed(s)
        pop = invdes.Population(24, "/media/work/evan/deep_learning_data/trained_models")
        pop.readObjective("input/objScatteredPower0.txt")
        pop.initialize(nT = 10, nC = 10, nP = 10, nR = 10, nN = 10)
        print(f"Seed: {s}")
        print(f"Birth Rate: {(k+1)/np.shape(entr)[0]}")
        print(f"Generation 0 Entropy: {calculateEntropy(pop)}")
        entr[k,0, s] = calculateEntropy(pop)
        for n in range(np.shape(entr)[1]-1):
            pop.update(birthRate=(k+1)/np.shape(entr)[0])
            print(f"Seed: {s}")
            print(f"Birth Rate: {(k+1)/np.shape(entr)[0]}")
            print(f"Generation {n+1} Entropy: {calculateEntropy(pop)}")
            entr[k,n+1, s] = calculateEntropy(pop)


for i in range(np.shape(entr)[0]):
    s = np.var(entr[i,:, :], axis=1)
    m = np.mean(entr[i,:, :], axis=1)
    x = np.linspace(0, np.shape(entr)[1]-1, np.shape(entr)[1])
    plt.plot(x, m, 'o-', label=f'Birth Rate: {(i+1)/np.shape(entr)[0]}')
    plt.fill_between(x, m-s, m+s, alpha=0.1)
plt.legend(loc='lower left')
plt.xlabel("Generations")
plt.ylabel("Entropy")
    
plt.savefig("entropy.png")


