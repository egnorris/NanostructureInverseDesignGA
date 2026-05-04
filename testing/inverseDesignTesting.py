import sys
import matplotlib.pyplot as plt

sys.path.append("../modules")
import InverseDesign as invdes

pop = invdes.Population(24, "/media/work/evan/deep_learning_data/trained_models")
pop.readObjective("input/objScatteredPower0.txt")
pop.initialize(nT = 10, nC = 10, nP = 10, nR = 10, nN = 10)

plt.plot(pop.cartesian[0,:,0],pop.cartesian[0,:,1])
plt.savefig("temp0.png")
plt.close()
print(pop.fitness)
pop.update(birthRate=0.8)

plt.plot(pop.cartesian[0,:,0],pop.cartesian[0,:,1])
plt.savefig("temp1.png")
plt.close()
print(pop.fitness)