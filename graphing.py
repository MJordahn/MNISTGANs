import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

f = open("Errors/simple100/g_errors.txt", "r")
f1 = open("Errors/simple100/d_errors.txt", "r")
f = f.read()
f1 = f1.read()
iteration = []
mu = []
a = []
b = []
dist_mu = []
dist_a = []
dist_b = []
for line in f.split("\n"):
    if line.find(";") != -1:
        split = line.split(";")
        mu.append(float(split[0]))
        iteration.append(int(split[1]))
for line in f1.split("\n"):
    if line.find(";") != -1:
        split = line.split(";")
        a.append(float(split[0]))
        b.append(float(split[1]))
plt.plot(iteration, mu)
plt.ylabel("Generator Mu")
plt.xlabel("Training iteration")
plt.savefig("Graphs/mu_convergence")

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(mu, a, b)
ax.set_xlabel("Mu")
ax.set_ylabel("a")
ax.set_zlabel("b")
plt.show()

zipped = zip(mu, a, b)
for element in zipped:
    dist_mu.append(2-element[0])
    theoretical_a = 2-element[0]
    theoretical_b = 1/2*(2**2-element[0]**2)
    dist_a.append(theoretical_a - element[1])
    dist_b.append(theoretical_b - element[2])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(dist_mu, dist_a, dist_b)
ax.set_xlabel("Distance to Mu")
ax.set_ylabel("Distance to a")
ax.set_zlabel("Distance to b")
plt.show()
