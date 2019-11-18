import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

f = open("Errors/simple50/g_errors.txt", "r")
f1 = open("Errors/simple50/d_errors.txt", "r")
f = f.read()
f1 = f1.read()
iteration = []
mu = []
a = []
b = []
train_ratio = []
dist_mu = []
dist_a = []
dist_b = []
for line in f.split("\n"):
    if line.find(";") != -1:
        split = line.split(";")
        mu.append(float(split[0]))
        iteration.append(int(split[1]))
        train_ratio.append(split[-1])
for line in f1.split("\n"):
    if line.find(";") != -1:
        split = line.split(";")
        a.append(float(split[0]))
        b.append(float(split[1]))
# plt.plot(iteration, mu)
# plt.ylabel("Generator Mu")
# plt.xlabel("Training iteration")
# plt.savefig("Graphs/mu_convergence")
print(train_ratio)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter3D(mu, a, b)
# ax.set_xlabel("Mu")
# ax.set_ylabel("a")
# ax.set_zlabel("b")
# plt.show()

zipped = zip(mu, a, b, train_ratio)
dist_a1 = []
dist_b1 = []
dist_a2 = []
dist_b2 = []
dist_a3 = []
dist_b3 = []
dist_a4 = []
dist_b4 = []
dist_a5 = []
dist_b5 = []
dist_a10 = []
dist_b10 = []
dist_a20 = []
dist_b20 = []
dist_a50 = []
dist_b50 = []
for i, element in enumerate(zipped):
    if i%10 == 0:
        dist_mu.append(0-element[0])
        theoretical_a = 0-element[0]
        theoretical_b = 1/2*(0-element[0]**2)
        if element[3] == '1':
            dist_a1.append(theoretical_a - element[1])
            dist_b1.append(theoretical_b - element[2])
        elif element[3] == '2':
            dist_a2.append(theoretical_a - element[1])
            dist_b2.append(theoretical_b - element[2])
        elif element[3] == '3':
            dist_a3.append(theoretical_a - element[1])
            dist_b3.append(theoretical_b - element[2])
        elif element[3] == '4':
            dist_a4.append(theoretical_a - element[1])
            dist_b4.append(theoretical_b - element[2])
        elif element[3] == '5':
            dist_a5.append(theoretical_a - element[1])
            dist_b5.append(theoretical_b - element[2])
        elif element[3] == '10':
            dist_a10.append(theoretical_a - element[1])
            dist_b10.append(theoretical_b - element[2])
        elif element[3] == '20':
            dist_a20.append(theoretical_a - element[1])
            dist_b20.append(theoretical_b - element[2])
        elif element[3] == '50':
            dist_a50.append(theoretical_a - element[1])
            dist_b50.append(theoretical_b - element[2])
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter3D(dist_mu, dist_a, dist_b)
# ax.set_xlabel("Distance to Mu")
# ax.set_ylabel("Distance to a")
# ax.set_zlabel("Distance to b")
# plt.show()
fig = plt.figure()
plt.plot(dist_a1, dist_b1, 'o', color='black', label='m=1')
plt.plot(dist_a2, dist_b2, 'o', color='red', label='m=2')
plt.plot(dist_a3, dist_b3, 'o', color='magenta', label='m=3')
plt.plot(dist_a4, dist_b4, 'o', color='purple', label='m=4')
plt.plot(dist_a5, dist_b5, 'o', color='blue', label='m=5')
plt.plot(dist_a10, dist_b10, 'o', color='orange', label='m=10')
plt.plot(dist_a20, dist_b20, 'o', color='green', label='m=20')
plt.plot(dist_a50, dist_b50, 'o', color='yellow', label='m=50')
plt.xlabel("Distance to theoretical a")
plt.ylabel("Distance to theoretical b")
plt.legend(loc="upper left")
plt.show()
