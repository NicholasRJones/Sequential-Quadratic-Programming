import numpy as np
from Optimization.Algorithm import classy as cl, SQP
import matplotlib.pyplot as plt


def bowl(input, para, p):
    x = np.array(input)
    if p == 0:
        return np.dot(x, x)
    if p == 1:
        return 2 * x
    if p == 2:
        return np.dot(x, x), 2 * x


def E(input):
    return np.array([0])


def gE(input):
    return np.zeros(len(input))


def I(input):
    return np.array([input[0] ** 2 + input[1] ** 2 - np.exp(input[0]) - np.sqrt(abs(input[0] + input[1]) + 0.00001)])


def gI(input):
    return np.array(
        [2 * input[0] - np.exp(input[0]) - (input[0] + input[1]) / (2 * (abs(input[0] + input[1]) + 0.00001) ** (3 / 2)),
         2 * input[1] - (input[0] + input[1]) / (2 * (abs(input[0] + input[1]) + 0.00001) ** (3 / 2))])


be = np.array([0])
bi = np.array([-1])

input = [-1, -1]

quad = cl.quad(E, gE, be, I, gI, bi)
para = cl.para(0.0001, 0.19, 0, 0, quad, 0, 0)
pr = cl.funct(bowl, '', '', input, para, 1)


x, stor = SQP.SQP(pr)

axis1 = []
axis2 = []
for j in range(len(stor.inp)):
    axis1.append(stor.inp[j][0])
    axis2.append(stor.inp[j][1])


a3 = np.linspace(-1.5, 1.5, 100)
a4 = np.linspace(-1.5, 1.5, 100)
axis3 = []
axis4 = []
for j in range(len(a3)):
    for i in range(len(a3)):
        if I([a3[j], a4[i]]) >= -1:
            axis3.append(a3[j])
            axis4.append(a4[i])

plt.plot(axis3, axis4, color='green', marker='o', linestyle='none')
plt.plot(axis1, axis2, color="maroon", marker='o')
plt.xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
plt.show()
