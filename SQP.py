import numpy as np
from Optimization.Algorithm import classy as cl, QP, criterion as ct

"""""""""
This file contains the code for the sequential quadratic program. This is a constrained optimization method for smooth functions.
This algorithm takes the function class input from the classy.py file.
"""""""""


def DampedBFGS(pr, stor, n):
    # Use the damped BFGS method for a hessian approximation
    if n == 0:
        initial = pr.function(pr.input, pr.para, 2)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(initial[1]))
        stor.H = np.identity(len(pr.input))
        stor.invH = stor.H + 0
    else:
        stor.H = stor.invH + 0
        sn = stor.inp[n] - stor.inp[n - 1]
        yn = stor.grad[n] - stor.grad[n - 1]
        prod = np.dot(sn, np.matmul(stor.H, sn))
        dot = np.dot(sn, yn)
        mult = np.matmul(stor.H, sn)
        if dot >= prod / 5:
            thetak = 1
        else:
            thetak = 4 / 5 * prod / (prod - dot)
        rn = thetak * yn + (1 - thetak) * mult
        stor.H += - np.matmul(mult, np.transpose(mult)) / prod + np.outer(rn, rn) / np.dot(sn, rn)
        stor.invH = stor.H + 0
        eig = np.min(np.linalg.eig(stor.H).eigenvalues)
        if eig <= 0:
            print('Hessian update: positive definite')
            stor.H = stor.H + (1 / 100 - eig) * np.identity(len(stor.grad[n]))


def SQP(pr):
    # Create storage class
    stor = cl.stor()
    n = 0
    c = False
    while not c:
        # Update Hessian
        DampedBFGS(pr, stor, n)
        # Print progress
        if n % pr.print == 0:
            print(str(n) + '___' + str(stor.inp[n]) + '___' + str(stor.val[n]))
        # Grab constraint information
        E, gE, be, I, gI, bi = pr.para.constraint.E(stor.inp[n]), pr.para.constraint.gE(stor.inp[n]), pr.para.constraint.be, pr.para.constraint.I(stor.inp[n]), pr.para.constraint.gI(stor.inp[n]), pr.para.constraint.bi
        # Evaluate violations
        CW = np.concatenate((abs(E - be), np.maximum(bi - I, np.zeros(len(I)))))
        # Initialize parameters
        if n == 0:
            mu = 10
        # Create new function class to solve for search direction
        quad = cl.quad(gE.reshape((len(E), len(gE))), '', be - E, gI.reshape((len(I), len(gI))), '', bi - I)
        para = cl.para('', '', pr.para.data, [stor.H, stor.grad[n]], quad, '', '')
        pR = cl.funct('', '', '', pr.input, para, 1000)
        xstep = QP.QP(pR)
        alpha = 0.9999
        # Chose appropriate directional derivative
        if CW.sum() == 0:
            directional = np.dot(stor.grad[n], xstep)
        else:
            mu = max(10, np.dot(stor.grad[n], xstep) / (CW.sum() / 2))
            directional = np.dot(stor.grad[n], xstep) - mu * CW.sum()
        # Search for improvement
        while pr.function(stor.inp[n] + alpha * xstep, pr.para, 0) + mu * np.array([abs(pr.para.constraint.E(stor.inp[n] + alpha * xstep) - be), np.maximum(bi - pr.para.constraint.I(stor.inp[n] + alpha * xstep), 0)]).sum() > stor.val[n] + mu * CW.sum() + pr.para.c1 * alpha * directional:
            alpha = alpha / 2
        # Update storage
        stor.inp.append(stor.inp[n] + alpha * xstep)
        stor.val.append(pr.function(stor.inp[n + 1], pr.para, 0))
        stor.grad.append(np.array(pr.function(stor.inp[n + 1], pr.para, 1)))
        stor.norm.append(np.linalg.norm(stor.grad[n + 1]))
        n += 1
        c = ct.criterion(stor, n)
    print(str(stor.inp[len(stor.inp) - 1]) + '___' + str(stor.val[len(stor.inp) - 1]))
    return pr, stor
