import numpy as np
import scipy as sp

"""""""""
This file contains the code for the quadratic program algorithm. This is a method for optimizing linearly constrained quadratic functions.
This algorithm takes the function class from the classy.py file.
"""""""""


def work(A, b, x):
    # Check for active constrains
    index = np.array([])
    W = np.array([])
    for j in range(len(A)):
        if np.dot(A[j], x) - b[j] <= 0:
            index = np.insert(index, len(index), j)
            W = np.insert(W, len(W), A[j])
    W = W.reshape((len(index), len(x)))
    return W, index


def schur(G, W, WT):
    # Return the inverse KKT matrix
    invG = np.linalg.inv(G)
    F = -np.linalg.inv(np.matmul(W, np.matmul(invG, WT)))
    E = -np.matmul(invG, np.matmul(WT, F))
    C = invG - np.matmul(E, np.matmul(W, invG))
    return np.block([[C, E], [np.transpose(E), F]])


def initial(c, x, I, bi, E, be):
    # Find a feasible initial input
    if np.linalg.norm(I) > 0 and np.linalg.norm(E) > 0:
        x = sp.optimize.linprog(c, A_ub=-I, b_ub=-bi, A_eq=E, b_eq=be, bounds=(None, None)).x
    elif np.linalg.norm(I) > 0:
        x = sp.optimize.linprog(c, A_ub=-I, b_ub=-bi, A_eq=None, b_eq=None, bounds=(None, None)).x
    elif np.linalg.norm(E) > 0:
        x = sp.optimize.linprog(c, A_ub=None, b_lub=None, A_eq=E, b_eq=be, bounds=(None, None)).x
    else:
        x = np.zeros(len(x))
    return x


def lmbda(E, I, G, c, x):
    C = np.concatenate((E, I))
    S = np.linalg.norm(np.dot(G, x) + c)
    lmbda = []
    for j in range(len(C)):
        if np.linalg.norm(C[j]) == 0:
            lmbda.append(0)
        else:
            lmbda.append(S / np.linalg.norm(C[j]))
    return lmbda




def QP(pr):
    # Grab constraint matrices
    E, be, I, bi = pr.para.constraint.E, pr.para.constraint.be, pr.para.constraint.I, pr.para.constraint.bi
    x = pr.input
    n = len(x)
    # Grab quadratic matrix and linear term
    G = pr.para.parameter[0]
    c = pr.para.parameter[1]
    # y = initial(c, x, I, bi, E, be)
    if np.linalg.norm(G) == 0:
        # If linear, DONE
        if pr.para.pr == 0:
            print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
        return x
    if np.linalg.norm(E) == 0 and np.linalg.norm(I) == 0:
        # If unconstrained, take Newton step
        invG = np.linalg.inv(G)
        x = - np.matmul(invG, c)
        if pr.para.pr == 0:
            print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
        return x
    if np.linalg.norm(I) == 0:
        # If only equality constraints use CG method
        if not np.matmul(E, x) == be:
            return print('Invalid initial point')
        ET = np.transpose(E)
        r = np.matmul(G, x) + c
        v = np.linalg.solve(np.matmul(E, ET), np.matmul(E, r))
        g = r - np.matmul(ET, v)
        d = -g
        while np.dot(r, g) > 10 ** -6:
            alpha = np.dot(r, g) / np.dot(d, np.matmul(G, d))
            x = x + alpha * d
            rt = r + alpha * np.matmul(G, d)
            v = np.linalg.solve(np.matmul(E, ET), np.matmul(E, rt))
            gt = rt - np.matmul(ET, v)
            beta = np.dot(rt, g) / np.dot(r, g)
            d = -gt + beta * d
            g = gt
            r = rt
        if pr.para.pr == 0:
            print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
        return x
    if np.linalg.norm(I) > 0:
        # If there are inequality constraints, call active set method
        if np.linalg.norm(E) == 0:
            A = I
            b = bi
        else:
            A = np.block((E, I))
            b = np.concatenate((be, bi))
        W, index = work(A, b, x)
        k = 0
        while k < 10000:
            k += 1
            if k % pr.print == 0 and pr.para.pr == 0:
                print(str(k) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
            if len(W) > 0:
                WT = np.transpose(W)
                invKKT = schur(G, W, WT)
                g = np.matmul(G, x) + c
                indb = []
                for j in index:
                    indb.append(b[int(j)])
                indb = np.array(indb)
                h = np.matmul(W, x) - indb
                y = np.matmul(invKKT, np.concatenate((g, h)))
                pk = -y[:n]
                lbda = y[n:]
            else:
                invG = np.linalg.inv(G)
                sol = - np.matmul(invG, c)
                pk = sol - x
            if np.linalg.norm(pk) < 10 ** -12:
                if len(W) == 0:
                    if pr.para.pr == 0:
                        print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
                    return x
                mn = min(lbda)
                t = True
                if mn < 0:
                    t = False
                    for j in range(len(lbda)):
                        if lbda[j] == mn:
                            index = np.delete(index, j)
                            W = np.delete(W, j, axis=0)
                if t:
                    if pr.para.pr == 0:
                        print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
                    return x
            else:
                ak = [1]
                for j in range(len(A)):
                    if np.dot(A[j], pk) < 0 and j not in index:
                        ak.append((b[j] - np.dot(A[j], x)) / np.dot(A[j], pk))
                ak = min(ak)
                x = x + ak * pk
                if not ak == 1:
                    W, index = work(A, b, x)
        if pr.para.pr == 0:
            print('Failed to converge')
            print(str(x) + '___' + str(np.dot(x, np.matmul(G, x)) + np.dot(x, c)))
        return x
