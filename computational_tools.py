import scipy
import numpy as np
import copy

def product_gradient(params, H, ansatz, ref):
    grad = []
    ket = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        ket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ket)
    hbra = ket.T.dot(H)
    for i in range(0, len(ansatz)):
        grad.append(2*hbra.dot(ansatz[i]).dot(ket)[0,0].real)
        hbra = hbra = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], hbra.T).T
        ket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ket)

    return np.array(grad)

def product_energy(params, H, ansatz, ref):
    state = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
    return state.T.dot(H).dot(state)[0,0].real

def vqe(H, ansatz, ref, params):
    res = scipy.optimize.minimize(product_energy, np.array(params), jac = product_gradient, method = 'bfgs', args = (H, ansatz, ref), options = {'gtol': 1e-9})
    return res.fun, list(res.x)
        
