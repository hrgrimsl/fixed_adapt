import scipy
import numpy as np
import copy

def product_gradient(params, H, ansatz, ref):
    grad = []
    ket = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        ket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ket)
    hbra = ket.T.dot(H)
    for i in range(0, len(ansatz)):
        grad.append(2*hbra.dot(ansatz[i]).dot(ket)[0,0].real)
        hbra = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], hbra.T).T
        ket = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], ket)
    return np.array(grad)

def prep_state(ops, ref, params):
    state = copy.copy(ref)
    for i in reversed(range(0, len(ops))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ops[i], state)
    return state

def product_energy(params, H, ansatz, ref):
    state = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
    return state.T.dot(H).dot(state)[0,0].real

def vqe(H, ansatz, ref, params, gtol = 1e-10):
    res = scipy.optimize.minimize(product_energy, np.array(params), jac = product_gradient, method = 'bfgs', args = (H, ansatz, ref), options = {'gtol': gtol})
    return res.fun, list(res.x)

def qse(K, L, H, S2, Sz, N):
    ops = [L, H, S2, Sz, N]
    qse_ops = [np.zeros((len(K), len(K))) for i in range(0, len(ops))]
    for idx in range(0, len(ops)):
        for i in range(0, len(K)):
            for j in range(i, len(K)):
                qse_ops[idx][i,j] = qse_ops[idx][j,i] = K[i].T.dot(ops[idx]).dot(K[j])[0,0] 
    #Note that we diagonalize L, not H
    v = np.linalg.eigh(qse_ops[1])[1][:, 0]
    evs = [v.T.dot(op).dot(v) for op in qse_ops]+[v]
    return evs 

def no_qse(K, L, H, S2, Sz, N):
    S = np.zeros((len(K), len(K)))
    for i in range(0, len(K)):
        for j in range(i, len(K)):
            S[i,j] = S[j,i] = K[i].T.dot(K[j])[0,0]
    ops = [L, H, S2, Sz, N]
    qse_ops = [np.zeros((len(K), len(K))) for i in range(0, len(ops))]
    for idx in range(0, len(ops)):
        for i in range(0, len(K)):
            for j in range(i, len(K)):
                qse_ops[idx][i,j] = qse_ops[idx][j,i] = K[i].T.dot(ops[idx]).dot(K[j])[0,0] 
   
    Lval, v = symmetric(S, qse_ops[0])
    evs = [v.T.dot(op).dot(v) for op in qse_ops]+[v]
    return evs 

def canonical(S, L):
    #Do canonical orthogonalization
    Leff = np.linalg.inv(S).dot(L)
    Ls, vs = np.linalg.eig(Leff)
    idx = Ls.argsort()
    return Ls[idx][0], vs[:,idx][:,0]

def symmetric(S, L):
    S2inv = rt_inv(S)
    Leff = S2inv.T.dot(L).dot(S2inv)
    Ls, vs = np.linalg.eigh(Leff)
    return Ls[0], S2inv.dot(vs[:,0])

def rt_inv(S):
    #builds S^{-1/2} exactly
    s, v = np.linalg.eigh(S)
    s2 = []
    for i in s:
        if abs(i) > 1e-10:
            s2.append(1/np.sqrt(i))
        else:
            s2.append(0)
    s2inv = np.array(s2)
    s2inv = np.diag(s2inv)
    S2inv = v.dot(s2inv).dot(v.T)
    return S2inv

def rt(S):
    #builds S^{1/2} exactly
    s, v = np.linalg.eigh(S)
    s2inv = np.array([np.sqrt(i) for i in s])
    s2inv = np.diag(s2inv)
    S2inv = v.T.dot(s2inv).dot(v)
    return S2inv

