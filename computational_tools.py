import scipy
import numpy as np
import copy
from opt_einsum import contract

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

def sum_energy(params, H, ansatz, ref):
    state = copy.deepcopy(ref)
    gen = params[0]*ansatz[0]
    for i in range(1, len(params)):
        gen += params[i]*ansatz[i]
    state = scipy.sparse.linalg.expm_multiply(gen, state)
    return state.T.dot(H).dot(state)[0,0].real

def analytical_hess(H, ansatz, ref):
    hess = np.zeros((len(ansatz), len(ansatz)))
    for i in range(0, len(ansatz)):
        for j in range(0, len(ansatz)): 
            hess[i,j] = ref.T.dot(H).dot(ansatz[i].dot(ansatz[j])+ansatz[j].dot(ansatz[i])).dot(ref)[0,0].real - 2*ref.T.dot(ansatz[i]).dot(H).dot(ansatz[j]).dot(ref)[0,0].real
    return hess
    
def analytical_grad(H, ansatz, ref):
    grad = np.zeros((len(ansatz)))
    for i in range(0, len(ansatz)):
        grad[i] = 2*ref.T.dot(H).dot(ansatz[i]).dot(ref)[0,0].real
    return grad

def diagonal_hess(h, H, ansatz, ref):
    #Find the diagonal part of the Hessian one parameter from equilibrium w/ commutators
    hess = []
    for op in ansatz:
        ket = scipy.sparse.linalg.expm_multiply(h*op, ref)
        hess.append(2*ket.T.dot(H.dot(op)-op.dot(H)).dot(op.dot(ket))[0,0].real)
    return np.array(hess)

def diag_jerk(h, H, ansatz, ref):
    jerk = []
    plus2 = diagonal_hess(2*h, H, ansatz, ref)
    plus = diagonal_hess(h, H, ansatz, ref)
    minus = diagonal_hess(-h, H, ansatz, ref)
    minus2 = diagonal_hess(-2*h, H, ansatz, ref)
    jerk = (-plus2+8*plus-8*minus+minus2)/(12*h)
    en_jerk = np.zeros((len(ansatz), len(ansatz), len(ansatz)))
    for i in range(0, len(ansatz)):
        en_jerk[i,i,i] = jerk[i]
    return en_jerk

def F3(F, ansatz, ref):
    F3 = np.zeros((len(ansatz), len(ansatz), len(ansatz)))
    for i in range(0, len(ansatz)):
        op1 = ansatz[i]
        for j in range(i, len(ansatz)):
            op2 = ansatz[j]
            for k in range(j, len(ansatz)):
                op3 = ansatz[k]
                comm3 = 0
                comm3 -= ref.T.dot(op1.dot(F).dot(op2).dot(op3).dot(ref))[0,0].real
                comm3 -= ref.T.dot(op1.dot(F).dot(op3).dot(op2).dot(ref))[0,0].real
                comm3 -= ref.T.dot(op2.dot(F).dot(op1).dot(op3).dot(ref))[0,0].real
                comm3 -= ref.T.dot(op2.dot(F).dot(op3).dot(op1).dot(ref))[0,0].real
                comm3 -= ref.T.dot(op3.dot(F).dot(op2).dot(op1).dot(ref))[0,0].real
                comm3 -= ref.T.dot(op3.dot(F).dot(op1).dot(op2).dot(ref))[0,0].real
                F3[i,j,k] = F3[j,i,k] = F3[k,j,i] = F3[j,k,i] = F3[i,k,j] = F3[k,i,j] = copy.copy(comm3)
    return F3

def deriv(params, H, ansatz, ref):
    deriv = []
    h = 1e-4
    for i in range(0, len(params)):
        forw = copy.copy(params)
        forw[i] += h
        forw2 = copy.copy(params)
        forw2[i] += 2*h
        back = copy.copy(params)
        back[i] -= h
        back2 = copy.copy(params)
        back2[i] -= 2*h
        plus2 = sum_energy(forw2, H, ansatz, ref)
        plus = sum_energy(forw, H, ansatz, ref)
        minus = sum_energy(back, H, ansatz, ref)
        minus2 = sum_energy(back2, H, ansatz, ref)
        deriv.append((-plus2+8*plus-8*minus+minus2)/(12*h))
    return np.array(deriv)

def hess(params, H, ansatz, ref):
    hess = []
    h = 1e-4
    for i in range(0, len(params)):
        forw = copy.copy(params)
        forw[i] += h
        forw2 = copy.copy(params)
        forw2[i] += 2*h
        back = copy.copy(params)
        back[i] -= h
        back2 = copy.copy(params)
        back2[i] -= 2*h
        plus2 = deriv(forw2, H, ansatz, ref)
        plus = deriv(forw, H, ansatz, ref)
        minus = deriv(back, H, ansatz, ref)
        minus2 = deriv(back2, H, ansatz, ref)
        hess.append((-plus2+8*plus-8*minus+minus2)/(12*h))
    return np.array(hess)

def jerk(params, H, ansatz, ref):
    jerk = []
    h = 1e-3
    for i in range(0, len(params)):
        forw = copy.copy(params)
        forw[i] += h
        forw2 = copy.copy(params)
        forw2[i] += 2*h
        back = copy.copy(params)
        back[i] -= h
        back2 = copy.copy(params)
        back2[i] -= 2*h
        plus2 = hess(forw2, H, ansatz, ref)
        plus = hess(forw, H, ansatz, ref)
        minus = hess(back, H, ansatz, ref)
        minus2 = hess(back2, H, ansatz, ref)
        jerk.append((-plus2+8*plus-8*minus+minus2)/(12*h))
    return np.array(jerk)

def UCC2_energy(x, E0, deriv, hess):
    return E0 + deriv.dot(x) + .5*x.T.dot(hess).dot(x)

def UCC3_energy(x, E0, deriv, hess, jerk):
    return E0 + deriv.dot(x) + .5*x.T.dot(hess).dot(x) + (1/6)*contract('ijk,i,j,k->', jerk, x, x, x) 

def EN_UCC3_energy(x, E0, deriv, hess, jerk):
    return E0 + deriv.dot(x) + .5*x.T.dot(hess).dot(x) + (1/6)*contract('iii,i,i,i->', jerk, x, x, x) 

def vqe(H, ansatz, ref, params, gtol = 1e-6):
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

