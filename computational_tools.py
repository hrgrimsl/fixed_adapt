import scipy
import numpy as np
import copy
from opt_einsum import contract




def product_hessian(params, H, ansatz, ref):
    #Can this be made faster by just storing things?
    global vqe_cache
    N = len(ansatz)
    hessian = np.zeros((N, N))
    
    if np.linalg.norm(params) == 0:
        print("Oh boy, a zero vector.  Time for SHUCC!")
        for b in range(0, N):
            for a in range(0, b+1):
                hessian[a,b] = hessian[b,a] = 2*(ref.T@H@ansatz[a]@ansatz[b]@ref).todense()[0,0] - 2*(ref.T@ansatz[a]@H@ansatz[b]@ref).todense()[0,0]
        try:
            vqe_cache[tuple(params)][2] = hessian
        except:
            pass
        return hessian

    state = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
    hket = H@state

    #hket = H...|0>
    #state = ...|0>

    #ebket = ...B...|0> - starts         ...B...|0>      
    #ehket = ...H...|0> - starts         H...|0>

    #eket = ...|0> - starts as           ...|0>
    #ehbket = ...H...B...|0> starts as   H...B...|0>
   
    for b in range(0, hessian.shape[0]):
        ebket = copy.copy(ref)  
        for i in reversed(range(b, len(ansatz))):
            ebket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ebket)
        ebket = ansatz[b]@ebket
        for i in reversed(range(0, b)):
            ebket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ebket)
        ehket = copy.copy(hket)
        eket = copy.copy(state)
        ehbket = H@ebket
                
        for a in range(0, b+1):
            hessian[a,b] = hessian[b,a] = 2*(ehket.T@ansatz[a]@ebket).todense()[0,0] - 2*(eket.T@ansatz[a]@ehbket).todense()[0,0]
            targ = scipy.sparse.hstack([ebket, ehket, eket, ehbket])
            res = scipy.sparse.linalg.expm_multiply(-params[a]*ansatz[a], targ).tocsr()
            ebket = res[:,0]
            ehket = res[:,1]
            eket = res[:,2]
            ehbket = res[:,3]
    try:
        vqe_cache[tuple(params)][2] = hessian
    except:
        pass
    
    return hessian
         

def product_gradient(params, H, ansatz, ref):
    grad = []
    ket = copy.copy(ref)

    for i in reversed(range(0, len(ansatz))):
        ket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ket)

    hbra = (H@ket).T
    for i in range(0, len(ansatz)):
        grad.append(2*(hbra@ansatz[i]@ket).todense()[0,0])
        targ = scipy.sparse.hstack([hbra.T, ket])
        res = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], targ).tocsr()
        hbra = res[:,0].T
        ket = res[:,1]
        #hbra = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], hbra.T).T
        #ket = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], ket)

    global vqe_cache
    try:
        vqe_cache[tuple(params)][1] = np.array(grad)
    except:
        pass
    return np.array(grad)

def prep_state(ops, ref, params):
    state = copy.copy(ref)
    for i in reversed(range(0, len(ops))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ops[i], state)
    return state

def product_energy(params, H, ansatz, ref):
    np.set_printoptions(precision = 16)
    #print(params)
    state = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
    
    E = (state.T@(H)@state).todense()[0,0]
    global vqe_cache
    try:
        vqe_cache[tuple(params)][0] = E
    except:
        try:
            vqe_cache[tuple(params)] = [E, None, None]
        except:
            pass
    #print(E)
    return E

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

def vqe(H, ansatz, ref, params, gtol = 1e-10):
    print("Doing VQE...")
    global iters 
    iters= 1
    global xk 
    xk = copy.copy(params)
    #Stores data as params: [E, grad, hess]
    global vqe_cache
    vqe_cache = {}
    vqe_cache['tuple(params)'] = [None, None, None]

    res = scipy.optimize.minimize(product_energy, np.array(params), jac = product_gradient, hess = product_hessian, method = 'Newton-CG', callback = prod_cb, args = (H, ansatz, ref), options = {'xtol': gtol, 'disp': True, 'verbose': 4})
    assert res.success == True
    return res.fun, list(res.x)

def resid(x, H, ansatz, ref):
    grad = product_gradient(x, H, ansatz, ref)
    return grad

def diis(H, ansatz, ref, guess, gtol = 1e-10, max_vec = 1000):
    print("Doing VQE via DIIS")
    print("Guess:")
    iters = 1
    ps = [copy.copy(guess)]    
    es = [resid(ps[-1], H, ansatz, ref)]     
    E = product_energy(ps[-1], H, ansatz, ref)
    print(f"Residual norm:  {np.linalg.norm(es[-1]):20.16e}")  
    print(f"Energy:         {E:20.16f}")
    print("Doing a Newton step:")
    for i in range(0, 1):
        hinv = np.linalg.pinv(product_hessian(ps[-1], H, ansatz, ref), rcond = 1e-10)    
        ps.append(ps[-1]-hinv@es[-1]) 
        es.append(resid(ps[-1], H, ansatz, ref))
        E = product_energy(ps[-1], H, ansatz, ref)
        print(f"Residual norm:  {np.linalg.norm(es[-1]):20.16e}")  
        print(f"Energy:         {E:20.16f}")

    print("Starting DIIS.")
    while np.linalg.norm(es[-1]) > gtol:
        if iters % 20 == 0:
            print("Restarting DIIS.")
            ps = [ps[-1]]
            es = [resid(ps[-1], H, ansatz, ref)]
            ps.append(es[-1])
            es.append(resid(ps[-1], H, ansatz, ref))
        B = np.ones((len(es)+1, len(es)+1))
        B[-1,-1] = 0
        for i in range(0, len(es)):
            for j in range(i, len(es)):
                B[i,j] = B[j,i] = es[i]@es[j]

        stationary = np.zeros(len(es)+1)
        stationary[-1] = 1
        C = np.linalg.inv(B)@stationary
        new_p = contract('rc,r', np.array(ps), C[:-1]) 
        ps.append(copy.copy(new_p)) 
        es.append(resid(ps[-1], H, ansatz, ref))
        if len(es) > max_vec:
            ps = ps[1:]
            es = es[1:]
        print(f"Iteration:      {iters}")
        print(f"Residual norm:  {np.linalg.norm(es[-1]):20.16e}")  
        E = product_energy(ps[-1], H, ansatz, ref)
        print(f"Energy:         {E:20.16f}")
        iters += 1
    return E, ps[-1]

def prod_cb(params):
    global vqe_cache
    global iters
    global xk
    E = vqe_cache[tuple(params)][0]
    grad = vqe_cache[tuple(params)][1]
    hess = vqe_cache[tuple(params)][2]
    print(f"VQE Iter.:              {iters}")
    print(f"Energy:                 {E:20.16f}")
    print(f"Norm of dx:             {np.linalg.norm(params-xk):20.16e}")
    print(f"Norm of grad.:          {np.linalg.norm(grad):20.16e}")
    if hess is not None:
        print(f"Cond. no. of Hessian:   {np.linalg.cond(hess):20.16e}")
    iters += 1
    xk = copy.copy(params)
    

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

