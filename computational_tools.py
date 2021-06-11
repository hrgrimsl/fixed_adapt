import scipy
import numpy as np
import copy

import time
import os
from opt_einsum import contract

from multiprocessing import Pool
from num2words import num2words



def oo_vqe(H, ansatz, singles, ref, params, knorm = 1e-8, tnorm = 1e-8, Etol = 1e-12):

    #Could be generalized so that no singles corresponds to normal VQE, no doubles is UCCSD?
    times = [time.time()]
    print("\n Orbital-optimized VQE:")
    H_eff = copy.copy(H) #H transformed by the orbital optimization
    psi = copy.copy(ref) #|0> transformed by the actual unitary in the VQE
    Done = False

    E_history = []
    E_history.append(simple_energy(params, H_eff, ansatz, ref))
    k_history = [] #list of previous orbital optimization params

    k_history.append(np.zeros(len(singles)))
    t_history = []
    t_history.append(params)
    iters = 0
    times.append(time.time())
    print(f"Initial energy: {E_history[-1]}")
    while Done == False:
        iters += 1
        print(f"\n Iteration {iters}:")
        print("Performing VQE with current orbitals:")

        E_vqe, t, g, E0, Na = simple_vqe(H_eff, ansatz, ref, t_history[-1])
        print(f"Energy:   {E_vqe}")
        print(f"dE:       {E_vqe-E_history[-1]}")
        print(f"dt norm:  {np.linalg.norm(t-t_history[-1])}")
        print(f"gnorm:    {np.linalg.norm(g)}")
        t_history.append(t)
        psi = copy.deepcopy(ref)
        for i in reversed(range(0, len(ansatz))):
            psi = scipy.sparse.linalg.expm_multiply(ansatz[i]*t[i], psi)
        print(f"Reoptimizing Orbitals")
        E_oo, k, g_oo, E0 = simple_uccsd(H, singles, psi, k_history[-1])
        print(f"Energy: {E_oo}")
        print(f"dE:       {E_oo - E_vqe}")
        print(f"dk norm:  {np.linalg.norm(k-k_history[-1])}")
        print(f"gnorm:    {np.linalg.norm(g_oo)}")
        E_history.append(E_oo)
        k_history.append(k)
        gen = 0*ansatz[0]
        for i in range(0, len(singles)):
            gen += singles[i]*k[i]
        U = scipy.sparse.linalg.expm(gen)
        H_eff = U.T@(H)@U
        if np.linalg.norm(k-k_history[-2]) < knorm and np.linalg.norm(t-t_history[-2]) < tnorm and np.linalg.norm(E_oo - E_history[-2]) < Etol:
            print("Converged.")
            Done = True
        times.append(time.time())
        print(f"Iteration completed in {times[-1]-times[-2]} seconds.")
    print(f"\nOO-VQE completed in {times[-1]-times[0]} seconds.\n")
    return E_history[-1], list(t_history[-1]), g, E_history[0], U
         
        
    
    

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

def ML_M(params, H, ansatz, ref):
    N = len(ansatz)
    M = np.zeros((N, N))
    V = np.zeros((N))        
    _ket = copy.copy(ref)
    vec = []
    for u in reversed(range(0, N)):
        _ket = scipy.sparse.linalg.expm_multiply(params[u]*ansatz[u], _ket)
        uket = ansatz[u]@_ket
        vec = [(_ket.T@uket).todense()[0,0]] + vec
        _uket = copy.copy(uket)
        _vket = copy.copy(_ket)
        for v in reversed(range(0, u+1)):
            targ = scipy.sparse.hstack([_uket, _vket])
            res = scipy.sparse.linalg.expm_multiply(params[v]*ansatz[v], targ).tocsr() 
            _uket = res[:,0]
            _vket = res[:,1]
            M[u,v] = M[v,u] = -2*(_vket.T@(ansatz[v]@_uket)).todense()[0,0]             
    for u in reversed(range(0, N)):
        for v in reversed(range(0, u+1)):
            M[u,v] += 2*vec[u]*vec[v]
            if u != v:
                M[v,u] += 2*vec[u]*vec[v]
    return M
    
def ML_M_dumb(params, H, ansatz, ref):       
    #Can this be made faster by just storing things?
    N = len(ansatz)
    M = np.zeros((N, N))
    
    state = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
     
    term2s = []    
    for b in range(0, M.shape[0]):
        ebket = copy.copy(ref)  
        for i in reversed(range(b, len(ansatz))):
            ebket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ebket)
        fket = copy.copy(ebket)
        ebket = ansatz[b]@ebket
        term2s.append((ebket.T@ebket).todense()[0,0])
        for i in reversed(range(0, b)):
            ebket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ebket)
        eket = copy.copy(state)

        for a in range(0, b+1):
            M[a,b] = M[b,a] = -2*(eket.T@ansatz[a]@ebket).todense()[0,0]
            targ = scipy.sparse.hstack([ebket, eket])
            res = scipy.sparse.linalg.expm_multiply(-params[a]*ansatz[a], targ).tocsr()
            ebket = res[:,0]
            eket = res[:,1]
    term2s = np.array(term2s)
    M += 2*np.outer(term2s, term2s)
    return M

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

def simple_energy(params, H, ansatz, ref):
    state = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
    E = (state.T@(H)@state).todense()[0,0]
    return E

def simple_uccsd_energy(params, H, ansatz, ref):
    gen = params[0]*ansatz[0]
    for i in range(1, len(ansatz)):
        gen += params[i]*ansatz[i]
    state = scipy.sparse.linalg.expm_multiply(gen, ref)
    E = (state.T@(H)@state).todense()[0,0]
    return E

def simple_gradient(params, H, ansatz, ref):
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

    return np.array(grad)

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




def brute_force_energy(param, x0, grad, H, ansatz, ref):
    return product_energy(param[0]*grad + x0, H, ansatz, ref)

def ML_X(params, H, ansatz, ref):
    X = np.zeros((H.shape[0], len(params)))
    mat = scipy.sparse.identity(H.shape[0])
    vec = copy.copy(ref)

    for i in reversed(range(0, len(ansatz))):
        vec = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], vec)
        ket = copy.copy(vec)

    for i in range(0, len(ansatz)):

        vec = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], vec)

        mat = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], mat.T).T
        X[:,i] = (mat@ansatz[i]@vec).todense().reshape((H.shape[0]))

    return X, ket

    

def VQITE(H, ansatz, ref, params, dt = 1e-1, tol = 1e-10):
    print("Doing a McLachlan Imaginary Time Evolution")
    #V = -product_gradient(params, H, ansatz, ref)
    iter = 0
    old_E = product_energy(params, H, ansatz, ref)
    V = None
    hf = copy.copy(old_E)
    E = copy.copy(old_E)
    dE = 0
    print(f"Iter.       |Energy              |dE                  |Grad Norm")
    while (V is None or np.linalg.norm(V) > tol): 
        V = -product_gradient(params, H, ansatz, ref)
        print(f"{iter:12d}|{E:20.16f}|{dE:20.8e}|{np.linalg.norm(V):20.8e}")

        if np.linalg.norm(V) < tol:
            break
        iter += 1
        M = ML_M(params, H, ansatz, ref)
        #try:
        #    direction = np.linalg.inv(M)@V
        #except:
        #    print("Singular matrix M, using pinv")
        #    direction = np.linalg.pinv(M, rcond=1e-12)@V
        direction = np.linalg.pinv(M, rcond = 1e-10)@V
        old_params = copy.copy(params)
        params += dt*direction
        E = product_energy(params, H, ansatz, ref)
        dE = E - old_E
        dt_cur = copy.copy(dt)        
        old_E = copy.copy(E)

    return E, list(params)
         

   
def one_param_energy(param, H, ansatz, ref, param_no, params):

    state = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        if i != param_no:
            state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
        else:
            state = scipy.sparse.linalg.expm_multiply(param[0]*ansatz[i], state)
    E = (state.T@(H)@state).todense()[0,0]

    return E

def one_param_grad(param, H, ansatz, ref, param_no, params):
    state = copy.deepcopy(ref)
    istate = copy.deepcopy(ref)
    for i in reversed(range(0, len(ansatz))):
        if i != param_no:
            istate = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
            state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
        else:
            istate = scipy.sparse.linalg.expm_multiply(param[0]*ansatz[i], state)
            istate = ansatz[i]@istate
            state = scipy.sparse.linalg.expm_multiply(param[0]*ansatz[i], state)
    g = 2*(state.T@(H)@istate).todense()[0,0]
    return np.array([g])



def best_vqe(results, dump_dir = 'dump'):
    Es = [i[0] for i in results]
    xs = [i[1] for i in results]
    gs = [i[2] for i in results]
    E0s = [i[3] for i in results]

    recycled = f"{len(xs[0])} {Es[0]} {np.linalg.norm(np.array(gs[0]))} {E0s[0]}"       
    dump = f"./{dump_dir}/converged_recycled_{len(xs[0])}"
    np.save(dump, np.array(xs[0]))
    print(f"Random Initializations ({dump_dir}):")   
    print("Seed | Operators | Energy | Gradient | Initial E | Recycled Data")
    for e in range(1, len(Es)):
        print(f"{e-1} {len(xs[0])} {Es[e]} {np.linalg.norm(np.array(gs[e]))} {E0s[e]} {recycled}")
        dump = f"./{dump_dir}/converged_seed_{e}_{len(xs[0])}"
        np.save(dump, np.array(xs[e]))

    idx = np.argsort(Es)
    print(f"Summary of Multi-initialized VQE with {len(xs[0])} operators:")
    print(f"Recycled parameters give the {num2words(list(idx).index(0)+1, to='ordinal_num')} best energy.")
    print(f"This energy is worse than the best one by:  {Es[idx[0]]- Es[0]}")
    print(f"Energy min: {Es[idx[0]]}")
    print(f"Energy max: {Es[idx[-1]]}")
    print(f"Energy mean:  {np.mean(np.array(Es))}") 
    print(f"Energy StdDev:  {np.std(np.array(Es))}")     
    return Es[idx[0]], list(xs[idx[0]]), np.array(gs[idx[0]]), E0s[idx[0]], None

def pre_sample(H, ansatz, ref, params, seeds = 10000, norm = 1):
    Es = []
    new_params = []

    print("Pre-sampling energies about the given parameters:") 
    for i in range(0, seeds):

        guess_vec = list(params + 20*np.random.random_sample(len(ansatz))-10)
        new_params.append(guess_vec)
        E = simple_energy(guess_vec, H, ansatz, ref)
        Es.append(E)
        print(f"Seed {i}: {E}")
    idx = np.argsort(np.array(Es))
    return [Es[i] for i in idx][:125], [new_params[i] for i in idx][:125]

def sample_uccsd(H, ansatz, ref, params, gtol = 1e-16, seeds = 125, dump_dir = 'dump_entangled'):
    print("UCCSD ansatz sampling")
    print("Recycled parameters:")
    print(params)
    if os.path.exists(dump_dir):
        os.system(f'rm -r {dump_dir}')
    os.makedirs(dump_dir)
    start = time.time()
    Es = []
    gnorms = []
    opt_params = []
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    param_tuple = [list(params)]
    dump = f"./{dump_dir}/recycled_{len(ansatz)}"
    np.save(dump, np.array(params))
    hf = np.zeros(len(ansatz))
    #pre_Es, new_params = pre_sample(H, ansatz, ref, hf)
    #param_tuple += new_params
    for i in range(0, 125):
        np.random.seed(i)
        guess_vec = list(2*np.random.random_sample(len(ansatz))-1)
        param_tuple.append(guess_vec)
        #guess_vec = param_tuple[i+1]
        dump = f"./{dump_dir}/guess_{i}_{len(ansatz)}"
        np.save(dump, np.array(guess_vec))
    with Pool(127) as p:
        L = p.starmap(simple_uccsd, iterable = [*zip([H for i in range(0, 126)], [ansatz for i in range(0, 126)], [ref for i in range(0, 126)], param_tuple)])
    assert(len(param_tuple) == len(L))
    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    best = best_vqe(L, dump_dir = dump_dir)
    best_params = best[1]
    print("Optimized parameters:")
    print(best_params)
    return best[0], best[1]

def sample_vqe(H, ansatz, ref, params, gtol = 1e-16, seeds = 125, dump_dir = 'dump_disentangled'):
    if os.path.exists(dump_dir):
        os.system(f'rm -r {dump_dir}')
    os.makedirs(dump_dir)
    start = time.time()
    Es = []
    gnorms = []
    opt_params = []
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    param_tuple = [list(copy.copy(params))]
    dump = f"./{dump_dir}/recycled_{len(ansatz)}"
    np.save(dump, np.array(params))
    hf = np.zeros(len(ansatz))
    #pre_Es, new_params = pre_sample(H, ansatz, ref, hf)
    #param_tuple += new_params
    for i in range(0, 125):
        np.random.seed(i)
        guess_vec = list(2*np.random.random_sample(len(ansatz))-1)
        param_tuple.append(guess_vec)
        #guess_vec = param_tuple[i+1]
        dump = f"./{dump_dir}/guess_{i}_{len(ansatz)}"
        np.save(dump, np.array(guess_vec))
    with Pool(127) as p:
        L = p.starmap(simple_vqe, iterable = [*zip([H for i in range(0, 126)], [ansatz for i in range(0, 126)], [ref for i in range(0, 126)], param_tuple)])
    assert(len(param_tuple) == len(L))

    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    return best_vqe(L, dump_dir = dump_dir)

def simple_vqe(H, ansatz, ref, params, gtol = 1e-16):
    E0 = simple_energy(params, H, ansatz, ref) 
    res = scipy.optimize.minimize(simple_energy, np.array(params), jac = simple_gradient, method = 'bfgs', args = (H, ansatz, ref), options = {'gtol': gtol})
    grad = simple_gradient(res.x, H, ansatz, ref)
    return res.fun, res.x, grad, E0, None

def simple_uccsd(H, ansatz, ref, params, gtol = 1e-16):
    E0 = simple_energy(params, H, ansatz, ref) 
    res = scipy.optimize.minimize(simple_uccsd_energy, np.array(params), jac = None, method = 'bfgs', args = (H, ansatz, ref), options = {'gtol': gtol})
    grad = res.jac
    return res.fun, res.x, grad, E0

def vqe(H, ansatz, ref, params, gtol = 1e-16, singles = []):
    with Pool(1) as p:
        if len(singles) == 0:
            L = p.starmap(simple_vqe, iterable = [*zip([H], [ansatz], [ref], list([params]))])
        else:
            return oo_vqe(H, ansatz, singles, ref, params)
            
    return best_vqe(L)


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
    try:
        print(f"VQE Iter.:              {iters}")
        print(f"Energy:                 {E:20.16f}")
        print(f"Norm of dx:             {np.linalg.norm(params-xk):20.16e}")
        print(f"Norm of grad.:          {np.linalg.norm(grad):20.16e}")
    except:
        pass
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

