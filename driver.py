import system_methods as sm
import computational_tools as ct
import openfermion as of
import numpy as np
import scipy
import copy
import os
import time
import math

#Globals
Eh = 627.5094740631

class Xiphos:
    """Class representing an individual XIPHOS calculation"""
    def __init__(self, H, ref, system, pool, v_pool, H_adapt = None, H_vqe = None, verbose = "INFO", sym_ops = None, sym_break = None):

        """Initialize a XIPHOS Solver Object.

        :param H: Molecular Hamiltonian.
        :type H: (2^N,2^N) array-like object
        :param ref: Reference wavefunction.
        :type ref: (2^N,1) array-like object
        :param system: Prescribes a working directory name to dump stuff like .npy files in.
        :type system: str        
        :param pool: Pool of operators.
        :type: list
        :param v_pool: List of strings associated with operator pool.
        :type: list

        :param H_adapt: Separate Hamiltonian to use during operator additions.  Defaults to H.
        :type H_adapt: (2^N,2^N) array-like object, optional
        :param H_vqe: Separate Hamiltonian to use during parameter minimization.  Defaults to H.
        :type H_vqe: (2^N,2^N) array-like object, optional
        :param verbose: Sets the logger level.  Defaults to "INFO". 
        :type verbose: str, optional
        :param sym_ops: Dictionary of operators of interest to track. (H, S_z, S^2, N, etc.).  Defaults to {'H': H}. 
        :type verbose: list, optional
        :param sym_break: Dictionary of perturbations to use to break symmetry in the CI wavefunction.  Defaults to None.
        :type sym_break: Array-like, optional
       
        :return: None
        :rtype: NoneType    
        """
          
        self.H = H
        self.ref = ref
        self.system = system
        self.pool = pool
        self.v_pool = v_pool
        self.vqe_iteration = 0
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
 
        if H_adapt is None:
            self.H_adapt = self.H
        else:
            self.H_adapt = H_adapt

        if H_vqe is None:
            self.H_vqe = self.H
        else:
            self.H_vqe = H_vqe

      
        if sym_ops is None:
            self.sym_ops = {'H': H}
        else:
            self.sym_ops = sym_ops


        if os.path.isdir(system):
            self.restart = True
        else:
            os.mkdir(system)
            self.restart = False

        self.log_file = f"{system}/log.dat"

        if self.restart == False:
            print("Starting a new calculation.\n")
        else:  
            print("Restarting the calculation.\n")
        print("---------------------------")

         
        #We do an exact diagonalization to check for degeneracies/symmetry issues
        print("\nReference information:")
        self.ref_syms = {}
        for key in self.sym_ops.keys():
            val = (ref.T@(self.sym_ops[key]@ref))[0,0]
            print(f"{key: >6}: {val:20.16f}")
            self.ref_syms[key] = val
        self.hf = self.ref_syms['H']        
        print("\nED information:") 
        w, v = scipy.sparse.linalg.eigsh(H, k = min(H.shape[0]-1,10), which = "SA")
        self.ed_energies = w
        self.ed_wfns = v
        self.ed_syms = []
        for i in range(0, len(w)):
            print(f"ED Solution {i+1}:")
            ed_dict = {}
            for key in self.sym_ops.keys():
                val = (v[:,i].T@(self.sym_ops[key]@v[:,i]))
                print(f"{key: >6}: {val:20.16f}")
                ed_dict[key] = copy.copy(val)
            self.ed_syms.append(copy.copy(ed_dict))

        for key in self.sym_ops.keys():
            if key != "H" and abs(self.ed_syms[0][key] - self.ref_syms[key]) > 1e-8:
                print(f"\nWARNING: <{key}> symmetry of reference inconsistent with ED solution.")
        if abs(self.ed_syms[0]["H"] - self.ed_syms[1]["H"]) < (1/Eh):
            print(f"\nWARNING:  Lowest two ED solutions may be quasi-degenerate.")

    def rebuild_ansatz(self, A):
        params = []
        ansatz = [] 
        #A is the number of operators in your ansatz that you know.
        os.system(f"grep -A{A} ansatz {self.log_file} > {self.system}/temp.dat")
        os.system(f"tail -n {A} {self.system}/temp.dat > {self.system}/temp2.dat")
        f = open(f"{self.system}/temp2.dat", "r")
        ansatz = []
        params = []
        for line in f.readlines():
            line = line.split()
            param = line[1]
            if len(line) == 5:
                op = line[2] + " " + line[3] + " " + line[4]
            else:
                op = line[2]
            #May or may not rebuild in correct order - double check
            ansatz = ansatz + [self.v_pool.index(op)]
            params = params + [float(param)]
        return params, ansatz
    
    def ucc_E(self, params, ansatz):
        G = params[0]*self.pool[ansatz[0]]
        for i in range(1, len(ansatz)):
            G += params[i]*self.pool[ansatz[i]] 
        state = scipy.sparse.linalg.expm_multiply(G, self.ref)
        E = ((state.T)@self.H@state).todense()[0,0]
        return E

    def comm(self, A, B):
        return A@B - B@A

    def ucc_grad_zero(self, ansatz):
        grad = []
        for i in range(0, len(ansatz)):
            g = ((self.ref.T)@(self.comm(self.H, self.pool[ansatz[i]]))@self.ref).todense()[0,0]
            grad.append(g)
        return np.array(grad)
   
    def ucc_hess_zero(self, ansatz):
        hess = np.zeros((len(ansatz), len(ansatz)))
        for i in range(0, len(ansatz)):
            for j in range(0, len(ansatz)):
                    hess[i,j] += .5*((self.ref.T)@(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[j]])@self.ref)).todense()[0,0]
                    hess[j,i] += .5*((self.ref.T)@(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[j]])@self.ref)).todense()[0,0]
        return hess
                 
    def ucc_diag_jerk_zero(self, ansatz):
        jerk = []
        for i in range(0, len(ansatz)):
            j = ((self.ref.T)@(self.comm(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[i]]), self.pool[ansatz[i]]))@self.ref).todense()[0,0]
            jerk.append(j)
        jmat = np.zeros((len(ansatz), len(ansatz), len(ansatz)))
        for i in range(0, len(jerk)):
            jmat[i,i,i] = jerk[i]
        return jmat

    def ucc_inf_d_E(self, params, ansatz, E0, grad, hess):
        E = E0 + grad.T@params + .5*params.T@hess@params
        for i in range(0, len(ansatz)):
            E += self.ucc_E(np.array([params[i]]), [ansatz[i]])
            E -= params[i]*grad[i]
            E -= .5*params[i]*params[i]*hess[i,i] 
            E -= E0
        return E
    
    def tucc_inf_d_E(self, params, ansatz, E0, grad, hess):
        E = E0 + grad.T@params + .5*params.T@hess@params
        for i in range(0, len(ansatz)):
            E += self.t_ucc_E(np.array([params[i]]), [ansatz[i]])
            E -= params[i]*grad[i]
            E -= .5*params[i]*params[i]*hess[i,i] 
            E -= E0

        return E[0,0]


    
    def H_eff_analysis(self, params, ansatz):
        H_eff = copy.copy(self.H)
        for i in reversed(range(0, len(params))):
            U = scipy.sparse.linalg.expm(params[i]*self.pool[ansatz[i]])
            H_eff = ((U.T)@H@U).todense()
        E = ((self.ref.T)@H_eff@self.ref)
        print("Analysis of H_eff:")
        print(f"<0|H_eff|0> = {E}")
        w, v = np.linalg.eigh(H_eff)
        for sv in w:
            spec_string += f"{sv},"
        print(f"Eigenvalues of H_eff:")
        print(spec_string)
        

    





    def aegis(self, params, ansatz, refs, deg_thresh = 1e-8, gtol = None, Etol = None, max_depth = None, criteria = 'grad'):
        """Aces and Eights Generation of Interpolation Subspace  
        :param params: Parameters associated with ansatz.
        :type params: list 
        :param ansatz: List of operator indices, applied to reference in reversed order.
        :type ansatz: list
        :param ref: Reference state.
        :type ref: (2^N,1) array-like
        :param deg_thresh: What are we calling a 'degenerate' gradient?
        :type gtol: float
        :param gtol: Stopping condition on gradient norm of all ops to add
        :type gtol: float
        :param Etol: Stopping condition on error from ED of all ops
        :type Etol: float
        :param max_depth: Stopping condition on operators
        :type max_depth: int
        """
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
        self.e_savings = 0
        self.grad_savings = 0
        self.state_savings = 0
        states = refs       
        while Done == False:
            gnorms = []
            for i in range(0, len(states)):
                state = states[i]
                print(f"Considering state {i}/{len(states)}")
                gradient = 2*np.array([((state.T@(self.H_adapt@(op@state)))[0,0]) for op in self.pool])
                gnorms.append(np.linalg.norm(gradient))
                idx = np.argsort(abs(gradient))
                 
        
    def adapt(self, params, ansatz, ref, gtol = None, Etol = None, max_depth = None, criteria = 'grad', multiple = False):
        """Vanilla ADAPT algorithm for arbitrary reference.  No sampling, no tricks, no silliness.  
        :param params: Parameters associated with ansatz.
        :type params: list 
        :param ansatz: List of operator indices, applied to reference in reversed order.
        :type ansatz: list
        :param ref: Reference state.
        :type ref: (2^N,1) array-like
        :param gtol: Stopping condition on gradient norm of all ops to add
        :type gtol: float
        :param Etol: Stopping condition on error from ED of all ops
        :type Etol: float
        :param max_depth: Stopping condition on operators
        :type max_depth: int
        """
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
        self.e_savings = 0
        self.grad_savings = 0
        self.state_savings = 0
        state = t_ucc_state(params, ansatz, self.pool, self.ref)
        iteration = len(ansatz)
        print(f"\nADAPT Iteration {iteration}")
        print("Performing ADAPT:")
        E = (state.T@(self.H@state))[0,0] 
        Done = False
        while Done == False:           
            gradient = 2*np.array([((state.T@(self.H_adapt@(op@state)))[0,0]) for op in self.pool])
            gnorm = np.linalg.norm(gradient)
            if criteria == 'grad':
                idx = np.argsort(abs(gradient))

            if criteria == 'lucc':
                denom =  np.array([((state.T@(self.comm(self.H_adapt,op)@(op@state)))[0,0]) for op in self.pool])
                dE_opt = np.divide(-.25*gradient*gradient, denom, out=np.zeros(gradient.shape), where=denom>1e-12)
                idx = np.argsort(dE_opt)[::-1]

                 
            E = (state.T@(self.H@state))[0,0] 
            error = E - self.ed_energies[0]
            fid = ((self.ed_wfns[:,0].T)@state)[0]**2

            print(f"Operator/ Expectation Value/ Error")
            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0]
                err = val - self.ed_syms[0][key]
                print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            print(f"Next operator to be added: {self.v_pool[idx[-1]]}")
            print(f"Operator multiplicity {1+ansatz.count(idx[-1])}.")                
            print(f"Associated gradient:       {gradient[idx[-1]]:20.16f}")
            print(f"Gradient norm:             {gnorm:20.16f}")
            print(f"Fidelity to ED:            {fid:20.16f}")
            print(f"Current ansatz:")
            for i in range(0, len(ansatz)):
                print(f"{i} {params[i]} {self.v_pool[ansatz[i]]}") 
            print("|0>")  
            if gtol is not None and gnorm < gtol:
                Done = True
                print(f"\nADAPT finished.  (Gradient norm acceptable.)")
                continue
            if max_depth is not None and iteration+1 > max_depth:
                Done = True
                print(f"\nADAPT finished.  (Max depth reached.)")
                continue
            if Etol is not None and error < Etol:
                Done = True
                print(f"\nADAPT finished.  (Error acceptable.)")
                continue
            iteration += 1
            print(f"\nADAPT Iteration {iteration}")

            params = np.array([0] + list(params))
            ansatz = [idx[-1]] + ansatz
            if multiple == True: 
                H_vqe = copy.copy(self.H_vqe)
                pool = copy.copy(self.pool)
                ref = copy.copy(self.ref)
                params = multi_vqe(params, ansatz, H_vqe, pool, ref, self)  
            else:
                res = self.vqe(params, ansatz)
                params = copy.copy(res.x)
            state = t_ucc_state(params, ansatz, self.pool, self.ref)
            np.save(f"{self.system}/params", params)
            np.save(f"{self.system}/ops", ansatz)
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
            
        print(f"\nConverged ADAPT energy:    {E:20.16f}")            
        print(f"\nConverged ADAPT error:     {error:20.16f}")            
        print(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        print(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        print("\n---------------------------\n")
        print("\"Adapt.\" - Bear Grylls\n")
        print("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")

#Stupid non-object methods because multiprocessing doesn't work with OOP for some reason

def t_ucc_state(params, ansatz, pool, ref):
    state = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*pool[ansatz[i]], state)
    return state

def t_ucc_E(params, ansatz, H_vqe, pool, ref):
    state = t_ucc_state(params, ansatz, pool, ref)
    E = (state.T@(H_vqe)@state).todense()[0,0]
    return E       

def t_ucc_grad(params, ansatz, H_vqe, pool, ref):
    state = t_ucc_state(params, ansatz, pool, ref)
    hstate = H_vqe@state
    grad = [2*((hstate.T)@pool[ansatz[0]]@state).todense()[0,0]]
    hstack = scipy.sparse.hstack([hstate,state]) 
    for i in range(0, len(params)-1):
        hstack = scipy.sparse.linalg.expm_multiply(-params[i]*pool[ansatz[i]], hstack).tocsr()
        grad.append(2*((hstack[:,0].T)@pool[ansatz[i+1]]@hstack[:,1]).todense()[0,0])
    grad = np.array(grad)
    return grad

def t_ucc_hess(params, ansatz, H_vqe, pool, ref):
    J = copy.copy(ref)
    for i in reversed(range(0, len(params))):
        J = scipy.sparse.hstack([pool[ansatz[i]]@J[:,-1], J]).tocsr()
        J = scipy.sparse.linalg.expm_multiply(pool[ansatz[i]]*params[i], J)
    J = J.tocsr()[:,:-1]
    u, s, vh = np.linalg.svd(J.todense())       
    hess = 2*J.T@(H_vqe@J).todense()       
    state = t_ucc_state(params, ansatz, pool, ref)
    hstate = H_vqe@state
    for i in range(0, len(params)):            
        hstack = scipy.sparse.hstack([copy.copy(hstate), copy.copy(J[:,i])]).tocsc() 
        for j in range(0, i+1):
            hstack = scipy.sparse.linalg.expm_multiply(-params[j]*pool[ansatz[j]], hstack)
            ij = 2*((hstack[:,0].T)@pool[ansatz[j]]@hstack[:,1]).todense()[0,0]
            if i == j:
                hess[i,i] += ij
            else:
                hess[i,j] += ij
                hess[j,i] += ij
    w, v = np.linalg.eigh(hess)
    energy = ((hstate.T)@state).todense()[0,0]
    grad = ((J.T)@hstate).todense()
    #print(f"Energy: {energy:20.16f}")
    #print(f"GNorm:  {np.linalg.norm(grad):20.16f}")
    #print(f"Jacobian Singular Values:")
    spec_string = ""
    for sv in s:
        spec_string += f"{sv},"
    #print(spec_string)
    #print(f"Hessian Eigenvalues:")
    spec_string = ""
    for sv in w:
        spec_string += f"{sv},"
    #print(spec_string)
    return hess

def t_ucc_jac(params, ansatz, H_vqe, pool, ref):
    J = copy.copy(ref)
    for i in reversed(range(0, len(params))):
        J = scipy.sparse.hstack([pool[ansatz[i]]@J[:,-1], J]).tocsr()
        J = scipy.sparse.linalg.expm_multiply(pool[ansatz[i]]*params[i], J)
    J = J.tocsr()[:,:-1]
    return J


def vqe(params, ansatz, H_vqe, pool, ref, strategy = "newton-cg", energy = None):
    if energy is None or energy == t_ucc_E:
        energy = t_ucc_E
        jac = t_ucc_grad
        hess = t_ucc_hess

    if strategy == "newton-cg":
        res = scipy.optimize.minimize(energy, params, jac = jac, hess = hess, method = "newton-cg", args = (ansatz, H_vqe, pool, ref), options = {'xtol': 1e-10})
    return res

def multi_vqe(params, ansatz, H_vqe, pool, ref, xiphos, energy = None, guesses = 10):
    from multiprocessing import Pool
    start = time.time()
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if energy is None or energy == t_ucc_E:
        energy = t_ucc_E
        jac = t_ucc_grad
        hess = t_ucc_hess
    param_list = [copy.copy(params)]
    seeds = ['Recycled']
    E0s = [energy(params, ansatz, H_vqe, pool, ref)]
    for i in range(0, guesses):
        seed = i+guesses*(len(params)-1)
        seeds.append(seed)
        np.random.seed(seed)
        param_list.append(4*np.sqrt(2)*np.random.rand(len(params)))
        E0s.append(energy(param_list[-1], ansatz, H_vqe, pool, ref))

    iterable = [*zip(param_list, [ansatz for i in range(0, len(param_list))], [H_vqe for i in range(0, len(param_list))], [pool for i in range(0, len(param_list))], [ref for i in range(0, len(param_list))])] 
    with Pool(6) as p:
        L = p.starmap(vqe, iterable = iterable)
    params = solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos)
    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    return params
    
def solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos):
    Es = [L[i].fun for i in range(0, len(L))]
    xs = [L[i].x for i in range(0, len(L))]
    gs = [t_ucc_grad(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    hess = [t_ucc_hess(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    jacs = [t_ucc_jac(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    idx = np.argsort(Es)
    print(f"\nSolution analysis for {len(ansatz)} parameters:")
    print("Seed,E_initial,E_final,G_Norm,Smallest Jac SV,Smallest Hess EV,Fidelity,E,S_z,S^2,N")
    smins = []
    for i in idx:
        seed = seeds[i]
        E0 = E0s[i]
        EF = Es[i]
        g_norm = np.linalg.norm(gs[i])
        u, s, vh = np.linalg.svd(jacs[i].todense())       
        s_min = np.min(s)
        smins.append(np.min(s))
        w, v = np.linalg.eigh(hess[i])
        e_min = 1/np.min(w)
        state = t_ucc_state(xs[i], ansatz, pool, ref)
        fid = ((xiphos.ed_wfns[:,0].T)@state)[0]**2
        string = f"{seed},{E0},{EF},{g_norm},{s_min},{e_min},{fid}"
        for key in xiphos.sym_ops.keys():
            val = ((state.T)@(xiphos.sym_ops[key]@state))[0,0]
            err = val - xiphos.ed_syms[0][key]
            string += f",{val:20.16f}"
        print(string)
    return xs[idx[0]]
