import system_methods as sm
import computational_tools as ct
import openfermion as of
import numpy as np
import scipy
import copy
import os
import logging
import time

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
        logging.basicConfig(filename = f"{system}/log.dat", level = verbose, format = "") 
        logging.info("---------------------------\n")
        if self.restart == False:
            logging.info("Starting a new calculation.\n")
        else:  
            logging.info("Restarting the calculation.\n")
        logging.info("---------------------------")

         
        #We do an exact diagonalization to check for degeneracies/symmetry issues
        logging.info("\nReference information:")
        self.ref_syms = {}
        for key in self.sym_ops.keys():
            val = (ref.T@(self.sym_ops[key]@ref))[0,0]
            logging.info(f"{key: >6}: {val:20.16f}")
            self.ref_syms[key] = val
        self.hf = self.ref_syms['H']        
        logging.info("\nED information:") 
        w, v = scipy.sparse.linalg.eigsh(H, k = min(H.shape[0]-1,10), which = "SA")
        self.ed_energies = w
        self.ed_wfns = v
        self.ed_syms = []
        for i in range(0, len(w)):
            logging.info(f"ED Solution {i+1}:")
            ed_dict = {}
            for key in self.sym_ops.keys():
                val = (v[:,i].T@(self.sym_ops[key]@v[:,i]))
                logging.info(f"{key: >6}: {val:20.16f}")
                ed_dict[key] = copy.copy(val)
            self.ed_syms.append(copy.copy(ed_dict))

        for key in self.sym_ops.keys():
            if key != "H" and abs(self.ed_syms[0][key] - self.ref_syms[key]) > 1e-8:
                logging.info(f"\nWARNING: <{key}> symmetry of reference inconsistent with ED solution.")
        if abs(self.ed_syms[0]["H"] - self.ed_syms[1]["H"]) < (1/Eh):
            logging.info(f"\nWARNING:  Lowest two ED solutions may be quasi-degenerate.")

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
            #temporarily reversing params, should be changed once I'm printing ansatz correctly
            ansatz = [self.v_pool.index(op)] + ansatz
            params = [float(param)] + params
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

    def t_ucc_E(self, params, ansatz):
        """Pseudo-trotterized UCC energy.  Ansatz and params are applied to reference in reverse order. 
        :param params: Parameters associated with ansatz.
        :type params: list 
        :param ansatz: List of operator indices, applied to reference in reversed order.
        :type ansatz: list
        :param ref: Reference state.
        :type ref: (2^N,1) array-like
        :return: energy, associated state 
        :rtype: list
        """
        try:
            E = self.e_dict(str(params))
            self.e_saving += 1
        except:
            state = self.t_ucc_state(params, ansatz)
            E = (state.T@(self.H_vqe)@state).todense()[0,0]
        return E       

    def t_ucc_state(self, params, ansatz):
        try:
            state = self.state_dict(str(params))
            self.state_saving += 1
        except:
            state = copy.copy(self.ref)
            for i in reversed(range(0, len(ansatz))):
                state = scipy.sparse.linalg.expm_multiply(params[i]*self.pool[ansatz[i]], state)
        return state

    def t_ucc_grad(self, params, ansatz):
        try:
            grad = self.grad_dict[str(params)]
            self.grad_saving += 1
        except:
            state = self.t_ucc_state(params, ansatz)
            hstate = self.H_vqe@state
            grad = [2*((hstate.T)@self.pool[ansatz[0]]@state).todense()[0,0]]
            hstack = scipy.sparse.hstack([hstate,state]) 
            for i in range(0, len(params)-1):
                hstack = scipy.sparse.linalg.expm_multiply(-params[i]*self.pool[ansatz[i]], hstack).tocsr()
                grad.append(2*((hstack[:,0].T)@self.pool[ansatz[i+1]]@hstack[:,1]).todense()[0,0])
            grad = np.array(grad)
            self.grad_dict[str(params)] = grad
        return grad
    

    def t_ucc_hess(self, params, ansatz):
        logging.info(f"\nVQE Iter. {self.vqe_iteration}")
        self.vqe_iteration += 1
        J = copy.copy(self.ref)
        for i in reversed(range(0, len(params))):
            J = scipy.sparse.hstack([self.pool[ansatz[i]]@J[:,-1], J]).tocsr()
            J = scipy.sparse.linalg.expm_multiply(self.pool[ansatz[i]]*params[i], J)

        J = J.tocsr()[:,:-1]
        u, s, vh = np.linalg.svd(J.todense())       
        hess = 2*J.T@(self.H_vqe@J).todense()       
        state = self.t_ucc_state(params, ansatz)
        hstate = self.H_vqe@state
        for i in range(0, len(params)):            
            hstack = scipy.sparse.hstack([copy.copy(hstate), copy.copy(J[:,i])]).tocsc() 
            for j in range(0, i+1):
                hstack = scipy.sparse.linalg.expm_multiply(-params[j]*self.pool[ansatz[j]], hstack)
                ij = 2*((hstack[:,0].T)@self.pool[ansatz[j]]@hstack[:,1]).todense()[0,0]
                if i == j:
                    hess[i,i] += ij
                else:
                    hess[i,j] += ij
                    hess[j,i] += ij
        w, v = np.linalg.eigh(hess)
        try:
            energy = self.e_dict[str(params)]
            self.e_saving += 1
        except: 
            energy = ((hstate.T)@state).todense()[0,0]
            self.e_dict[str(params)] = energy 
        try:
           grad = self.grad_dict[str(params)]
           self.grad_saving += 1
        except:
           grad = ((J.T)@hstate).todense()
           self.grad_dict[str(params)] = grad

        logging.info(f"Energy: {energy:20.16f}")
        logging.info(f"GNorm:  {np.linalg.norm(grad):20.16f}")
        logging.info(f"Jacobian Singular Values:")
        spec_string = ""
        for sv in s:
            spec_string += f"{sv},"
        logging.info(spec_string)
        logging.info(f"Hessian Eigenvalues:")
        spec_string = ""
        for sv in w:
            spec_string += f"{sv},"
        logging.info(spec_string)
        return hess
    
    def H_eff_analysis(self, params, ansatz):
        H_eff = copy.copy(self.H)
        for i in reversed(range(0, len(params))):
            U = scipy.sparse.linalg.expm(params[i]*self.pool[ansatz[i]])
            H_eff = ((U.T)@H@U).todense()
        E = ((self.ref.T)@H_eff@self.ref)
        logging.info("Analysis of H_eff:")
        logging.info(f"<0|H_eff|0> = {E}")
        w, v = np.linalg.eigh(H_eff)
        for sv in w:
            spec_string += f"{sv},"
        logging.info(f"Eigenvalues of H_eff:")
        logging.info(spec_string)
        

    def vqe(self, params, ansatz, strategy = "newton-cg", energy = None):
        """Variational quantum eigensolver for one ansatz 
        :param params: Parameters associated with ansatz.
        :type params: list 
        :param ansatz: List of operator indices, applied to reference in reversed order.
        :type ansatz: list
        :param ref: Reference state.
        :type ref: (2^N,1) array-like
        :param strategy: What minimization approach to use.  Defaults to BFGS.
        :type strategy: string, optional
        :param energy: Energy function to minimize- Defaults to Trotter UCC.
        :type energy: function, optional       
        :rtype: energy, gnorm, 
        """
        self.vqe_iteration = 0
        if energy is None or energy == self.t_ucc_E:
            energy = self.t_ucc_E
            jac = self.t_ucc_grad
            hess = self.t_ucc_hess

        if strategy == "newton-cg":
            res = scipy.optimize.minimize(energy, params, jac = jac, hess = hess, method = "newton-cg", args = (ansatz), options = {'xtol': 1e-16})

        return res
        
    def adapt(self, params, ansatz, ref, gtol = None, Etol = None, max_depth = None, criteria = 'grad'):
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
        state = self.t_ucc_state(params, ansatz)
        iteration = len(ansatz)
        logging.info(f"\nADAPT Iteration {iteration}")
        logging.info("Performing ADAPT:")
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

            logging.info(f"Operator/ Expectation Value/ Error")
            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0]
                err = val - self.ed_syms[0][key]
                logging.info(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            logging.info(f"Next operator to be added: {self.v_pool[idx[-1]]}")
            logging.info(f"Operator multiplicity {1+ansatz.count(idx[-1])}.")                
            logging.info(f"Associated gradient:       {gradient[idx[-1]]:20.16f}")
            logging.info(f"Gradient norm:             {gnorm:20.16f}")
            logging.info(f"Fidelity to ED:            {fid:20.16f}")
            logging.info(f"Current ansatz:")
            for i in range(0, len(ansatz)):
                logging.info(f"{i} {params[i]} {self.v_pool[ansatz[i]]}") 
            logging.info("|0>")  
            if gtol is not None and gnorm < gtol:
                Done = True
                logging.info(f"\nADAPT finished.  (Gradient norm acceptable.)")
                continue
            if max_depth is not None and iteration+1 > max_depth:
                Done = True
                logging.info(f"\nADAPT finished.  (Max depth reached.)")
                continue
            if Etol is not None and error < Etol:
                Done = True
                logging.info(f"\nADAPT finished.  (Error acceptable.)")
                continue
            iteration += 1
            logging.info(f"\nADAPT Iteration {iteration}")

            params = np.array([0] + list(params))
            ansatz = [idx[-1]] + ansatz
            res = self.vqe(params, ansatz)
            params = copy.copy(res.x)
            state = self.t_ucc_state(params, ansatz)
            np.save(f"{self.system}/params", params)
            np.save(f"{self.system}/ops", ansatz)
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
            
        logging.info(f"\nConverged ADAPT energy:    {E:20.16f}")            
        logging.info(f"\nConverged ADAPT error:     {error:20.16f}")            
        logging.info(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        logging.info(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        logging.info("\n---------------------------\n")
        logging.info("\"Adapt.\" - Bear Grylls\n")
        logging.info("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")

