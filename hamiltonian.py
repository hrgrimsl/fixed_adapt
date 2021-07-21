from pyscf_backend import *
from of_translator import *
import numpy as np
import copy
from sys import argv
from pyscf import gto, scf
import system_methods as sm
import computational_tools as ct
geom = "H 0 0 0; H 0 0 1; H 0 1 1; H 1 -1 -1; H 1, -1, -2; H 1, -1, -3"
E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, "STO-3G", "rhf", chkfile = 'h6_chk', read = False, feed_C = "B3LYP_C.npy")
N_e = int(np.trace(D))
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
system = sm.system_data(H, ref, N_e, N_qubits)
ops, v_pool = system.raw_uccsd_pool()
zero_params = [0 for i in ops]
N = len(ops)

def num_grad(fun, x, h):
    deriv = []
    for i in range(0, len(x)):
        forw = copy.copy(x)
        forw[i] += h
        back = copy.copy(x)
        back[i] -= h
        deriv.append((fun(forw, H, ops, ref) - fun(back, H, ops, ref))/(2*h))
    return np.array(deriv)

def num_hess(fun, x, h):
    hess = []
    for i in range(0, len(x)):
        forw = copy.copy(x)
        forw[i] += h
        back = copy.copy(x)
        back[i] -= h
        hess.append((num_grad(fun, forw, h) - num_grad(fun, back, h))/(2*h))
    return np.array(hess)

def shucc_E(x, E0, g, hessian):
    return E0 + g.T@x + .5*(x.T)@(hessian@x)

def shucc_grad(x, E0, g, hessian):
    return g + hessian@x

E0 = (ref.T@(H@ref))[0,0]

#Untrotterized energy
#grad = num_grad(ct.simple_uccsd_energy, zero_params, 1e-6)
#np.save("B3LYP_UCCSD_grad", grad)
#hess = num_hess(ct.simple_uccsd_energy, zero_params, 1e-3)
#np.save("B3LYP_UCCSD_hess", hess)

#res = scipy.optimize.minimize(shucc_E, np.zeros((len(ops),1)), jac = shucc_grad, method = "bfgs", args = (E0, np.load("B3LYP_UCCSD_grad.npy"), np.load("B3LYP_UCCSD_hess.npy")), options = {"gtol": 1e-12, "disp": True})
#print(res.fun)

grad = num_grad(ct.simple_energy, zero_params, 1e-6)
np.save("B3LYP_tUCCSD_grad", grad)
hess = num_hess(ct.simple_energy, zero_params, 1e-4)
np.save("B3LYP_tUCCSD_hess", hess)

        
