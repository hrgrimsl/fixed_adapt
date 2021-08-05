from pyscf_backend import *
from of_translator import *
import numpy as np
import copy
from sys import argv
from pyscf import gto, scf
import system_methods as sm
import computational_tools as ct
from driver import *


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

def num_jerk(fun, x, h):
    jerk = []
    for i in range(0, len(x)):
        forw = copy.copy(x)
        forw[i] += h
        back = copy.copy(x)
        back[i] -= h
        jerk.append((num_hess(fun, forw, h) - num_hess(fun, back, h))/(2*h))
    return np.array(jerk)

def shucc_E(x, E0, g, hessian):
    return E0 + g.T@x + .5*(x.T)@(hessian@x)

def shucc_grad(x, E0, g, hessian):
    return g + hessian@x

def shucc_hess(x, E0, g, hessian):
    return hessian 

def enucc_E(x, E0, g, hessian, jerk):
    E = E0 + g.T@x + .5*x.T@hessian@x + (1/6)*contract('iii,i,i,i', jerk, x, x, x)
    return E

def enucc_grad(x, E0, g, hessian, jerk):
    grad = g + hessian@x + .5*contract('iii,i,i->i', jerk, x, x)
    return grad


    
geom = "H 0 0 0; H 0 .1 1; H .2 0 2.1; H 0 -.1 3.5; H 0 0 4; H 0 0 5"
E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, "STO-3G", "rhf", chkfile = 'h6_chk', read = False, feed_C = "B3LYP_C.npy")
N_e = int(np.trace(D))
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
s = sm.system_data(H, ref, N_e, N_qubits)
pool, v_pool = s.raw_uccsd_pool(spin_adapt = False)
N = len(pool)

E0 = (ref.T@(H@ref))[0,0]
xiphos = Xiphos(H, ref, "h6_shucc", pool, v_pool, verbose = "DEBUG")
ansatz = [i for i in range(0, len(pool))]

grad = xiphos.ucc_grad_zero(ansatz)
hess = xiphos.ucc_hess_zero(ansatz)
'''
jerk = xiphos.ucc_diag_jerk_zero(ansatz)

res = scipy.optimize.minimize(shucc_E, np.zeros((len(pool))), jac = shucc_grad, method = "bfgs", args = (E0, grad, hess), options = {"gtol": 1e-12, "disp": True})
print("SHUCC solution:")
print("E")
print(res.fun)
print("GNORM")
print(np.linalg.norm(res.jac))
print("Success")
print(res.success)

res = scipy.optimize.minimize(enucc_E, np.zeros((len(pool))), jac = enucc_grad, method = "bfgs", args = (E0, grad, hess, jerk), options = {"gtol": 1e-12, "disp": True})
print("ENUCC solution:")
print("E")
print(res.fun)
print("GNORM")
print(np.linalg.norm(res.jac))
print("Success")
print(res.success)
'''

def inf_cb(x):
    print(xiphos.ucc_inf_d_E(x, ansatz, E0, grad, hess))

res = scipy.optimize.minimize(xiphos.ucc_inf_d_E, np.zeros((len(pool))), method = "bfgs", jac = '3-point', args = (ansatz, E0, grad, hess), callback = inf_cb, options = {"disp": True})
print("HATER2_Inf solution:")
print("E")
print(res.fun)
print("GNORM")
print(np.linalg.norm(res.jac))
print("Success")
print(res.success)
'''
#res = scipy.optimize.minimize(shucc_E, np.zeros((len(pool),1)), jac = shucc_grad, method = "bfgs", args = (E0, np.load("B3LYP_UCCSD_grad.npy"), np.load("B3LYP_UCCSD_hess.npy")), options = {"gtol": 1e-12, "disp": True})

#print(res.fun)

#Untrotterized energy
#grad = num_grad(ct.simple_uccsd_energy, zero_params, 1e-6)
#np.save("B3LYP_UCCSD_grad", grad)
#hess = num_hess(ct.simple_uccsd_energy, zero_params, 1e-3)
#np.save("B3LYP_UCCSD_hess", hess)
#jerk = num_jerk(ct.simple_uccsd_energy, zero_params, 1e-2)
#np.save("B3LYP_UCCSD_jerk", jerk)




#grad = num_grad(ct.simple_energy, zero_params, 1e-6)
#np.save("B3LYP_tUCCSD_grad", grad)
#hess = num_hess(ct.simple_energy, zero_params, 1e-3)
#np.save("B3LYP_tUCCSD_hess", hess)
#jerk = num_jerk(ct.simple_energy, zero_params, 1e-2)
#np.save("B3LYP_UCCSD_jerk", jerk)
'''
        
