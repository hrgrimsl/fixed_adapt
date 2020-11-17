from driver import *
from p4n_backend import *
from of_translator import *
import numpy as np
geometry = """
0 1
H 0 0 0
H 0 0 3
H 0 0 6
H 0 0 9
H 0 0 12
H 0 0 15
symmetry c1
"""
N_c = 0


E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))

E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
N_e -= N_c
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H, I, N_e)
s2 = 0
sz = 0
n = 6 
L = H
L += (S2-s2*scipy.sparse.identity(S2.shape[0])).dot(S2 - s2*scipy.sparse.identity(S2.shape[0]))
#L += (Sz-sz*scipy.sparse.identity(Sz.shape[0])).dot(Sz - sz*scipy.sparse.identity(Sz.shape[0]))
#L += (Nop-n*scipy.sparse.identity(Nop.shape[0])).dot(Nop - n*scipy.sparse.identity(Nop.shape[0]))
adapt(H, ref, N_e, N_qubits, S2, Sz, Nop, L = L, pool = "4qubit", factor = Eh, thresh = 1e-6, verbose = True)

