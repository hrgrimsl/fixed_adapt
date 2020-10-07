from driver import *
from p4n_backend import *
from of_translator import *
import numpy as np
geometry = """
0 1
H 0 0 -1.391988561785
Li 0 0 .199954391039
symmetry c1
"""
N_c = 2

print("Real ADAPT:")
E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))


E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
N_e -= N_c
H, ref, N_qubits = of_from_arrays(E_nuc, H, I, N_e)
qubit_adapt(H, ref, N_e, N_qubits, factor = Eh)
