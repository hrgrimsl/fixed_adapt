from driver import *
from p4n_backend import *
from of_translator import *
import numpy as np
geometry = """
0 1
    H            0.000000000000     0.000000000000    0
    H           0.000000000000     0.000000000000    0.744359313148
symmetry c1
"""
N_c = 0

print("Real ADAPT:")
E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))
E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
N_e -= N_c
H, ref, N_qubits = of_from_arrays(E_nuc, H, I, N_e)

qubit_adapt(H, ref, N_e, N_qubits, factor = Eh, thresh = 1e-5)

print("Fixed ADAPT:")

for r in range(-5,31): 
    geometry = """
    0 1
    H 0 0 0
    """
    geometry += "H 0 0 "+str(0.744359313148 + .1*r)+"\nsymmetry c1"
    N_c = 0

    E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
    N_e = int(np.trace(D))

    #H, I = rotate(H, I, R, rotate_rdm = False)

    E_nuc, H, I, D = freeze_core(E_nuc, H, I, D, N_c)
    N_e -= N_c

    H, ref, N_qubits = of_from_arrays(E_nuc, H, I, N_e)
    fixed_adapt(H, ref, N_e, N_qubits, factor = Eh)




