from driver import *
from pyscf_backend import *
from of_translator import *
import numpy as np
import copy
#geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5"
#geom = 'H 0 0 0; H 0 0 .9; H 0 0 1.8; H 0 0 2.7; H 0 0 3.6; H 0 0 4.5'
geom = 'H 0 0 0; H 0 0 1.5; H 0 0 3; H 0 0 4.5; H 0 0 6; H 0 0 7.5'
basis = "sto-3g"
reference = "rhf"
interval1 = [.05*i for i in range(30, 71)]
interval2 = [.05*i for i in range(10, 30)]

E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False)
N_e = int(np.trace(D))
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
E, params, error = xiphos(H, ref, N_e, N_qubits, S2, Sz, Nop, thresh = None, pool = "uccsd", out_file = "scr2", chem_acc = True, subspace_algorithm = 'adapt', units = 'Eh')

Es = []
errs = []
hf = []
read = False
guess = 'read'


for x in interval1:
    geom = "H 0 0 0"
    for i in range(1, 6):
        geom += f"; H 0 0 {x*i}"
    print(geom)
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = read)
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    E, error, params = fixed_adapt(H, ref, N_e, N_qubits, S2, Sz, Nop, pool = 'uccsd', in_file = "scr2", units = 'Eh', guess = guess)
    Es.append(E)
    errs.append(error)
    hf.append(hf_energy)
    read = True
    guess = params
    for i in range(0, len(Es)):
        print(f"{hf[i]:20.16f} {Es[i]:20.16f} {errs[i]:20.16f}")

guess = 'read'
for x in reversed(interval2):
    geom = "H 0 0 0"
    for i in range(1, 6):
        geom += f"; H 0 0 {x*i}"
    print(geom)
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = read)
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    E, error, params = fixed_adapt(H, ref, N_e, N_qubits, S2, Sz, Nop, pool = 'uccsd', in_file = "scr2", units = 'Eh', guess = guess)
    Es = [E] + Es
    errs = [error] + errs
    hf = [hf_energy] + hf
    read = True
    guess = params
    for i in range(0, len(Es)):
        print(f"{hf[i]:20.16f} {Es[i]:20.16f} {errs[i]:20.16f}")
