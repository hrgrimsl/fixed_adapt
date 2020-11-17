from pyscf_backend import *
from of_translator import *
import numpy as np
from sys import argv
from driver import *
prog, batch_start, batch_end, infile, r = argv

geometry = "H 0 0 0"
for i in range(1, 6):
    geometry += "; H 0 0 "+str(i*float(r))

in_file = infile + '.dat'
N_c = 0

E_nuc, H, I, D, C_ao_mo = get_integrals(geometry, "sto-3g", "rhf")
N_e = int(np.trace(D))
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H, I, N_e, unpaired = 0)
verbose = True
min_error = 1000
for i in range(int(batch_start), int(batch_end)):    
    error, guess = fixed_adapt(H, ref, N_e, N_qubits, S2, Sz, Nop, pool = "uccsd", verbose = verbose, in_file = in_file, guess = i)
    verbose = False
    if error < min_error:
        min_error = error
log_name = infile+'_'+r+'_'+batch_start+'_'+batch_end+'.tmp'
log = open(log_name, "w")
log.write(str(min_error))
