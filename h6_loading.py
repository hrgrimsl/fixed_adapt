if __name__ == '__main__':
    from driver import *
    from pyscf_backend import *
    from of_translator import *
    import system_methods as sm
    import numpy as np
    import copy
    import scipy
    import openfermion as of
    #geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5"
    #geom = 'H 0 0 0; H 0 0 .9; H 0 0 1.8; H 0 0 2.7; H 0 0 3.6; H 0 0 4.5'
    #geom = 'H 0 0 0; H 0 0 3; H 0 0 6; H 0 0 9; H 0 0 12; H 0 0 15'
    #geom = 'H 0 0 0; H 0 0 3; H 0 0 6; H 0 0 9'
    geom = 'H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5'
    #geom = 'H 0 0 0; H 0 0 3'
    basis = "sto-3g"
    reference = "rhf"
    
    
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "h6_1A.chk", read = False)
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    s = sm.system_data(H, ref, N_e, N_qubits)
    pool, v_pool = s.uccsd_pool(approach = 'vanilla')
    xiphos = Xiphos(H, ref, "h6_1A_loading", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop}, verbose = "DEBUG")
    params = np.load('h6_1A_adapt/bre_params.npy')
    ansatz = list(np.load('h6_1A_adapt/bre_ops.npy'))
    E = t_ucc_E(params, ansatz, H, pool, ref)
    print(f"Energy of recycled ansatz: {E}")
   
