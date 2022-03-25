if __name__ == '__main__':
    from driver import *
    from pyscf_backend import *
    from of_translator import *
    import system_methods as sm
    import numpy as np
    import copy
    import scipy
    #geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5"
    #geom = 'H 0 0 0; H 0 0 .9; H 0 0 1.8; H 0 0 2.7; H 0 0 3.6; H 0 0 4.5'
    #geom = 'H 0 0 0; H 0 0 3; H 0 0 6; H 0 0 9; H 0 0 12; H 0 0 15'
    #geom = 'H 0 0 0; H 0 0 3; H 0 0 6; H 0 0 9'
    geom = 'H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3'
    #geom = 'H 0 0 0; H 0 0 3'
    basis = "sto-3g"
    reference = "rhf"
    ''' 
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False, feed_C = False)
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    s = sm.system_data(H, ref, N_e, N_qubits)
    
    pool, v_pool = s.uccsd_pool(approach = 'vanilla')
    #pool, v_pool = s.tang_pool()
     
    xiphos = Xiphos(H, ref, "shucc_tests", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop}, verbose = "DEBUG")
    ansatz = [i for i in range(0, len(pool))]

    grad = xiphos.ucc_grad_zero(ansatz)
    hess = xiphos.ucc_hess_zero(ansatz)
    trotter_hess = np.array(t_ucc_hess(np.zeros(len(pool)), ansatz, H, pool, ref))
    jerk = xiphos.ucc_diag_jerk_zero(ansatz)
   
    shucc = hf_energy-.5*grad.T@(np.linalg.inv(hess))@(grad)
    shtucc = hf_energy-(.5*grad.T@(np.linalg.inv(trotter_hess))@(grad))
    enucc = hf_energy + scipy.optimize.minimize(xiphos.cubic_energy, np.zeros(len(ansatz)), method = 'bfgs', args = (grad, hess, jerk)).fun 
    entucc = hf_energy + scipy.optimize.minimize(xiphos.cubic_energy, np.zeros(len(ansatz)), method = 'bfgs', args = (grad, trotter_hess, jerk)).fun
    d_inf = scipy.optimize.minimize(xiphos.ucc_inf_d_E, np.zeros(len(ansatz)), method = 'bfgs', args = (ansatz, hf_energy, grad, hess)).fun 
    td_inf = scipy.optimize.minimize(xiphos.tucc_inf_d_E, np.zeros(len(ansatz)), method = 'bfgs', args = (ansatz, hf_energy, grad, trotter_hess)).fun 
    print("Canonical Orbitals")
    print(f"SHUCC: {shucc}")
    print(f"SHtUCC: {shtucc}")
    print(f"ENUCC: {enucc}")
    print(f"ENtUCC: {entucc}")
    print(f"d_inf: {d_inf}")
    print(f"td_inf: {td_inf}")
    '''
    dft_C = np.array([[ 0.25693606,  0.54359893,  0.76656818,  0.64787664],
    [ 0.44234996,  0.35608392, -0.49908811, -1.12594983],
    [ 0.44234996, -0.35608392, -0.49908811,  1.12594983],
    [ 0.25693606, -0.54359893,  0.76656818, -0.64787664]])

    np.save('dft_C.npy', dft_C)

    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False, feed_C = 'dft_C.npy')
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)

    E0 = (ref.T@(H@ref))[0,0]
    s = sm.system_data(H, ref, N_e, N_qubits)
     
    pool, v_pool = s.uccs_then_d_pool(approach = 'vanilla')
    print(v_pool, flush = True)
    #pool, v_pool = s.tang_pool()
     
    xiphos = Xiphos(H, ref, "shucc_tests", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop}, verbose = "DEBUG")
    ansatz = [i for i in range(0, len(pool))]

    grad = xiphos.ucc_grad_zero(ansatz)
    hess = xiphos.ucc_hess_zero(ansatz)
    trotter_hess = np.array(t_ucc_hess(np.zeros(len(pool)), ansatz, H, pool, ref))
    jerk = xiphos.ucc_diag_jerk_zero(ansatz)
   
    shucc = E0-.5*grad.T@(np.linalg.inv(hess))@(grad)
    shtucc = E0-(.5*grad.T@(np.linalg.inv(trotter_hess))@(grad))
    enucc = E0 + scipy.optimize.minimize(xiphos.cubic_energy, np.zeros(len(ansatz)), method = 'bfgs', args = (grad, hess, jerk)).fun 
    entucc = E0 + scipy.optimize.minimize(xiphos.cubic_energy, np.zeros(len(ansatz)), method = 'bfgs', args = (grad, trotter_hess, jerk)).fun
    d_inf = scipy.optimize.minimize(xiphos.ucc_inf_d_E, np.zeros(len(ansatz)), method = 'bfgs', args = (ansatz, E0, grad, hess)).fun 
    td_inf = scipy.optimize.minimize(xiphos.tucc_inf_d_E, np.zeros(len(ansatz)), method = 'bfgs', args = (ansatz, E0, grad, trotter_hess)).fun 
    print("DFT Orbitals")
    print(f"SHUCC: {shucc}")
    print(f"SHtUCC: {shtucc}")
    print(f"ENUCC: {enucc}")
    print(f"ENtUCC: {entucc}")
    print(f"d_inf: {d_inf}")
    print(f"td_inf: {td_inf}")
