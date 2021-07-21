#Driver for new ADAPT project with Bryan
import system_methods as sm
import computational_tools as ct
import scipy
import copy
import openfermion as of
import numpy as np
import warnings
import os

#Globals
Eh = 627.5094740631

def xiphos(H, ref, N_e, N_qubits, S2, Sz, Nop, thresh = 1e-3, depth = None, L = None, pool = "4qubit", spin_adapt = True, out_file = 'out', units = 'kcal/mol', verbose = True, subspace_algorithm = 'xiphos', screen = False, xiphos_no = 1, persist = False, qse_cull = False, eps = 1e-8, chem_acc = False, sample_vqe = False, seeds = 125, sample_uccsd = False, sample_vccsd = False, oo = False, dump_dir = 'dumpdir', seed_offset = 0, state_analysis = False, load = False, gimbal = False, rscale = 1):
    if os.path.exists(dump_dir):
        os.system(f'rm -r {dump_dir}')
    os.makedirs(dump_dir)
    if L is None:
        L = copy.copy(H)
    system = sm.system_data(H, ref, N_e, N_qubits)
    eig = scipy.sparse.linalg.eigsh(L, k = 1, which = 'SA')
    #eig2 = np.linalg.eigh(L.todense())
    exact_E = eig[0][0]
    #try:
    #    assert(eig2[0][0] - eig[0][0] < 1e-12)
    #except:
    #    print("Numpy and scipy are giving different ground states.")
    if thresh is None:
        thresh = 0
    if units == 'kcal/mol':
        factor = Eh
        unit_str = '(kcal/mol)'
    elif units == 'Eh':
        factor = 1
        unit_str = '(a.u.)'
    else:
        print("Units not recognized.")
        exit()
    

        
    #Fetch pool
    if pool == "4qubit":
        pool = []
        string_pool = []
        for i in range(1, 5):
            ops, strings = system.k_qubit_pool(i)
            pool += ops
            string_pool += strings

    elif pool == "uccsd":
        pool, string_pool = system.uccsd_pool(spin_adapt = spin_adapt)        
    elif pool == "uccgsd":
        pool, string_pool = system.uccgsd_pool(spin_adapt = spin_adapt)        
    elif pool == "uccd":
        pool, string_pool = system.uccd_pool(spin_adapt = spin_adapt)
    elif pool == "sc_uccsd":
        pool, string_pool = system.sc_uccsd_pool()
    else:
        print("Pool not recognized.")
        exit()
    if sample_vccsd == True:
        vccsd_pool, vcc_string_pool = system.vccsd_pool(spin_adapt = spin_adapt)
    if oo == True:
        singles, v_singles = system.uccgs_pool(spin_adapt = True)
    else:
        singles = []
    if load == False:
        K = [ref]
        ops = [[]]
        params = [[]]
        uccsd_params = []
        vccsd_params = []
        full_ops = [[]]
        full_params = [[]]
        Done = False
        iters = 0
    else:
        params = [list(np.load(f"{load}_params.npy"))]
        ops = [list(np.load(f"{load}_ops.npy"))]
        state = ct.prep_state([pool[j] for j in ops[0]], system.ref, params[0])
        K = [state]
        Done = False
        iters = len(ops)
        full_ops = copy.copy(ops)
        full_params = copy.copy(params)
    if verbose == True:
         print('-'*170)
         if subspace_algorithm == 'xiphos':
             print(" "*50+"XIPHOS: eXpressive Interpolation by Parallel Handling of Operator Selection")
         elif subspace_algorithm == 'aegis':
             print(" "*52+"AEGIS: Aces & Eights Generation of Interpolative Subspace")
         elif subspace_algorithm == 'adapt':
             print(" "*73+"Traditional ADAPT")
         else:
             print("Algorithm not recognized.")
             exit()
         print(" "*75+"H.R. Grimsley")
         print('-'*170)
         #print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Old   |New   |All   ', 'Energy '+unit_str, 'Error '+unit_str, '<S^2>', '<S_z>', '<N>'))

    qse_L, E, s2v, szv, nv, v = ct.qse(K, L, H, S2, Sz, Nop)
    #if verbose == True:
        #print('{:<20}|{:<6}|{:<6}|{:<6}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|'.format(iters, len(K)-len(ops), len(ops), len(K), factor*E, factor*(E-exact_E), s2v, szv, nv))
    print(f"Current full energy: {E:20.16f} {units}")
    print(f"Current error:       {E-exact_E:20.16f} {units}")
    print(f"Current <S^2>:        {s2v}")
    print(f"Current <S_z>:        {szv}")
    print(f"Current <N>:          {nv}")

    
    while Done == False:
        iters += 1
        new_ops = []
        new_params = []
        Ks = []
        for i in range(0, len(ops)):
            grad = abs(factor*np.array([2*(K[i].T.dot(L).dot(op).dot(K[i]))[0,0] for op in pool]))
            sort = np.argsort(abs(grad))[::-1]

            if subspace_algorithm == 'xiphos':
                new_ops += [[sort[j]]+ops[i] for j in range(0, xiphos_no) if (grad[sort[j]] > thresh or (persist == False and xiphos_no > 1))] 
                new_params += [[0]+params[i] for j in range(0, xiphos_no) if (grad[sort[j]] > thresh or (persist == False and xiphos_no > 1))] 
                Ks += [K[i] for j in range(0, xiphos_no) if (grad[sort[j]] > thresh or (persist == False and xiphos_no > 1))]
               
            if subspace_algorithm == 'aegis':
                new_ops += [[sort[j]]+ops[i] for j in range(0, len(sort)) if grad[sort[0]]-grad[sort[j]] < eps] 
                new_params += [[0]+params[i] for j in range(0, len(sort)) if grad[sort[0]]-grad[sort[j]] < eps] 
                Ks += [K[i] for j in range(0, len(sort)) if grad[sort[0]]-grad[sort[j]] < eps] 
            
            if subspace_algorithm == 'adapt':
                new_ops += [[sort[0]]+ops[i] for j in range(0,1) if np.linalg.norm(grad) > thresh]
                new_params += [[0] + params[i] for j in range(0,1) if np.linalg.norm(grad) > thresh]
                Ks += [K[i] for j in range(0,1) if np.linalg.norm(grad) > thresh]
                print(f"Iteration:               {iters:20d}")
                print(f"Gradient norm:           {np.linalg.norm(grad):20.16e}")
                print(f"Largest gradient:        {grad[sort[0]]:20.16e}")
                print(f"Associated operator:     {sort[0]:20d}")
                print(string_pool[sort[0]])
        print("Saving current ops and params.")
        pfile = f"{out_file}_params"
        ofile = f"{out_file}_ops"
        np.save(pfile, np.array(full_params[0]))
        np.save(ofile, np.array(full_ops[0]))
        if len(new_ops) == 0 or (depth != None and iters == depth+1) or ((E-exact_E)*Eh < 1 and chem_acc == True):

            print(f"Final energy:     {factor*E:20.16f}")
            print(f"Final error:      {factor*(E-exact_E):20.16f}")

            return factor*E, params[0], factor*(E-exact_E)

        #One-param Screening- if one-parameter optimization doesn't give a linearly independent vector, kick it.                 
        if screen == True:                 
            scrn = [] 
            for k in range(0, len(new_ops)):
                scrn_L, scrn_params, gnorm, hcond = ct.vqe(L, [pool[new_ops[k][0]]], Ks[k], [0])
                scrn.append(scipy.sparse.linalg.expm_multiply(pool[new_ops[k][0]]*scrn_params[0], Ks[k]))    

            scrn2 = []
            new_ops2 = []
            new_params2 = []
            for i in range(0, len(scrn)):
                for j in range(0, len(scrn2)):
                    scrn[i] -= scrn2[j].T.dot(scrn[i])[0,0]*scrn2[j]/scrn2[j].T.dot(scrn2[j])[0,0] 
                if scrn[i].T.dot(scrn[i])[0,0] > eps:
                    scrn2.append(scrn[i]/np.sqrt(scrn[i].T.dot(scrn[i])[0,0]))
                    new_ops2.append(new_ops[i])
                    new_params2.append(new_params[i]) 
            new_ops = copy.copy(new_ops2)
            new_params = copy.copy(new_params2)


        #Rigorous Independence Screening
        new_K = []
        old_K = copy.copy(K)
        prev_ops = copy.copy(full_ops)
        prev_params = copy.copy(full_params)
        preold_K = copy.copy(K)
        for i in range(0, len(new_params)):
            if sample_vqe == False:    
                L_val, new_params[i], gnorm, hcond, U = ct.vqe(L, [pool[j] for j in new_ops[i]], system.ref, new_params[i], singles = singles, dump_dir = dump_dir)
            else:
                L_val, new_params[i], gnorm, hcond, U = ct.sample_vqe(L, [pool[j] for j in new_ops[i]], system.ref, new_params[i], singles = singles, dump_dir = dump_dir, seeds = seeds, seed_offset = seed_offset, gimbal = gimbal, rscale = rscale)
            if isinstance(U, type(3)):
                pass
                
            elif isinstance(U, type(None)) == False:
                L = U.T@(L)@U
                H = U.T@(H)@U
            
            if sample_uccsd == True:
                dummy, uccsd_params = ct.sample_uccsd(L, [pool[j] for j in new_ops[i]], system.ref, [0]+uccsd_params, singles = singles, dump_dir = dump_dir + '_uccsd')
            if sample_vccsd == True:
                dummy2, vccsd_params = ct.sample_vccsd(L, [vccsd_pool[j] for j in new_ops[i]], system.ref, [0]+vccsd_params, dump_dir = dump_dir + '_vccsd')
            state = ct.prep_state([pool[j] for j in new_ops[i]], system.ref, new_params[i])

            new_K.append(state.real) 
            E_val = state.T.dot(H).dot(state).real[0,0]
            print(f"GNORM from BFGS:  {np.linalg.norm(np.array(gnorm))}")      

        preK = copy.copy(new_K)         
        K2 = []
        ops = []
        params = []

        for i in range(0, len(new_K)):
            for j in range(0, len(K2)):
                new_K[i] -= K2[j].T.dot(new_K[i])[0,0]*K2[j]/K2[j].T.dot(K2[j])[0,0] 
            if new_K[i].T.dot(new_K[i])[0,0] > eps:
                K2.append(new_K[i]/np.sqrt(new_K[i].T.dot(new_K[i])[0,0]))
                ops.append(new_ops[i])
                params.append(new_params[i])
        
         
        full_ops = copy.copy(ops)                
        full_params = copy.copy(params)                 


        K = [preK[i] for i in range(0, len(new_K)) if new_K[i].T.dot(new_K[i])[0,0] > eps]

        if persist == True:
            for i in range(0, len(old_K)):
                for j in range(0, len(K2)):
                    old_K[i] -= K2[j].T.dot(old_K[i])[0,0]*K2[j]
                if old_K[i].T.dot(old_K[i])[0,0] > eps:
                    K2.append(old_K[i]/np.sqrt(old_K[i].T.dot(old_K[i])[0,0]))
                    K.append(preold_K[i])
                    full_ops.append(prev_ops[i])
                    full_params.append(prev_params[i])
        
        new_ks = len(ops)
        old_ks = len(K)-len(ops)
        qse_L, E, s2v, szv, nv, v = ct.qse(K2, L, H, S2, Sz, Nop)

        if qse_cull == True:
            save_new = [i for i in range(0, new_ks) if abs(v[i]) > 5e-3] 
            save_old = [i for i in range(new_ks, len(v)) if abs(v[i]) > 5e-3]
            ops = [ops[i] for i in save_new]
            params = [params[i] for i in save_new]
            K = [K[i] for i in save_new + save_old]
            full_ops = [full_ops[i] for i in save_new + save_old]
            full_params = [full_params[i] for i in save_new + save_old]
 
        #if verbose == True:
            #print('{:<20}|{:<6}|{:<6}|{:<6}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|'.format(iters, len(K)-len(ops), len(ops), len(K), factor*E, factor*(E-exact_E), s2v, szv, nv))
        print(f"Current full energy: {E:20.16f} {units}")
        print(f"Current error:       {E-exact_E:20.16f} {units}")
        print(f"Current <S^2>:        {s2v}")
        print(f"Current <S_z>:        {szv}")
        print(f"Current <N>:          {nv}")
        print("Current ansatz (1st op closest to H):")
        for j in range(0, len(full_ops)):
            p = full_params[j]
            o = full_ops[j]

            for k in range(0, len(o)):
                print(f"{k:5d}     {p[k]:20.8f}     {str(string_pool[o[k]]).split('+')[0]}")




def fixed_adapt(H, ref, N_e, N_qubits, S2, Sz, Nop, L = None, pool = "4qubit", spin_adapt = True, in_file = 'out', units = 'kcal/mol', verbose = True, eps = 1e-9, guess = 'hf'):    
    if L == None:
        L = copy.copy(H)
    system = sm.system_data(H, ref, N_e, N_qubits)

    eig = scipy.sparse.linalg.eigsh(L, k = 1, which = 'SA')
    #eig2 = np.linalg.eigh(L.todense())
    exact_E = eig[0][0]
    #try:
    #    assert(eig2[0][0] - eig[0][0] < 1e-12)
    #except:
    #    print("Numpy and scipy are giving different ground states.")
 
    if units == 'kcal/mol':
        factor = Eh
        unit_str = '(kcal/mol)'
    elif units == 'Eh':
        factor = 1
        unit_str = '(a.u.)'
    else:
        print("Units not recognized.")
        exit()
 
    #Fetch pool
    if pool == "4qubit":
        pool = []
        for i in range(1, 5):
            pool += system.k_qubit_pool(i)
    elif pool == "uccsd":
        pool, string_pool = system.uccsd_pool(spin_adapt = spin_adapt)        
    elif pool == "uccgsd":
        pool, string_pool = system.uccgsd_pool(spin_adapt = spin_adapt)        
    else:
        print("Pool not recognized.")
        exit()

    K = []


    parity = 0
    params = [list(np.load(f"{in_file}_params.npy"))]
    ops = [list(np.load(f"{in_file}_ops.npy"))]

    new_params = copy.copy(params)
    #print([list(np.array(i)) for i in new_params])
    if guess == 'read':        
        guess = [list(np.array(i)) for i in new_params]
    elif guess == 'hf':
        guess = [list(np.zeros(len(i))) for i in new_params]
    elif guess == 'perturbed_hf':
        guess = [[1e-6 for j in range(0, len(i))] for i in new_params]
    elif isinstance(guess, int):
        np.random.seed(seed = guess)
        guess = [list(np.random.uniform(-1,1,np.array(new_params[0]).shape))]
    for i in range(0, len(ops)):
        L_val, new_params[i], gnorm, hcond = ct.vqe(L, [pool[j] for j in ops[i]], system.ref, guess[i], gtol = eps)
        K.append(ct.prep_state([pool[j] for j in ops[i]], system.ref, new_params[i]))
    qse_L, E, s2v, szv, nv, v = ct.no_qse(K, L, H, S2, Sz, Nop)

    if verbose == True:
        print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Energy '+unit_str, 'Error '+unit_str, '<S^2>', '<S_z>', '<N>'))
    print('{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20.12}|'.format(factor*E, factor*(E-exact_E), s2v, szv, nv))
    return factor*E, factor*(E-exact_E), new_params, gnorm, hcond



