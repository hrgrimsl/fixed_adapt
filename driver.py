#Driver for new ADAPT project with Bryan
import system_methods as sm
import computational_tools as ct
import scipy
import copy
import openfermion as of
import numpy as np

#Globals
Eh = 627.5094740631

def qubit_adapt(H, ref, N_e, N_qubits, S2, depth = None, thresh = 1e-3, out_file = 'out.dat', factor = Eh):
    f = open(out_file, 'w')
    system = sm.system_data(H, ref, N_e, N_qubits)
    print("Done! Generating qubit pool...")
    pool = []
    for i in range(1, 5):
        pool += system.k_qubit_pool(i)
    print("SCF Energy:")
    print(system.hf_energy)
    print("FCI Energy:")
    print(system.ci_energy)

    print("Done! Performing ADAPT calculation...")

    cur_state = system.ref
    ansatz = []
    params = []
    Done = False
    iteration = 0
    print('\n')
    print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Energy (kcal/mol)', 'Error (kcal/mol)', 'Last del (a.u.)', 'Newest Operator', '<S^2>'))
    while Done == False:
        iteration += 1
        max_grad = 0
        new_op = None
        for i in range(len(pool)):
            op = copy.copy(pool[i])
            grad = 2*abs(cur_state.T.dot(system.H).dot(op).dot(cur_state)[0,0].real)
            if grad > max_grad:
                max_grad = copy.copy(grad)
                new_op = copy.copy(op)
                new_idx = copy.copy(i)
        if (max_grad < thresh and depth == None) or (depth != None and depth == iteration-1):
            Done = True 
        else:
            ansatz = [new_op] + ansatz
            params = [0] + params
            energy, params = ct.vqe(system.H, ansatz, system.ref, params)
            cur_state = copy.copy(system.ref)
            for i in reversed(range(0, len(params))): 
                cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
            S2_state = cur_state.T.dot(S2).dot(cur_state)[0,0].real
            print('-'*105)
            print('{:<20}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20}|{:<20}|'.format(iteration, factor*energy, factor*(energy-system.ci_energy), max_grad, system.pool[new_idx], S2_state))
            #print(str(factor*energy)+' '+str(factor*(energy-system.ci_energy)))
            f.write(str(system.pool[new_idx])+'\n')
    print('\n')
    return energy, params

def fixed_adapt(H, ref, N_e, N_qubits, params, thresh = 1e-3, in_file = 'out.dat', factor = Eh):
    system = sm.system_data(H, ref, N_e, N_qubits)
    #print("Performing VQE on pre-specified ansatz...\n")
    ansatz = []

    f = open(in_file, 'r')
    count = 0
    for line in f.readlines():
        ansatz = [of.get_sparse_operator(1j * of.ops.QubitOperator(line), system.N_qubits)] + ansatz


    E, params = ct.vqe(system.H, ansatz, system.ref, params)
    #print("VQE energy (kcal/mol):")
    cur_state = copy.copy(system.ref)
    for i in reversed(range(0, len(params))): 
        cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
    exact_overlap = cur_state.T.dot(system.ci_soln)[0].real
    print(str(E*factor)+' '+str(factor*(E-system.ci_energy))+' '+str(exact_overlap))
    #print("Error (kcal/mol):")
    #print(factor*(E-system.ci_energy))
    return E, params

def uccgsd_adapt(H, ref, N_e, N_qubits, S2, thresh = 1e-3, depth = None, out_file = 'out.dat', factor = Eh, spin_adapt = False):
    system = sm.system_data(H, ref, N_e, N_qubits)
    pool = []
    if spin_adapt == False:
        for i in range(0, N_qubits):
            for a in range(i, N_qubits):
                if (i+a)%2 == 0 and a != i:
                    pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                for j in range(i+1, N_qubits):
                    for b in range(a+1, N_qubits):
                        if i%2+j%2 == a%2+b%2 and (i,j) != (a,b) and (b>j or a>i):
                            pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
 
    elif spin_adapt == True:
       M = int(N_qubits/2)
       for i in range(0, M):
           for a in range(i, M):
               if i!= a:
                   pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
               for j in range(i, M):
                   for b in range(a, M):
                       if (i, j) != (a, b) and (i<a or j<b):
                           if i == j and a == b:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                           elif i == j:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                           elif a == b:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                           else: 
                               pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
    print('Operators:')
    print(len(pool))
    prepool = [of.transforms.get_sparse_operator(i, n_qubits = N_qubits).real for i in pool]
    
    jw_pool = [] 
    for i in prepool:
        op = i - i.T
        jw_pool.append(op)


    print("SCF Energy:")
    print(system.hf_energy)
    print("FCI Energy:")
    print(system.ci_energy)
    f = open(out_file, 'w')
    cur_state = system.ref
    ansatz = []
    params = []
    Done = False
    iteration = 0
    print('\n')
    print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Energy (kcal/mol)', 'Error (kcal/mol)', 'Last del (a.u.)', 'Newest Operator', '<S^2>'))
    while Done == False:
        iteration += 1
        max_grad = 0
        new_op = None
        for i in range(len(jw_pool)):
            op = copy.copy(jw_pool[i])
            grad = 2*abs(cur_state.T.dot(system.H).dot(op).dot(cur_state)[0,0].real)
            if grad > max_grad:
                max_grad = copy.copy(grad)
                new_op = copy.copy(op)
                new_idx = copy.copy(i)
        if (max_grad < thresh and depth == None) or (depth != None and depth == iteration-1):
            Done = True 
        else:
            ansatz = [new_op] + ansatz
            params = [0] + params
            energy, params = ct.vqe(system.H, ansatz, system.ref, params)
            cur_state = copy.copy(system.ref)
            for i in reversed(range(0, len(params))): 
                cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
            S2_state = cur_state.T.dot(S2).dot(cur_state)[0,0].real
            print('-'*105)
            print('{:<20}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20}|{:<20}|'.format(iteration, factor*energy, factor*(energy-system.ci_energy), max_grad, new_idx, S2_state))
            print(pool[new_idx])
            #print(str(factor*energy)+' '+str(factor*(energy-system.ci_energy)))
            f.write(str(pool[new_idx]).replace('\n','')+'\n')
    print('\n')
    return energy, params



def fixed_fermionic(H, ref, N_e, N_qubits, params, in_file = 'out.dat', factor = Eh, spin_adapt = False):
    system = sm.system_data(H, ref, N_e, N_qubits)
    cur_state = copy.copy(ref)
    pool = []
    Done = False
    iteration = 0
    f = open(in_file, 'r')
    for line in f.readlines():
        pool = [of.FermionOperator(line)] + pool
    prepool = [of.transforms.get_sparse_operator(i, n_qubits = N_qubits).real for i in pool]
    jw_pool = [] 
    for i in prepool:
        op = i - i.T
        jw_pool.append(op)
    ansatz = jw_pool
    E, params = ct.vqe(H, ansatz, ref, list(params))
    #print("VQE energy (kcal/mol):")
    cur_state = copy.copy(ref)
    for i in reversed(range(0, len(params))): 
        cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
    exact_overlap = cur_state.T.dot(system.ci_soln)[0].real
    print(str(E*factor)+' '+str(factor*(E-system.ci_energy))+' '+str(exact_overlap))
    return E, params

def uccsd_adapt(H, ref, N_e, N_qubits, S2, thresh = 1e-3, depth = None, out_file = 'out.dat', factor = Eh, spin_adapt = False):
    system = sm.system_data(H, ref, N_e, N_qubits)
    pool = []
    if spin_adapt == False:
        for i in range(0, N_e):
            for a in range(N_e, N_qubits):
                if (i+a)%2 == 0:
                    pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                for j in range(i+1, N_e):
                    for b in range(a+1, N_qubits):
                        if i%2+j%2 == a%2+b%2:
                            pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
 
    elif spin_adapt == True:
       M = int(N_qubits/2)
       N = int(N_e/2)
       for i in range(0, N):
           for a in range(N, M):
               pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
               for j in range(i, N):
                   for b in range(a, M):
                       if (i, j) != (a, b):
                           if i == j and a == b:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                           elif i == j:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                           elif a == b:
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                           else: 
                               pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
    print('Operators:')
    print(len(pool))
    prepool = [of.transforms.get_sparse_operator(i, n_qubits = N_qubits).real for i in pool]
    
    jw_pool = [] 
    for i in prepool:
        op = i - i.T
        jw_pool.append(op)


    print("SCF Energy:")
    print(system.hf_energy)
    print("FCI Energy:")
    print(system.ci_energy)
    f = open(out_file, 'w')
    cur_state = system.ref
    ansatz = []
    params = []
    Done = False
    iteration = 0
    print('\n')
    print('{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|{:<20}|'.format('Iteration', 'Energy (kcal/mol)', 'Error (kcal/mol)', 'Last del (a.u.)', 'Newest Operator', '<S^2>'))
    while Done == False:
        iteration += 1
        max_grad = 0
        new_op = None
        for i in range(len(jw_pool)):
            op = copy.copy(jw_pool[i])
            grad = 2*abs(cur_state.T.dot(system.H).dot(op).dot(cur_state)[0,0].real)
            if grad > max_grad:
                max_grad = copy.copy(grad)
                new_op = copy.copy(op)
                new_idx = copy.copy(i)
        if (max_grad < thresh and depth == None) or (depth != None and depth == iteration-1):
            Done = True 
        else:
            ansatz = [new_op] + ansatz
            params = [0] + params
            energy, params = ct.vqe(system.H, ansatz, system.ref, params)
            cur_state = copy.deepcopy(system.ref)
            for i in reversed(range(0, len(params))): 
                cur_state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], cur_state)
            S2_state = cur_state.T.dot(S2).dot(cur_state)[0,0].real
            print('-'*105)
            print('{:<20}|{:<20.12}|{:<20.12}|{:<20.12}|{:<20}|{:<20}|'.format(iteration, factor*energy, factor*(energy-system.ci_energy), max_grad, new_idx, S2_state))
            print(pool[new_idx])
            #print(str(factor*energy)+' '+str(factor*(energy-system.ci_energy)))
            f.write(str(pool[new_idx]).replace('\n','')+'\n')

    print('\n')
    return energy, params

if __name__ == "__main__":
    geometry = [('H', (0,0,0)), ('H', (0,0,.74))]
    basis = 'sto-3g'
    multiplicity = 1
    system = sm.system_data(geometry, basis, multiplicity)
    pool = system.full_qubit_pool()

