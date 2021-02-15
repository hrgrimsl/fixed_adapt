#Gives molecule class w/ methods for deriving pools, etc.
import openfermion as of
import openfermionpsi4 as ofp
import scipy
import re
import psi4
import numpy as np
import copy

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
from opt_einsum import contract
class system_data:
    def __init__(self, H, ref, N_e, N_qubits):
        self.N_qubits = N_qubits
        self.ref = ref
        self.H = H
        self.N_e = N_e
        self.pool = []
        #energies, wfns = np.linalg.eigh(H.toarray())
        #self.ci_energy = energies[0]
        #self.ci_soln = wfns[:,0]
        #self.hf_energy = self.ref.T.dot(self.H).dot(self.ref)[0,0]                

    def recursive_qubit_op(self, op, qubit_index):
        if qubit_index == self.N_qubits-1:
            return [op, op + ' X' + str(qubit_index), op + ' Y' + str(qubit_index), op + ' Z' + str(qubit_index)]
        else:
            return self.recursive_qubit_op(op, qubit_index+1) + self.recursive_qubit_op(op + ' X' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Y' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Z' + str(qubit_index), qubit_index+1)
        
    def choose_next(self, set_of_lists, cur_list, k):
        if len(cur_list) == k:
            set_of_lists += [cur_list]
        else:
            if len(cur_list) == 0:
                floor = 0
            else:
                floor = max(cur_list)+1
            for i in range(floor, self.N_qubits):
                self.choose_next(set_of_lists, cur_list+[i], k)

    def choose_paulis(self, paulis, sub_list, k):
        if len(sub_list) == k:
            paulis += [sub_list]
        else:
            for let in ['X', 'Y', 'Z']:
                self.choose_paulis(paulis, sub_list + [let], k)
                        

    def full_qubit_pool(self):
        pool = []
        pool += self.recursive_qubit_op("", 0)
        assert(len(pool) == 4**self.N_qubits)
        self.pool += [i for i in pool if len(re.findall("Y", i))%2 == 1]
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.N_qubits) for i in self.pool]

    def k_qubit_pool(self, k):
        indices = []
        self.choose_next(indices, [], k)
        paulis = []
        self.choose_paulis(paulis, [], k)
        pool = []
        for i in indices:
            for j in paulis:
                string = str(j[0])+str(i[0])
                for l in range(1, len(i)):
                    string += " "+str(j[l])+str(i[l])
                pool.append(string)
        pool = [i for i in pool if len(re.findall("Y", i))%2 == 1]
        self.pool += pool 
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.N_qubits) for i in pool]

    def uccsd_pool(self, spin_adapt = False):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
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
                   v_pool.append(f"{i}->{a}")
                   for j in range(i, N):
                       for b in range(a, M):
                           if (i, j) != (a, b):

                               if i == j and a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))

                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif i == j:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               else:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)
             coeff = 0
             for t in op.terms:                 
                 coeff_t = op.terms[t]
                 coeff += coeff_t * coeff_t
             op = op/np.sqrt(coeff)
             pool[i] = copy.copy(op)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool 
        return jw_pool, v_pool
