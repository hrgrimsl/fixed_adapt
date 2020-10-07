#Gives molecule class w/ methods for deriving pools, etc.
import openfermion as of
import openfermionpsi4 as ofp
import scipy
import re
import psi4
import numpy as np
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
        self.ci_energy = np.linalg.eigh(H.toarray())[0][0]
        self.hf_energy = self.ref.T.dot(self.H).dot(self.ref)[0,0] 
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

        
            
