import numpy as np

def tripartite_mutual_information(vec, L, n=1, threshold=1e-10, return_singular_values=True):
    """
    Compute the tripartite mutual information and return singular values for each partition.
    
    Returns:
    --------
    If return_singular_values=False:
        float: Tripartite mutual information
    If return_singular_values=True:
        tuple: (I_3, singular_values_dict) where singular_values_dict contains SVD values for each partition
    """
    if L % 4 != 0:
        raise ValueError("L must be divisible by 4")
    
    if n == 0:
        threshold = 1e-15
    else:
        threshold = 1e-10

    subregion_size = L // 4

    # Define subregions
    A = list(range(subregion_size))
    B = list(range(subregion_size, 2*subregion_size))
    C = list(range(2*subregion_size, 3*subregion_size))

    # Modified entropy calculations to capture singular values
    S_A, sv_A = von_Neumann_entropy_pure(vec, A, L, n, threshold, return_singular_values=True)
    S_B, sv_B = von_Neumann_entropy_pure(vec, B, L, n, threshold, return_singular_values=True)
    S_C, sv_C = von_Neumann_entropy_pure(vec, C, L, n, threshold, return_singular_values=True)

    S_AB, sv_AB = von_Neumann_entropy_pure(vec, A + B, L, n, threshold, return_singular_values=True)
    S_AC, sv_AC = von_Neumann_entropy_pure(vec, A + C, L, n, threshold, return_singular_values=True)
    S_BC, sv_BC = von_Neumann_entropy_pure(vec, B + C, L, n, threshold, return_singular_values=True)

    S_ABC, sv_ABC = von_Neumann_entropy_pure(vec, A + B + C, L, n, threshold, return_singular_values=True)

    # Compute tripartite mutual information
    I_3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    if return_singular_values:
        singular_values = {
            'A': sv_A,
            'B': sv_B,
            'C': sv_C,
            'AB': sv_AB,
            'AC': sv_AC,
            'BC': sv_BC,
            'ABC': sv_ABC
        }
        return I_3, singular_values
    return I_3

def von_Neumann_entropy_pure(vec, subregion, L_T, n=1, threshold=1e-10, return_singular_values=False):
    """
    Modified to use consistent matrix orientation like Haining's implementation.
    """
    vec_tensor = vec.reshape((2,) * L_T)
    subregion = list(subregion)
    not_subregion = [i for i in range(L_T) if i not in subregion]
    
    # Always use the same orientation as Haining's implementation
    vec_tensor_T = vec_tensor.transpose(np.hstack([subregion, not_subregion]))
    S = np.linalg.svd(vec_tensor_T.reshape((2**len(subregion), 2**len(not_subregion))), compute_uv=False)
    
    S_pos = np.clip(S, threshold, None)
    eigenvalues = S_pos**2

    if n == 1:
        entropy = -np.sum(np.log(eigenvalues) * eigenvalues)
        if np.isnan(entropy):
            entropy = 0
    elif n == 0:
        entropy = np.log((eigenvalues > threshold**2).sum())
    elif n == np.inf:
        entropy = -np.log(np.max(eigenvalues))
    else:
        entropy = np.log(np.sum(eigenvalues**n)) / (1-n)

    if return_singular_values:
        return entropy, S_pos
    return entropy

def half_system_entanglement_entropy(vec, L, selfaverage=True, n=1, threshold=1e-10):
    """Calculate the half-system entanglement entropy for a given state vector.

    Parameters
    ----------
    vec : np.array, shape=(2**L,) or (2,)*L
        State vector
    L : int
        Number of qubits
    selfaverage : bool, optional
        If true, average over all possible halves, by default True
    n : int, optional
        n-th Renyi entropy, by default 1
    threshold : float, optional
        Threshold for singular values, by default 1e-10

    Returns
    -------
    float
        Half-system entanglement entropy
    """
    if selfaverage:
        return np.mean([von_Neumann_entropy_pure(vec, np.arange(i, i+L//2), L, n, threshold) for i in range(L//2)])
    else:
        return von_Neumann_entropy_pure(vec, np.arange(L//2), L, n, threshold)

def calculate_afm_neel_order(state, L):
    """
    Calculate the antiferromagnetic (AFM) Néel order parameter for a given quantum state.
    
    Parameters:
    state (np.ndarray): The state vector in the computational basis.
    L (int): The number of qubits.
    
    Returns:
    float: The average value of the AFM Néel order parameter.
    """
    O_AFM = 0.0
    
    for i in range(L):
        Z_i = pauli_z_operator(L, i)
        Z_ip1 = pauli_z_operator(L, (i + 1) % L)
        
        ZiZi1 = np.dot(Z_i, Z_ip1)
        
        expectation_value = np.vdot(state, np.dot(ZiZi1, state))
        O_AFM += expectation_value.real
    
    O_AFM = -O_AFM / L
    
    return O_AFM

def pauli_z_operator(L, qubit_index):
    """
    Constructs the Pauli-Z operator acting on the specified qubit in a system of L qubits.
    
    Parameters:
    L (int): The number of qubits.
    qubit_index (int): The index of the qubit on which the Pauli-Z operator acts.
    
    Returns:
    np.ndarray: The (2^L x 2^L) matrix representing the Pauli-Z operator on the specified qubit.
    """
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    operator = 1
    for i in range(L):
        operator = np.kron(operator, Z if i == qubit_index else I)
    
    return operator