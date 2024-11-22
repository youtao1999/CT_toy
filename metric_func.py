import numpy as np

def tripartite_mutual_information_tao(vec, L, n=1, threshold=1e-10):
    """
    Compute the tripartite mutual information for a given state vector.

    Parameters:
    -----------
    vec : np.array, shape=(2**L,) or (2,)*L
        State vector
    L : int
        Total number of qubits (must be divisible by 4)
    n : int, optional
        n-th Renyi entropy, by default 1
    threshold : float, optional
        Threshold for singular values, by default 1e-10

    Returns:
    --------
    float
        Tripartite mutual information
    """
    if L % 4 != 0:
        raise ValueError("L must be divisible by 4")

    subregion_size = L // 4

    # Define subregions
    A = list(range(subregion_size))
    B = list(range(subregion_size, 2*subregion_size))
    C = list(range(2*subregion_size, 3*subregion_size))

    # Compute entropies
    S_A = von_Neumann_entropy_pure(vec, A, L, n, threshold)
    S_B = von_Neumann_entropy_pure(vec, B, L, n, threshold)
    S_C = von_Neumann_entropy_pure(vec, C, L, n, threshold)

    S_AB = von_Neumann_entropy_pure(vec, A + B, L, n, threshold)
    S_AC = von_Neumann_entropy_pure(vec, A + C, L, n, threshold)
    S_BC = von_Neumann_entropy_pure(vec, B + C, L, n, threshold)

    S_ABC = von_Neumann_entropy_pure(vec, A + B + C, L, n, threshold)

    # Compute tripartite mutual information
    I_3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    return I_3

def von_Neumann_entropy_pure(vec, subregion, L_T, n=1, threshold=1e-10):
    """Calculate the von Neumann entropy of a pure state for a given subregion.

    Parameters
    ----------
    vec : np.array, shape=(2**L_T,) or (2,)*L_T
        State vector
    subregion : list of int or np.array
        The spatial subregion to calculate the von Neumann entropy
    L_T : int
        Total number of qubits
    n : int, optional
        n-th Renyi entropy, by default 1
    threshold : float, optional
        Threshold to clip the singular value, by default 1e-10

    Returns
    -------
    float
        Von Neumann entropy of the subregion
    """
    vec_tensor = vec.reshape((2,) * L_T)
    subregion = list(subregion)
    not_subregion = [i for i in range(L_T) if i not in subregion]
    vec_tensor_T = vec_tensor.transpose(np.hstack([subregion, not_subregion]))
    S = np.linalg.svd(vec_tensor_T.reshape((2**len(subregion), 2**len(not_subregion))), compute_uv=False)
    S_pos = np.clip(S, threshold, None)
    S_pos2 = S_pos**2

    if n == 1:
        entropy = -np.sum(np.log(S_pos2) * S_pos2)
        if np.isnan(entropy):
            entropy = 0
        return entropy
    elif n == 0:
        return np.log((S_pos2 > threshold**2).sum())
    elif n == np.inf:
        return -np.log(np.max(S_pos2))
    else:
        return np.log(np.sum(S_pos2**n)) / (1-n)

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