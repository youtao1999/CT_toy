'''Corrected/debugged functions'''

import numpy as np
from numba import jit
from haining_correct_functions import adder_cpu
from functools import partial

def op_dict_tao(L):
    return {
        ('C',): partial(apply_control_map_global, L=L),
        ('B',): partial(apply_bernoulli_map, L=L),
        (f'P{L-1}', 0): partial(apply_projection_to_state, L=L, pos=L-1, target=0),
        (f'P{L-2}', 0): partial(apply_projection_to_state, L=L, pos=L-2, target=0),
        (f'P{L-1}', 1): partial(apply_projection_to_state, L=L, pos=L-1, target=1),
        (f'P{L-2}', 1): partial(apply_projection_to_state, L=L, pos=L-2, target=1)
    }

def step_evolution(state, L, p_ctrl, p_proj):
    '''
    Parameters:
    state (np.ndarray): The state vector in the computational basis.
    L (int): The number of qubits.
    p_ctrl (float): The probability of control.
    
    Returns:
    tuple: A tuple containing:
        - np.ndarray: The new state vector after applying the control map or the Bernoulli map.
        - str: The operator applied ('C' for control map, 'B' for Bernoulli map).
    '''

    
    rng = np.random.default_rng()
    op_dict = op_dict_tao(L)

    if rng.random() < p_ctrl:
        state = op_dict[('C',)](state)
        applied_operator = ['C']
    else:
        state = op_dict[('B',)](state)
        applied_operator = ['B']

    # After applying the Bernoulli map, there is a projective probability to apply the projective measurement to
    # the last two qubits. 

    if 'B' in applied_operator:
        if rng.random() < p_proj:
            for i, pos in enumerate([L-1, L-2]):
                prob = born_prob(state, pos, 0, L)
                if rng.random() < prob:
                    state = op_dict[(f'P{pos}', 0)](state)
                else:
                    state = op_dict[(f'P{pos}', 1)](state)
                    applied_operator.append(f'P{pos}')
    return state, applied_operator

def U(n,rng=None,size=1):
    """Calculate Haar random unitary matrix of dimension `n`. The method is based on QR decomposition of a random matrix with Gaussian entries.

    Parameters
    ----------
    n : int
        dimension of the unitary matrix
    rng : numpy.random.Generator, optional
        Random generator, by default None
    size : int, optional
        Number of unitary matrix, by default 1

    Returns
    -------
    numpy.array, shape=(size,n,n)
        Haar random unitary matrix
    """
    import numpy as np
    import scipy

    if rng is None:
        rng=np.random.default_rng(None)
    return scipy.stats.unitary_group.rvs(n,random_state=rng,size=size)

def S_tensor(vec, L, rng=None):
    """Apply scrambler gate to the last two qubits. Directly convert to tensor and apply to the last two indices.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector
    rng : numpy.random.Generator
        random number generator

    Returns
    -------
    numpy.array, 
        state vector after applying scrambler gate
    """
    if rng is None:
        rng=np.random.default_rng(None)
    U_4=U(4,rng=rng)
    vec=vec.reshape((2**(L-2),2**2)).T
    return (U_4@vec).T

def apply_bernoulli_map(state, L):
    """
    Apply the Bernoulli map to the quantum state.
    
    Parameters:
    state (np.ndarray): The state vector in the computational basis.
    L (int): The number of qubits.
    
    Returns:
    np.ndarray: The new state vector after applying the Bernoulli map.
    """
    state = apply_cyclic_leftward_shift_to_state(state, L)
    state = S_tensor(state, L, rng = None).flatten()
    return state

@jit(nopython=True)
def apply_cyclic_right_shift_to_state(state, L):
    new_state = np.empty_like(state)
    mask = (1 << L) - 1
    for i in range(2**L):
        shifted_index = ((i >> 1) | (i << (L - 1))) & mask
        new_state[shifted_index] = state[i]
    return new_state

@jit(nopython=True)
def measure_reset_rightmost_qubit(state, L):
    p0 = born_prob(state, L-1, 0, L)
    p1 = 1 - p0

    if np.random.random() < p0:
        target = 0
        state[1::2] = 0
        state[::2] /= np.sqrt(p0)
    else:
        target = 1
        state[::2] = state[1::2]  # Move odd indices to even indices
        state[1::2] = 0  # Set odd indices to zero
        state[::2] /= np.sqrt(p1)  # Normalize the state
    # this function currently does not reset the last qubit to 0
    return state, target

def apply_control_map_global(state, L):
    """
    Apply the global control map to the quantum state.
    
    Parameters:
    state (np.ndarray): The state vector in the computational basis.
    L (int): The number of qubits.
    
    Returns:
    np.ndarray: The new state vector after applying the global control map.
    """
    state, _ = measure_reset_rightmost_qubit(state, L)
    state = apply_cyclic_right_shift_to_state(state, L)
    state = adder_cpu(state, L)
    
    return state

@jit(nopython=True)
def apply_cyclic_leftward_shift_to_state(state, L):
    new_state = np.empty_like(state)
    mask = (1 << L) - 1
    for i in range(2**L):
        shifted_index = ((i << 1) | (i >> (L - 1))) & mask
        new_state[shifted_index] = state[i]
    return new_state

@jit(nopython=True)
def apply_projection_to_state(state, L, pos, target):
    mask = 1 << (L - 1 - pos)
    for i in range(len(state)):
        if ((i & mask) >> (L - 1 - pos)) == 1 - target:  # Changed condition
            state[i] = 0
    return state / np.linalg.norm(state)

# Add this function to the file
@jit(nopython=True)
def born_prob(state, pos, target, L):
    mask = 1 << (L - 1 - pos)
    prob = 0.0
    for i in range(2**L):
        if (i & mask) >> (L - 1 - pos) == target:
            prob += abs(state[i])**2
    return prob

def warmup_jit_functions(L):
    """
    Warm up all functions by calling them once with dummy data for a given system size L.

    Parameters:
    -----------
    L : int
        Number of qubits in the system
    """
    # Dummy data
    dummy_state = np.random.rand(2**L)
    dummy_state /= np.linalg.norm(dummy_state)

    # Warm up apply_cyclic_right_shift_to_state
    _ = apply_cyclic_right_shift_to_state(dummy_state.copy(), L)

    # Warm up measure_reset_rightmost_qubit
    _, _ = measure_reset_rightmost_qubit(dummy_state.copy(), L)

    # Warm up apply_cyclic_leftward_shift_to_state
    _ = apply_cyclic_leftward_shift_to_state(dummy_state.copy(), L)

    # Warm up apply_projection_to_state
    _ = apply_projection_to_state(dummy_state.copy(), L, L-1, 0)

    # Warm up born_prob
    _ = born_prob(dummy_state.copy(), L-1, 0, L)

    print(f"JIT functions warmed up for L={L}.")