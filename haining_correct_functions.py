'''This is a collection of Haining's correct functions'''

import numpy as np
from functools import partial
from fractions import Fraction
import scipy.sparse as sp

xj=frozenset([Fraction(1,3),Fraction(2,3)])
feedback=True

def Haar_state(L,ensemble,rng=None,k=1):
    """Generate `k` orthogonal Haar random states, using the method in https://quantumcomputing.stackexchange.com/questions/15754/confusion-about-the-output-distribution-of-haar-random-quantum-states

    Parameters
    ----------
    L : int
        Length of the system, gives 2**L dimension of the Hilbert space
    ensemble : int
        Ensemble size
    rng : np.random.Generator, optional
        Random generator, by default None
    k : int, optional, {0,1}
        Number of orthrogonal Haar random state, by default 1

    Returns
    -------
    np.array, shape=(2**L,k,ensemble)
        The orthogonal `k` Haar random states
    """
    if rng is None:
        rng=np.random.default_rng(None)
    assert k in [1,2], f'k ({k}) is not 1 or 2'
    state=rng.normal(size=(2**L,2,k,ensemble)) # (wf, re/im, k, ensemble)
    z=state[:,0,:,:]+1j*state[:,1,:,:] # (wf, k, ensemble)
    norm=np.sqrt((np.abs(z[:,0,:])**2).sum(axis=0)) # (ensemble,)
    z[:,0,:]/=norm
    if k==2:
        z[:,1,:]-=(z[:,0,:].conj()*z[:,1,:]).sum(axis=0)*z[:,0,:]
        norm=np.sqrt((np.abs(z[:,1,:])**2).sum(axis=0))
        z[:,1,:]/=norm
    return z


def T_tensor(vec, L, left=True):
    """Left shift (times 2) and right shift (divided by 2). Directly transpose the index of tensor.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector
    left : bool, optional
        if True, left shift, else, right shift, by default True

    Returns
    -------
    numpy.array, shape=(2,)*L_T
        state vector after shift
    """
    if vec.ndim!=L:
        vec=vec.reshape((2,)*L)
    if left:
        idx_list_2=list(range(1,L))+[0]
    else:
        idx_list_2=[L-1]+list(range(L-1))
    return vec.transpose(idx_list_2)

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
    U_4=U(4,rng=rng)
    vec=vec.reshape((2**(L-2),2**2)).T
    return (U_4@vec).T

def XL_tensor(vec, L):
    """Flip the last qubit of the state vector `vec` (excluding the ancilla qubit). Directly swap 0 and 1 index.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector

    Returns
    -------
    numpy.array, shape=(2,)*L_T
        state vector after applying sigma_x to the last qubit
    """
    if vec.ndim!=L:
        vec=vec.reshape((2,)*L)
    vec=np.roll(vec,1,axis=L-1)
    return vec

def P_tensor(vec,n,pos, L):
    if vec.ndim!=L:
        vec=vec.reshape((2,)*L)
    for n_i,pos_i in zip(n,pos):
        idx_list=[slice(None)]*L
        idx_list[pos_i]=1-n_i
        vec[tuple(idx_list)]=0
    return vec
        
def normalize(vec):
    """Normalize the state vector `vec`. If the norm is zero, leave it unchanged.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector

    Returns
    -------
    numpy.array, shape=(2**L_T,) or (2,)*L_T
        normalized state vector
    """
    if vec.ndim!=1:
        vec=vec.flatten()
    norm2=(vec.conj().T@vec).real
    if norm2 > 0:
        return vec/np.sqrt(norm2)
    else:
        return vec

def R_tensor(vec,n,pos, L):
    vec=P_tensor(vec,n,pos, L)
    # if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]),frozenset([0]),frozenset([1]),frozenset([-1])]:
    if len(n)==1:
        # projection on the last bits
        if n[0]==1:
            if feedback:
                vec=XL_tensor(vec, L)
    # elif self.xj==frozenset([Fraction(1,3),Fraction(-1,3)]):
    elif len(n)==2:
        # projection on the first and last bits
        if n[0]^n[1]==0:
            if feedback:
                vec=XL_tensor(vec, L)

    vec=normalize(vec)
    return vec


def Bernoulli_map(vec, L, rng = None):
    """Apply Bernoulli map to the state vector `vec`. The Bernoulli map contains a left shift and a scrambler gate (U(4) Haar random unitary matrix) to the last two qubits.

    Parameters
    ----------
    vec : np.array, shape=(2**L_T,) or (2,)*L_T
        state vector

    Returns
    -------
    np.array, shape=(2**L_T,)
        state vector after Bernoulli map
    """
    vec=T_tensor(vec, L, left=True)
    vec=S_tensor(vec, L, rng=rng)
    return vec.flatten()

def dec2bin(x,L,base=2):
    """convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 as integer part is truncated. Example, 1/3 is 010101...

    Parameters
    ----------
    x : float, 0<=x<1
        float number to be converted
    L : int
        length of the binary form

    Returns
    -------
    numpy.array, shape=(L,)
        array of binary form
    """
    assert 0<=x<1, f'{x} is not in [0,1)'
    if base == 2:
        return int(x*(1<<L))
    else:
        return int(x*(base**L))
    
def adder_cpu(vec,L):

    if vec.ndim > 1:
        vec = vec.flatten() # flatten the tensor to 1D array

    vec=adder(L)@vec
    
    return vec

def adder(L):
    """Calculate the adder matrix, which is the shuffle of the state basis. Note that this is not a full adder, which assumes the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).

    Returns
    -------
    numpy.array, shape=(2**L,2**L)
        adder matrix
    """
    int_1_6=dec2bin(Fraction(1,6), L)|1
    int_1_3=dec2bin(Fraction(1,3), L)

    old_idx=np.arange(2**(L-1))
    adder_idx=np.array([int_1_6]*2**(L-2)+[int_1_3]*2**(L-2))
    new_idx=(old_idx+adder_idx)
    ones=np.ones(2**(L-1))

    # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 
    mask_1=(new_idx&(1<<L-1) == (1<<L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
    mask_2=(new_idx&(1<<L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
    new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
    # print(old_idx,new_idx)
    return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**L,2**L))
    
def control_map(vec,n,pos,L):
    """Apply control map the state vector `vec`. The control map is a combination of projection, right shift and an adder. The projection is applied to the last qubit (if the outcome is 1, sigma_x is applied to flip the last qubit); the right shift is applied to all qubits in the system (excluding the ancilla qubit); the adder is the shuffle of the state basis. 

    Parameters
    ----------
    vec : np.array, shape=(2**L_T,) or (2,)*L_T
        state vector
    n : int, {0,1}
        the outcome of the measurement of the last bit, 0 or 1. 

    Returns
    -------
    np.array, shape=(2**L_T,)
        state vector after the control map
    """
    
    vec=R_tensor(vec,n,pos,L)

    # right shift 
    vec=T_tensor(vec,L,left=False)
    # Adder
    if vec.ndim >1 :
        vec=vec.flatten()
    vec=adder_cpu(vec,L=L)
    
    return vec

def projection_map(vec,pos,n,L):
    """Projective measurement to `pos`-th qubit with outcome of `n` (0-index). `pos=L-1` is the last bit.

    Parameters
    ----------
    vec : np.array, shape=(2**L_T,) or (2,)*L_T
        state vector
    pos : int, {0,1,...,L-1}
        position to apply the projection
    n : int, {0,1}
        the outcome of the measurement, 0 or 1

    Returns
    -------
    np.array, shape=(2**L_T,)
        state vector after projection
    """
    vec=P_tensor(vec,n,pos,L)
    vec=normalize(vec)
    return vec

def _initialize_op(L):
    """Initialize the operators in the circuit, including the control, projection, and Bernoulli map. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

    Returns
    -------
    dict of operators
        possible operators in the circuit
    """ 
    return {("C",0):partial(control_map,pos=[L-1],n=[0],L=L),
            ("C",1):partial(control_map,pos=[L-1],n=[1],L=L),
            ("C",0,0):partial(control_map,pos=[0,L-1],n=[0,0],L=L),
            ("C",0,1):partial(control_map,pos=[0,L-1],n=[0,1],L=L),
            ("C",1,0):partial(control_map,pos=[0,L-1],n=[1,0],L=L),
            ("C",1,1):partial(control_map,pos=[0,L-1],n=[1,1],L=L),
            (f"P{L-1}",0):partial(projection_map,pos=[L-1],n=[0],L=L),
            (f"P{L-1}",1):partial(projection_map,pos=[L-1],n=[1],L=L),
            (f"P{L-2}",0):partial(projection_map,pos=[L-2],n=[0],L=L),
            (f"P{L-2}",1):partial(projection_map,pos=[L-2],n=[1],L=L),
            ("B",):partial(Bernoulli_map, L=L, rng = None),
            ("I",):lambda x:x
            }

def inner_prob(vec,pos,n_list, L):
    """Calculate the probability of measuring 0 at position `pos` for the state vector `vec`. First, convert the vector to tensor (2,2,..), take about the specific `pos`-th index, and flatten to calculate the inner product.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector
    pos : int, {0,1,...,L-1}
        position to apply to calculate the probability of measuring 0

    Returns
    -------
    float, 0<=inner_prod<=1
        probability of measuring 0 at position `pos`
    """
    if vec.ndim != (2,)*L:
        vec=vec.reshape((2,)*L)
    idx_list=np.array([slice(None)]*L)
    idx_list[pos]=n_list
    vec_0=vec[tuple(idx_list)].flatten()
    inner_prod=vec_0.conj()@vec_0
    inner_prod=np.clip(inner_prod.real,0,1)
    return inner_prod

def ZZ_tensor(vec, L):
    """Calculate the order parameter for Neel state. The order parameter is defined as \sum_{i=0..L-1} <Z_iZ_{i+1}>, where Z_i is the Pauli Z matrix at site i.

    Parameters
    ----------
    vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
        state vector

    Returns
    -------
    float
        order parameter for Neel state
    """
    if vec.ndim != (2,)*L:
        vec=vec.reshape((2,)*L)
    rs=0
    for i in range(L):
        for zi in range(2):
            for zj in range(2):
                # idx_list=[slice(None)]*self.L_T
                # idx_list[i],idx_list[(i+1)%self.L]=zi,zj
                inner_prod=inner_prob(vec, [i,(i+1)%L],[zi,zj], L)
                exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                # vec_i=vec[tuple(idx_list)].flatten()
                # rs+=vec_i.conj()@vec_i*exp
                rs+=inner_prod*exp
    return -rs/L

def random_control(vec, p_ctrl, p_proj, L):
    """The competition between chaotic random unitary, control map and projection, where the projection can only be applied after the unitary. The probability of control is `p_ctrl`, and the probability of projection is `p_proj`.

    Parameters
    ----------
    p_ctrl : float, 0<=p_ctrl<=1
        probability of control
    p_proj : float, 0<=p_proj<=1
        probability of projection
    """     
    op_list=_initialize_op(L)
    op_l=[]
    if np.random.rand()<=p_ctrl:
        # control map
        p_0=inner_prob(vec, pos=[L-1],n_list=[0], L=L)
        op=('C',0) if np.random.rand()<=p_0 else ('C',1)
    else:
        op = ('B',)

    op_l.append(op)
    vec=op_list[op](vec)

    if "B" in op:
        if np.random.rand()<=p_proj:
            for idx,pos in enumerate([L-1,L-2]):
                # projection map
                p_2=(inner_prob(vec, pos=[pos],n_list=[0],L=L))
                op_2=(f"P{pos}",0) if np.random.rand()<p_2 else (f"P{pos}",1)
                vec=op_list[op_2](vec)
                op_l.append(op_2)
    return vec, op_l

def tripartite_mutual_information_haining(vec,subregion_A,subregion_B, subregion_C, L,selfaverage=False,n=1,threshold=1e-10):
    """Calculate tripartite entanglement entropy. The tripartite entanglement entropy is defined as S_A+S_B+S_C-S_AB-S_AC-S_BC+S_ABC, where S_A is the von Neumann entropy of subregion A, S_AB is the von Neumann entropy of subregion A and B, etc. The system size `L` should be a divided by 4 such that the subregion A, B and C are of the same size.

    Parameters
    ----------
    subregion_A : list of int or np.array
        subregion A
    subregion_B : list of int or np.array
        subregion B
    subregion_C : list of int or np.array
        subregion C
    selfaverage : bool, optional
        if true, average over all possible partitions, by default False
    vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
        state vector, by default None

    Returns
    -------
    float
        Tripartite entanglement entropy
    """
    if selfaverage:
        return np.mean([tripartite_mutual_information_haining((subregion_A+shift)%L,(subregion_B+shift)%L,(subregion_C+shift)%L,selfaverage=False) for shift in range(len(subregion_A))])
    else:
        S_A=von_Neumann_entropy_pure(vec,subregion_A, L, n=n,threshold=threshold)
        S_B=von_Neumann_entropy_pure(vec, subregion_B, L, n=n,threshold=threshold)
        S_C=von_Neumann_entropy_pure(vec,subregion_C, L, n=n,threshold=threshold)
        S_AB=von_Neumann_entropy_pure(vec, np.concatenate([subregion_A,subregion_B]), L, n=n,threshold=threshold)
        S_AC=von_Neumann_entropy_pure(vec, np.concatenate([subregion_A,subregion_C]), L, n=n,threshold=threshold)
        S_BC=von_Neumann_entropy_pure(vec, np.concatenate([subregion_B,subregion_C]), L, n=n,threshold=threshold)
        S_ABC=von_Neumann_entropy_pure(vec, np.concatenate([subregion_A,subregion_B,subregion_C]), L, n=n,threshold=threshold)
        return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC
    
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
        return -np.sum(np.log(S_pos2) * S_pos2)
    elif n == 0:
        return np.log((S_pos2 > threshold**2).sum())
    elif n == np.inf:
        return -np.log(np.max(S_pos2))
    else:
        return np.log(np.sum(S_pos2**n)) / (1-n)
