import numpy as np
from typing import Union, Optional, List

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
    
def Haar_state(L,ensemble,rng,k=1):
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

# CZ gate matrix
CZ_mat = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0+0j]
], dtype=complex)

def Rx(theta: float) -> np.ndarray:
    """Create Rx gate"""
    return np.array([
        [np.cos(theta / 2), -1j*np.sin(theta / 2)],
        [-1j*np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    """Create Rz gate"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)

def U_simp(CZ: bool, rng: Optional[Union[int, np.random.Generator]] = None, 
           theta: Optional[List[float]] = None) -> np.ndarray:
    """Create a simplified Haar random unitary.
    The unitary is defined as 
    ---Rx(θ1)---Rz(θ2)---Rx(θ3)---CZ---Rx(θ7)---Rz(θ8)---Rx(θ9)---
                                             |
    ---Rx(θ4)---Rz(θ5)---Rx(θ6)---CZ---Rx(θ10)---Rz(θ11)---Rx(θ12)---
    If `CZ` is True, applied a CZ gate, otherwise, it is skipped.
    Here, 12 θ's are independently chosen as a random number in [0,2pi), and Rx and Rz are single qubit rotation gates along the x and z axes, respectively.
    For simplicity, we denote θ as θ[0], θ[1], ..., θ[5] on the top qubit, and θ[6], θ[7], ..., θ[11] on the bottom qubit.
    
    Parameters:
    -----------
    CZ : bool
        Whether to apply CZ gate
    rng : int or np.random.Generator, optional
        Random number generator or seed
    theta : List[float], optional
        List of 12 rotation angles. If None, random angles are generated.
        
    Returns:
    --------
    np.ndarray
        4x4 unitary matrix
    """
    if theta is None:
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        theta = rng.random(12) * 2 * np.pi
    
    # Layer 1 (Left)
    U1 = np.kron(Rx(theta[0]), Rx(theta[3]))
    # Layer 2
    U2 = np.kron(Rz(theta[1]), Rz(theta[4]))
    # Layer 3
    U3 = np.kron(Rx(theta[2]), Rx(theta[5]))
    # Layer 4 (CZ)
    U4 = CZ_mat if CZ else np.eye(4, dtype=complex)
    # Layer 5
    U5 = np.kron(Rx(theta[6]), Rx(theta[9]))
    # Layer 6
    U6 = np.kron(Rz(theta[7]), Rz(theta[10]))
    # Layer 7 (Right)
    U7 = np.kron(Rx(theta[8]), Rx(theta[11]))

    # Combine layers (matrix multiplication from right to left)
    U_final = U7 @ U6 @ U5 @ U4 @ U3 @ U2 @ U1

    # Transpose is important to ensure that is consistent with Qiskit's convention (which should also be the state vector)
    return U_final.T