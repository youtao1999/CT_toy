def monitor(func,):
    def wrapper(self, *args, **kwargs):
        
        result = func(self, *args, **kwargs)
        if self.monitor is not False and func.__name__ in self.monitor['op']:
            if 'args' not in self.monitor:
                self.monitor['args']={}
            if self.monitor['metric'] == 'SA':
                delta=self.von_Neumann_entropy_pure([self.L],vec=result,**self.monitor['args'])
            elif self.monitor['metric'] == 'EE':
                delta=self.half_system_entanglement_entropy(vec=result,**self.monitor['args'])
            elif self.monitor['metric'] == 'TMI':
                delta=self.tripartite_mutual_information(vec=result,subregion_A=range(self.L//4),subregion_B=range(self.L//4,self.L//4*2),subregion_C=range(self.L//4*2,self.L//4*3),**self.monitor['args'])
            else:
                raise ValueError(f'Invalid monitor {self.monitor}')
            if not hasattr(self, 'EE_history'):
                self.EE_history=[delta]
            else:
                self.EE_history.append(delta)
        return result
    return wrapper

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
    import numpy as np
    
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

from functools import lru_cache
@lru_cache(maxsize=None)
def bin_pad(x,L):
    """Convert an integer `x` to binary form with length `L`, with 0 padding to the left.

    Parameters
    ----------
    x : int
        integer in decimal form
    L : int
        length of the binary form

    Returns
    -------
    str
        bitstring of length `L` in binary form
    """
    return (bin(x)[2:]).rjust(L,'0')


import numpy as np
from fractions import Fraction
from functools import partial, lru_cache
import scipy.sparse as sp
class CT_quantum:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,x0=None,xj=frozenset([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False,normalization=True,debug=False,monitor=False,feedback=True,add_x=False,fixed_point=[]):
        """Initialize the quantum circuit for the control transition (CT)

        Parameters
        ----------
        L : int
            the length of the physical system, excluding ancilla qubit
        store_vec : bool, optional
            store the history of state vector , by default False
        store_op : bool, optional
            store the history of operators applied to the circuit, by default False
        store_prob : bool, optional
            store the history of each probability at projective measurement, by default False
        seed : int, optional
            (1) the random seed in the measurement outcome; (2) if `seed_vec` and `seed_C` is None, this random seed also applied to initial state vector and circuit, by default None
        seed_vec : int, optional
            the random seed in the state vector, by default None
        seed_C : int, optional
            the random seed in the circuit, by default None
        x0 : float|Fraction(a,b), optional
            the initial state represented by a float number within [0,1), by default None, if None, the initial state is Haar random state
        xj : frozenset of Fractions, optional
            the frozenset of attractors using Fractions, by default frozenset([Fraction(1,3),Fraction(2,3)])
        _eps : float, optional
            error threshold, by default 1e-10
        ancilla : bool, optional
            if true, an ancilla qubit is maximally entangled to the system , by default False
        normalization : bool, optional
            if false, the state vector will not be normalized after the projective measurement, by default True
        debug : bool, optional
            if true, all assertions will be checked, by default False
        monitor: bool, optional
            if true, half-system EE will be calculated after each operation
        feedback: bool, optional
            if false, the feedback in the measurement is turned off
        """        
        self.L=L 
        self.L_T=L+1 if ancilla else L # the length combining the physical system and the ancilla qubit
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.monitor=monitor 
        self.rng=np.random.default_rng(seed)
        self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.op_history=[] # store the history of operators applied to the circuit
        self.prob_history=[]  # store the history of each probability at projective measurement
        self.ancilla=ancilla
        self.debug=debug
        self.vec=self._initialize_vector() # initialize the state vector
        self.vec_history=[self.vec] # store the history of state vector
        self._eps=_eps
        self.xj=frozenset(xj)
        self.op_list=self._initialize_op() # initialize operators in the circuit
        self.normalization=normalization
        self.feedback=feedback   
        self.add_x=add_x
        self.fixed_point=fixed_point
        
    @monitor
    def _initialize_vector(self):
        """Save the state vector using an array of 2**L_T, if ancilla qubit, the last qubit is the ancilla qubit.
        If ancilla is False, the initial state is either the Haar random state or the state represented by `x0` (if specified)
        If ancilla is True, the initial state is the maximally entangled state between the system and the ancilla qubit, where the system is Haar random state. Namely, |psi>=1/sqrt(2)(|psi_0>|0>+|psi_1>|1>), where |psi_0> and |psi_1> are two orthogonal Haar random states. 

        Returns
        -------
        np.array, shape=(2**L_T,) or (2,)*L_T
            the initial state vector
        """
        if not self.ancilla:
            if self.x0 is not None:
                vec_int=dec2bin(self.x0,self.L)
                vec=np.zeros((2**self.L,),dtype=complex)
                vec[vec_int]=1
            else:
                vec=Haar_state(self.L, 1,rng=self.rng_vec,k=1).flatten()
        else:
            vec=Haar_state(self.L, 1,rng=self.rng_vec,k=2).flatten()/np.sqrt(2)
        return vec.reshape((2,)*self.L_T)
    
    def _initialize_op(self):
        """Initialize the operators in the circuit, including the control, projection, and Bernoulli map. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

        Returns
        -------
        dict of operators
            possible operators in the circuit
        """ 
        return {("C",0):partial(self.control_map,pos=[self.L-1],n=[0]),
                ("C",1):partial(self.control_map,pos=[self.L-1],n=[1]),
                ("C",0,0):partial(self.control_map,pos=[0,self.L-1],n=[0,0]),
                ("C",0,1):partial(self.control_map,pos=[0,self.L-1],n=[0,1]),
                ("C",1,0):partial(self.control_map,pos=[0,self.L-1],n=[1,0]),
                ("C",1,1):partial(self.control_map,pos=[0,self.L-1],n=[1,1]),
                (f"P{self.L-1}",0):partial(self.projection_map,pos=[self.L-1],n=[0]),
                (f"P{self.L-1}",1):partial(self.projection_map,pos=[self.L-1],n=[1]),
                (f"P{self.L-2}",0):partial(self.projection_map,pos=[self.L-2],n=[0]),
                (f"P{self.L-2}",1):partial(self.projection_map,pos=[self.L-2],n=[1]),
                ("B",):self.Bernoulli_map,
                ("I",):lambda x:x
                }

    def Bernoulli_map(self,vec):
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
        vec=self.T_tensor(vec,left=True)
        vec=self.S_tensor(vec,rng=self.rng_C)
        return vec
    
    def control_map(self,vec,n,pos):
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
        
        vec=self.R_tensor(vec,n,pos)

        # right shift 
        vec=self.T_tensor(vec,left=False)
        if self.debug:
            assert np.abs(vec[vec.shape[0]//2:]).sum() == 0, f'first qubit is not zero ({np.abs(vec[vec.shape[0]//2:]).sum()}) after right shift '
        # Adder
        if vec.ndim >1 :
            vec=vec.flatten()
        if self.xj==frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)]):
            if len(n)==1:
                vec=self.adder_cpu(vec,xj=frozenset([Fraction(1,3),Fraction(2,3)]))
            elif len(n)==2:
                vec=self.adder_cpu(vec,xj=frozenset([Fraction(1,3),Fraction(-1,3)]))
            else:
                raise NotImplementedError(f"control map with len(n)={len(n)} not supported")
        else:
            vec=self.adder_cpu(vec,xj=self.xj)
        
        return vec

    def projection_map(self,vec,pos,n):
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
        vec=self.P_tensor(vec,n,pos)
        if self.normalization:
            vec=self.normalize(vec)
        return vec

    def encoding(self):
        """Encoding process: Randomly apply Bernoulli map
        """
        vec=self.vec_history[-1].copy()
        vec=self.op_list[('B',)](vec)
        self.update_history(vec,('B',),None)

    def random_control(self,p_ctrl,p_proj,p_global=None):
        """The competition between chaotic random unitary, control map and projection, where the projection can only be applied after the unitary. The probability of control is `p_ctrl`, and the probability of projection is `p_proj`.

        Parameters
        ----------
        p_ctrl : float, 0<=p_ctrl<=1
            probability of control
        p_proj : float, 0<=p_proj<=1
            probability of projection
        p_global : float, 0<=p_global<=1
            probability of using global control
        """ 
        vec=self.vec_history[-1].copy()
        
        op_l=[]
        if self.rng_C.random()<=p_ctrl:
            # control map
            if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]),frozenset([0]),frozenset([1]),frozenset([-1])]:
                p_0=self.inner_prob(vec, pos=[self.L-1],n_list=[0])
                op=('C',0) if self.rng.random()<=p_0 else ('C',1)
            elif self.xj in [frozenset([Fraction(1,3),Fraction(-1,3)])]:
                p={n:self.inner_prob(vec,pos=[0,self.L-1],n_list=n) for n in [(0,0),(0,1),(1,0)]}
                p[(1,1)]=np.clip(1-p[(0,0)]-p[(0,1)]-p[(1,0)],0,1)
                op=self.rng.choice([(0,0),(0,1),(1,0),(1,1)],p=[p[(0,0)],p[(0,1)],p[(1,0)],p[(1,1)]])
                op=('C',)+tuple(op)
            elif self.xj in [frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)])]:
                assert p_global is not None, "p_global should not be None"
                if self.rng_C.random()<=p_global:
                    p_0=self.inner_prob(vec, pos=[self.L-1],n_list=[0])
                    op=('C',0) if self.rng.random()<=p_0 else ('C',1)
                else:
                    p={n:self.inner_prob(vec,pos=[0,self.L-1],n_list=n) for n in [(0,0),(0,1),(1,0)]}
                    p[(1,1)]=np.clip(1-p[(0,0)]-p[(0,1)]-p[(1,0)],0,1)
                    op=self.rng.choice([(0,0),(0,1),(1,0),(1,1)],p=[p[(0,0)],p[(0,1)],p[(1,0)],p[(1,1)]])
                    op=('C',)+tuple(op)
        else:
            # chaotic map
            op=('B',)
        op_l.append(op)
        vec=self.op_list[op](vec)

        if "B" in op:
            for idx,pos in enumerate([self.L-1,self.L-2]):
                if self.rng_C.random()<p_proj:
                    # projection map
                    p_2=(self.inner_prob(vec, pos=[pos],n_list=[0]))
                    op_2=(f"P{pos}",0) if self.rng.random()<p_2 else (f"P{pos}",1)
                    vec=self.op_list[op_2](vec)
                    op_l.append(op_2)
        self.update_history(vec,op_l,None)


    def reference_control(self,op_history):
        """The reference protocol, where the operators are specified by `op_history`. The operators are applied sequentially.

        Parameters
        ----------
        op_history : list of list of str
            The history of operators applied to the circuit. See `__init__` for the definition.
        """
        vec=self.vec_history[-1].copy()
        for op_l in op_history:
            self.rng_C.random() # dummy random to keep same random sequence, same as p_ctrl
            for idx,op in enumerate(op_l):
                vec=self.op_list[op](vec)
                if 'C' in op:
                    self.rng.random() # dummy random for C0/C1
                if 'B' in op:
                    self.rng_C.random() # dummy in projection at L-1, L-2
                    self.rng_C.random() # dummy in projection at L-1, L-2
                if 'P' in op:
                    self.rng.random() # dummy in random pro Pi0/Pi1
            self.update_history(vec,op_l,None)
    
    def fixed_control(self,p_ctrl):
        '''a temporary protocol, with fixed Unitary and N_c control from binomial distribution'''
        vec=self.vec_history[-1].copy()
        op_l=[]
        
        n=self.rng_C.binomial(self.L,p_ctrl)
        for _ in range(n):
            p_0=self.inner_prob(vec, pos=[self.L-1],n_list=[0])
            op=('C',0) if self.rng.random()<=p_0 else ('C',1)
            vec=self.op_list[op](vec)
            op_l.append(op)


        
        for _ in range(self.L):
            vec=self.op_list[('B',)](vec)
            op_l.append(('B',))
        
        

        self.update_history(vec,op_l,None)

    def order_parameter(self,vec=None):
        """Calculate the order parameter. For `xj={1/3,2/3}`, it is \sum Z.Z, for `xj={0}`, it is \sum Z.

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None

        Returns
        -------
        float
            the order parameter. 
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]),frozenset([Fraction(1,3),Fraction(-1,3)]),frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)])]:
            O=self.ZZ_tensor(vec)
        elif self.xj in [frozenset([0]),]:
            O=self.Z_tensor(vec)
        elif self.xj ==frozenset([1]):
            O=-self.Z_tensor(vec)
        elif self.xj in [frozenset([-1])]:
            O=self.overlap(vec)
        else:
            raise NotImplementedError(f"Order parameter of {self.xj} is not implemented")
        if self.debug:
            assert np.abs(O.imag)<self._eps, f'<O> is not real ({O}) '
        return O.real
    
    def von_Neumann_entropy_pure(self,subregion,vec=None,n=1,threshold=1e-10):
        """Calculate the von Neumann entropy of a pure state, where the state vector is `vec` and the subregion is `subregion`. Using the Schmidt decomposition, the von Neumann entropy is -\sum_i \lambda_i^2 \log \lambda_i^2, where \lambda_i is the singular value of the reshaped state vector `vec`.

        Parameters
        ----------
        subregion : list of int or np.array
            The spatial subregion to calculate the von Neumann entropy
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None
        n: int, optional,
            n-th Renyi entropy
        threshold: float, optional
            threshold to clip the singular value, by default 1e-10. For 0-th Reny entropy, threshold is 1e-8, by empirical observation.

        Returns
        -------
        float
            Von Neumann entropy of the subregion
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        vec_tensor=vec.reshape((2,)*(self.L_T))
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L_T) if i not in subregion]
        vec_tensor_T=vec_tensor.transpose(np.hstack([subregion , not_subregion]))
        S=np.linalg.svd(vec_tensor_T.reshape((2**len(subregion),2**len(not_subregion))),compute_uv=False)
        S_pos=np.clip(S,1e-18,None)
        S_pos2=S_pos**2
        # if threshold is not None:
        #     S_pos2=np.clip(S,threshold,None)**2
        # else:
        #     S_pos2=S**2
        if n==1:
            return -np.sum(np.log(S_pos2)*S_pos2)
        elif n==0:
            return np.log((S_pos2>threshold**2).sum())
        elif n==np.inf:
            return -np.log(np.max(S_pos2))
        else:
            return np.log(np.sum(S_pos2**n))/(1-n)

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False,n=1,threshold=1e-10):
        """Calculate the half-system entanglement entropy, where the state vector is `vec`. The half-system entanglement entropy is defined as \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), where S_([i,i+L/2)) is the von Neumann entropy of the subregion [i,i+L/2).

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None
        selfaverage : bool, optional
            if true, average over all possible halves, namely, \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), by default False

        Returns
        -------
        float
            Half-system entanglement entropy
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        if selfaverage:
            return np.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)])
        else:
            return self.von_Neumann_entropy_pure(np.arange(self.L//2),vec,n=n,threshold=threshold)

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None,n=1,threshold=1e-10):
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
        if self.debug:
            assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
            assert np.intersect1d(subregion_A,subregion_C).size==0 , "Subregion A and C overlap"
            assert np.intersect1d(subregion_B,subregion_C).size==0 , "Subregion B and C overlap"
        if vec is None:
            vec=self.vec_history[-1].copy()
        if selfaverage:
            return np.mean([self.tripartite_mutual_information((subregion_A+shift)%self.L,(subregion_B+shift)%self.L,(subregion_C+shift)%self.L,selfaverage=False) for shift in range(len(subregion_A))])
        else:
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec,n=n,threshold=threshold)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec,n=n,threshold=threshold)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec,n=n,threshold=threshold)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec,n=n,threshold=threshold)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC

    def update_history(self,vec=None,op=None,p=None):
        """Update the history of state vector, operators and Born probability. If `store_vec` is True, the state vector is appended to the history. If `store_op` is True, the operators are appended to the history. If `store_prob` is True, the Born probability is appended to the history.

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, if None, do not save it, by default None
        op : tuple, optional
            operations, if None, do not save it, by default None
        p : float, optional
            Born probability, if None, do not save it, by default None
        """
        if vec is not None:
            vec=vec.copy().flatten()
            if self.store_vec:
                self.vec_history.append(vec)
            else:
                self.vec_history=[vec]

        if op is not None:
            if self.store_op:
                self.op_history.append(op)
            else:
                self.op_history=[op]
        
        if p is not None:
            if self.store_prob:
                self.prob_history.append(p)
            else:
                self.prob_history=[p]

    def inner_prob(self,vec,pos,n_list):
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
        if vec.ndim != (2,)*self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        idx_list=np.array([slice(None)]*self.L_T)
        idx_list[pos]=n_list
        vec_0=vec[tuple(idx_list)].flatten()
        inner_prod=vec_0.conj()@vec_0
        if self.debug:
            assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
            assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
            assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        inner_prod=np.clip(inner_prod.real,0,1)
        return inner_prod

    def normalize(self,vec):
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
        self.update_history(None,None,norm2)
        if norm2 > 0:
            return vec/np.sqrt(norm2)
        else:
            return vec
        
    def XL_tensor(self,vec):
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
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        vec=np.roll(vec,1,axis=self.L-1)
        return vec

    def P_tensor(self,vec,n,pos):
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        for n_i,pos_i in zip(n,pos):
            idx_list=[slice(None)]*self.L_T
            idx_list[pos_i]=1-n_i
            vec[tuple(idx_list)]=0
        return vec

    @monitor
    def T_tensor(self,vec,left=True):
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
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        if left:
            idx_list_2=list(range(1,self.L))+[0]
        else:
            idx_list_2=[self.L-1]+list(range(self.L-1))
        if self.ancilla:
            idx_list_2.append(self.L)
        return vec.transpose(idx_list_2)

    @monitor
    def S_tensor(self,vec,rng):
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
        U_4=U(4,rng)
        if not self.ancilla:
            vec=vec.reshape((2**(self.L-2),2**2)).T
            return (U_4@vec).T
        else:
            vec=vec.reshape((2**(self.L-2),2**2,2)).transpose((1,0,2)).reshape((2**2,2**(self.L-1)))
            return (U_4@vec).reshape((2**2,2**(self.L-2),2)).transpose((1,0,2))
    @monitor
    def R_tensor(self,vec,n,pos):
        vec=self.P_tensor(vec,n,pos)
        # if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]),frozenset([0]),frozenset([1]),frozenset([-1])]:
        if len(n)==1:
            # projection on the last bits
            if n[0]==1:
                if self.feedback:
                    vec=self.XL_tensor(vec)
        # elif self.xj==frozenset([Fraction(1,3),Fraction(-1,3)]):
        elif len(n)==2:
            # projection on the first and last bits
            if n[0]^n[1]==0:
                if self.feedback:
                    vec=self.XL_tensor(vec)
        else:
            raise NotImplementedError(f"Reset of {self.xj} is not implemented")
        if self.normalization:
            vec=self.normalize(vec)
        return vec
    
    def ZZ_tensor(self,vec):
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
        if vec.ndim != (2,)*self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    # idx_list=[slice(None)]*self.L_T
                    # idx_list[i],idx_list[(i+1)%self.L]=zi,zj
                    inner_prod=self.inner_prob(vec, [i,(i+1)%self.L],[zi,zj])
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    # vec_i=vec[tuple(idx_list)].flatten()
                    # rs+=vec_i.conj()@vec_i*exp
                    rs+=inner_prod*exp
        return -rs/self.L
    
    def Z_tensor(self,vec):
        """Calculate the order parameter for Ferromagnetic state. The order parameter is defined as \sum_{i=0..L-1} <Z_i>, where Z_i is the Pauli Z matrix at site i.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector

        Returns
        -------
        float
            order parameter for ferromagnetic state
        """
        if vec.ndim != (2,)*self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec,pos=[i],n_list=[0])
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L

    def overlap(self,vec):
        """Calculate the overlap between the state vector `vec` and the fixed point. The fixed point is the state vector with all 1s, encoded in integer
        """
        if vec.ndim >1 :
            vec=vec.flatten()
        return np.sum(np.abs(vec[self.fixed_point])**2).real

    @monitor
    def adder_cpu(self,vec,xj):
        if not self.ancilla:
            vec=self.adder(xj)@vec
        else:
            vec=(self.adder(xj)@vec.reshape((2**self.L,2))).flatten()
        
        if self.normalization and not self.feedback:
            print(f'before {vec.conj()@vec}')
            vec=self.normalize(vec)
        return vec
        
    @lru_cache(maxsize=None)
    def adder(self,xj):
        """Calculate the adder matrix, which is the shuffle of the state basis. Note that this is not a full adder, which assumes the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).

        Returns
        -------
        numpy.array, shape=(2**L,2**L)
            adder matrix
        """
        if xj==frozenset([Fraction(1,3),Fraction(2,3)]):
            int_1_6=dec2bin(Fraction(1,6), self.L)|1
            int_1_3=dec2bin(Fraction(1,3), self.L)

            if self.feedback:
                old_idx=np.arange(2**(self.L-1))
                adder_idx=np.array([int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2))
                new_idx=(old_idx+adder_idx)
                ones=np.ones(2**(self.L-1))
            else:
                old_idx=np.arange(2**(self.L))
                adder_idx=np.array([int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2)+[int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2))
                new_idx=(old_idx+adder_idx)%2**self.L
                ones=np.ones(2**(self.L))
                
            # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 
            mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
            mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
            new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
            return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**self.L,2**self.L))

        elif xj in [frozenset([0]),frozenset([Fraction(1,3),Fraction(-1,3)])]:
            return sp.eye(2**self.L)   
        elif xj == frozenset([1]):
            # int_1=(1<<self.L)-1
            # int_1_2=(1<<(self.L-1))+1
            # old_idx=np.arange(2**(self.L-1))
            # adder_idx=np.array([int_1]*2**(self.L-2)+[int_1_2]*2**(self.L-2))

            int_1_2=dec2bin(Fraction(1,2), self.L)
            if self.feedback:
                old_idx=np.arange(2**(self.L-1))
                adder_idx=np.array([int_1_2]*2**(self.L-1))
                new_idx=(old_idx+adder_idx)%2**self.L
                ones=np.ones(2**(self.L-1))
            else:
                raise NotImplementedError("Non-feedback is not implemented for xj=1")
            return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**self.L,2**self.L))
        elif xj == frozenset([-1]):
            old_idx=np.arange(2**(self.L))
            # adder_idx=np.array([self.add_x]*2**self.L)
            new_idx=(old_idx+self.add_x)%2**self.L
            ones=np.ones(2**(self.L))
            return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**self.L,2**self.L))
        else:
            raise NotImplementedError(f"{xj} is not implemented")