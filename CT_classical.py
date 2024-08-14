def dec2bin(x,L,base=2):
    """convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 as integer part is truncated. Example, 1/3 is 010101...
    It is important to note that the output is not a binary string, but an integer of the L-bit approximated form of x. For example, if L=3, x=0.5 (binary form 0.100...), 
    the output would be 4, which is 100 in binary form, corresponding to 0.100... after truncating the leading 0 and the trailing 0s.
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

from fractions import Fraction
import numpy as np
class CT_classical:
    def __init__(self,L,store_vec=False,store_op=False,seed=None,seed_vec=None,seed_C=None,x0=None,xj=set([Fraction(1,3),Fraction(2,3)]),random=True,feedback=True,add_x=False):
        '''
        use int to represent the state vector, the float number is int/2**L
        if classical is False: save using an array of 2^L
        '''
        self.L=L
        self.store_vec=store_vec
        self.store_op=store_op
        self.rng=np.random.default_rng(seed)    # for backward campatibility
        self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng # control initial state
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng # control circuit
        self.x0=x0
        self.xj=set(xj)
        self.binary_xj=self._initialize_binary(xj)
        self.allone=(1<<(self.L))-1 # 0b1..11
        self.mask101=((1<<self.L-1))+5 # 0b10..0101
        self.half=1<<(self.L-1)
        self.leading=(1<<(self.L-1))+1 # 0b10..01
        self.op_history=[]  # control: true, Bernoulli: false
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self.random=random  # if true, scrambler is random, else, use fixed mapping `self.scrambler`
        self.scrambler={7:1,6:4,5:0,4:3,3:2,2:5,1:6,0:7}
        self.add_x=add_x
        self.feedback=feedback

    def _initialize_binary(self,xj):
        '''The output is not a binary string, but rather an integer of the L-bit approximated form of x. For example, if L=3, x=0.5 (binary form 0.100...),
        in which '''
        if not self.xj in [set([0]),set([-1]),set([Fraction(1,3),Fraction(-1,3)])]:
            binary_xj={xj/2:dec2bin(xj/2,self.L) for xj in xj}
            if Fraction(1,6) in binary_xj:
                binary_xj[Fraction(1,6)]+=1
            return binary_xj

    def _initialize_vector(self):
        '''save using an array of L'''
        if self.x0 is None:
            vec=self.rng_vec.integers(low=0,high=1<<self.L)
        else:
            # vec=int(self.x0*(1<<self.L))
            vec=dec2bin(self.x0,self.L)
        return vec

    def Bernoulli_map(self,vec):
        vec=self.T(vec,left=True)
        vec=self.S(vec)        
        return vec
    
    def control_map(self,vec):
        vec=self.P(vec)
        vec=self.T(vec,left=False)
        vec=self.adder(vec)
        return vec

    def random_control(self,p):
        '''
        p: the control probability
        '''
        vec=self.vec_history[-1]
        p0=self.rng_C.random()

        if p0<p:
            vec=self.control_map(vec)
        else:
            vec=self.Bernoulli_map(vec)
        self.update_history(vec,(p0<p))
    
    def update_history(self,vec=None,op=None):
        if vec is not None:
            if self.store_vec:
                self.vec_history.append(vec)
            else:
                self.vec_history=[vec]
        if op is not None:
            if self.store_op:
                self.op_history.append(op)
            else:
                self.op_history=[op]

    def P(self,vec):
        '''P is the projection operator which set the last bit to 0 if xj is {1/3,2/3} or 0,
          set the last bit to 1 if xj is {1/3,-1/3} or 1, do nothing if xj is 1 or -1'''
        if self.xj in [set([Fraction(1,3),Fraction(2,3)]),set([0]),set([1]),set([-1])]:
            if self.feedback:
                vec=vec&(~1) # set the last bit to 0
        elif self.xj == set([Fraction(1,3),Fraction(-1,3)]):
            if self.feedback:
                if bin(self.leading&vec).count('1')%2==0:
                    vec=vec^1 # flip the last bit
        return vec

    def T(self,vec,left=True):
        # Cyclic shift
        if left:
            vec=(vec>>self.L-1)^(vec<<1)&(self.allone)
        else:
            vec=(vec>>1)^(vec<<self.L-1)&(self.allone)
        return vec
    
    def S(self,vec):
        first=vec&(~0b111)
        if self.random:
            vec=first+self.rng_C.integers(low=0,high=8)
        else:
            last_three=vec&0b111
            vec=first+self.scrambler[last_three]
        return vec

    def adder(self,vec):
        if self.xj==set([Fraction(1,3),Fraction(2,3)]):
            secondbit=(vec>>(self.L-2))&1
            if secondbit==0:
                # add 1/6
                vec+=self.binary_xj[Fraction(1,6)]
            else:
                # add 1/3
                vec+=self.binary_xj[Fraction(1,3)]
            # handle extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0
            if vec&self.mask101 in [4,self.leading]:
                # flip the second bit from the right
                vec=vec^2
            vec&=self.allone
        elif self.xj in [set([0]),set([Fraction(1,3),Fraction(-1,3)])]:
            pass
        elif self.xj==set([1]):
            # secondbit=(vec>>(self.L-2))
            # if secondbit==0:
            #     # add 1
            #     vec+=self.allone
            #     vec&=self.allone
            # else:
            #     # minus 1/2
            #     vec+=self.leading
            #     vec&=self.allone
            vec+=(self.half)
            vec%=(1<<self.L)
        elif self.xj==set([-1]):
            vec+=self.add_x
            vec%=(1<<self.L)
        else:
            raise NotImplementedError(f"{self.xj} is not implemented")
        return vec
        
    def order_parameter(self,vec=None):
        if vec is None:
            vec=self.vec_history[-1]
        if self.xj in [set([Fraction(1,3),Fraction(2,3)]),set([Fraction(1,3),Fraction(-1,3)])]:
            O=self.ZZ(vec)
        elif self.xj in [set([0]),set([-1])]:
            O=self.Z(vec)
        elif self.xj in [set([1])]:
            O=-self.Z(vec)
        return O
        
    def ZZ(self,vec):
        zz=self.T(vec,left=True)^vec
        return 2*bin(zz).count('1')/self.L-1

    def Z(self,vec):
        return 1-2*bin(vec).count('1')/self.L