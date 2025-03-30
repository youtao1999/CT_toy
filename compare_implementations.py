import numpy as np
from haining_correct_functions import Haar_state, random_control, Bernoulli_map, control_map, projection_map, inner_prob
from QCT import QCT
from QCT_util import *

initial_state_rng = np.random.default_rng(42)
initial_state = Haar_state(12, 1, rng=initial_state_rng)
print(len(initial_state))

qct_tao = QCT(12, 0.5, 0, seed_initial=42, seed_ctrl=42, seed_proj=42, seed_born=42, seed_scramble=42)
for i in range(10):
    qct_tao.step_evolution()
    print(qct_tao.state.flatten()[0:10])

