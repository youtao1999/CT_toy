import sys
sys.path.append('/scratch/ty296/CT_toy')
import sv
from sv import SingularValueComputer
import numpy as np
# Note: p_scan_values should be an array, not a string

p_scan_values = np.array([0.0, 1.0])  # or use np.linspace(0.0, 1.0, 2)

sv_computer = SingularValueComputer(L=12, p_fixed=0.0, p_fixed_name="p_ctrl", p_scan_values=p_scan_values, chunk_size=2, comparison=False)
result = sv_computer.compute_chunk()
print(f"Result keys: {result.keys()}")
print(f"Singular values keys: {result['singular_values'].keys()}")
print(f"Singular values: {np.shape(result['singular_values']['AB'])}")