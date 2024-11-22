'''
Compare the tripartite mutual information of the toy model with different p_ctrl values between Tao's and Haining's version.
'''

import numpy as np
import QCT as qct
from haining_correct_functions import random_control
from metric_func import tripartite_mutual_information_tao
from QCT_util import Haar_state
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time

def compare_tmi(L_values, p_ctrl_values, p_proj, num_iterations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_time_steps = { L : L**2*2 for L in L_values}
    
    for L in L_values:
        results = {'tao': [], 'haining': [], 'p_ctrl_values': p_ctrl_values.tolist()}
        
        time_start = time.time()
        for p_ctrl in tqdm(p_ctrl_values, desc=f"Processing L={L}"):
            tao_tmi = 0
            for iter_idx in range(num_iterations):
                qct_tao = qct.QCT(L, p_ctrl, p_proj)
                for step in range(num_time_steps[L]):
                    qct_tao.step_evolution()
                tao_tmi += tripartite_mutual_information_tao(qct_tao.state, L)
                
            tao_tmi /= num_iterations
            results['tao'].append(tao_tmi)

        time_end = time.time()
        print(f"Time taken for Tao's version: {time_end - time_start} seconds")

        time_start = time.time()
        for p_ctrl in tqdm(p_ctrl_values, desc=f"Processing L={L}"):
            haining_tmi = 0
            for _ in range(num_iterations):
                state = Haar_state(ensemble=1, rng=np.random.default_rng(), L=L).flatten()
                state_haining = state.copy()
                for _ in range(num_time_steps[L]):
                    state_haining, _ = random_control(state_haining, p_ctrl, p_proj, L)
                haining_tmi += tripartite_mutual_information_tao(state_haining, L)
            haining_tmi /= num_iterations
            results['haining'].append(haining_tmi)
        time_end = time.time()
        print(f"Time taken for Haining's version: {time_end - time_start} seconds")
        
        # Write results to file
        filename = os.path.join(output_dir, f'tmi_results_L{L}.json')
        with open(filename, 'w') as f:
            json.dump(results, f)

def plot_results(output_dir):
    plt.figure(figsize=(12, 8))

    # Get all result files
    result_files = [f for f in os.listdir(output_dir) if f.startswith('tmi_results_L') and f.endswith('.json')]

    for file in result_files:
        with open(os.path.join(output_dir, file), 'r') as f:
            results = json.load(f)
        
        L = int(file.split('L')[1].split('.')[0])
        p_ctrl_values = results['p_ctrl_values']

        plt.plot(p_ctrl_values, results['tao'], label=f"Tao's Version (L={L})", linestyle='-', marker='o')
        plt.plot(p_ctrl_values, results['haining'], label=f"Haining's Version (L={L})", linestyle='--', marker='s')

    plt.xlabel('p_ctrl')
    plt.ylabel('Tripartite Mutual Information')
    plt.title("Comparison of Tripartite Mutual Information: Tao vs Haining's Version")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tmi_comparison.png'))
    plt.close()

if __name__ == "__main__":
    L_values = [8]  # You can add more L values if needed
    p_ctrl_values = np.linspace(0, 0.6, 21)
    num_iterations = 20
    p_proj = 0.3
    output_dir = 'tmi_pctrl_comparison'

    compare_tmi(L_values, p_ctrl_values, p_proj, num_iterations, output_dir)
    plot_results(output_dir)

    print(f"Comparison completed. Results saved in '{output_dir}' folder.")
