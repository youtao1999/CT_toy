import numpy as np
import matplotlib.pyplot as plt
import QCT as qct
from metric_func import half_system_entanglement_entropy
from haining_correct_functions import random_control
from QCT_util import Haar_state
from tqdm import tqdm
import os
import json

def compare_entanglement_entropy(L_values, p_ctrl_values, p_proj, num_iterations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_time_steps = { L : L**2*2 for L in L_values}
    for L in L_values:
        results = {'tao': [], 'haining': [], 'p_ctrl_values': p_ctrl_values.tolist()}
        
        for p_ctrl in tqdm(p_ctrl_values, desc=f"Processing L={L}"):
            tao_entropy = 0
            haining_entropy = 0

            for _ in range(num_iterations):
                qct_tao = qct.QCT(L, p_ctrl, p_proj)
                state_haining = Haar_state(L, 1, rng=np.random.default_rng(), k=1).flatten()
                
                for _ in range(num_time_steps[L]):
                    qct_tao.step_evolution()
                    state_haining, _ = random_control(state_haining, p_ctrl, p_proj, L)

                tao_entropy += half_system_entanglement_entropy(qct_tao.state, L)
                haining_entropy += half_system_entanglement_entropy(state_haining, L)

            tao_entropy /= num_iterations
            haining_entropy /= num_iterations

            results['tao'].append(tao_entropy)
            results['haining'].append(haining_entropy)

        # Write results to file
        filename = os.path.join(output_dir, f'entropy_results_L{L}.json')
        with open(filename, 'w') as f:
            json.dump(results, f)

def plot_results(output_dir):
    plt.figure(figsize=(12, 8))

    # Get all result files
    result_files = [f for f in os.listdir(output_dir) if f.startswith('entropy_results_L') and f.endswith('.json')]

    for file in result_files:
        with open(os.path.join(output_dir, file), 'r') as f:
            results = json.load(f)
        
        L = int(file.split('L')[1].split('.')[0])
        p_ctrl_values = results['p_ctrl_values']

        plt.plot(p_ctrl_values, results['tao'], label=f"Tao's Version (L={L})", linestyle='-', marker='o')
        plt.plot(p_ctrl_values, results['haining'], label=f"Haining's Version (L={L})", linestyle='--', marker='s')

    plt.xlabel('p_ctrl')
    plt.ylabel('Self-Averaged Entanglement Entropy')
    plt.title("Comparison of Self-Averaged Entanglement Entropy: Tao vs Haining's Version")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entanglement_entropy_comparison.png'))
    plt.close()

if __name__ == "__main__":
    L_values = [8, 10, 12, 14]  # You can add more L values if needed
    p_ctrl_values = np.linspace(0, 1, 21)
    num_iterations = 2000
    p_proj = 0.3
    output_dir = 'ee_vs_p_ctrl_comparison'

    # compare_entanglement_entropy(L_values, p_ctrl_values, p_proj, num_iterations, output_dir)
    plot_results(output_dir)

    print(f"Comparison completed. Results saved in '{output_dir}' folder.")
