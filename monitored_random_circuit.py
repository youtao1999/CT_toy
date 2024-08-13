from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import random_unitary, Statevector, DensityMatrix
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from IPython.display import display
from collections import Counter

class MonitoredRandomCircuit:
    def __init__(self, L, pm, pf, num_steps, num_runs):
        self.L = L
        self.pm = pm
        self.pf = pf
        self.num_steps = num_steps
        self.num_runs = num_runs

    def block_diagonal_unitary(self):
        """Generate a block-diagonal unitary matrix that leaves |↑↑⟩ invariant and acts randomly on |↑↓⟩, |↓↑⟩, |↓↓⟩."""
        U3 = random_unitary(3).data  # Haar-random unitary in the 3x3 subspace
        U = np.eye(4, dtype=complex)
        U[1:, 1:] = U3  # Applying the random unitary to the |↑↓⟩, |↓↑⟩, |↓↓⟩ states
        return U

    def apply_custom_unitaries(self, qc, qreg, step):
        """Apply custom block-diagonal unitaries on the system in a zigzag pattern."""
        if step % 2 == 0:
            # Apply unitaries to pairs (0-1, 2-3, 4-5, ...)
            for i in range(0, self.L-1, 2):
                U = self.block_diagonal_unitary()
                qc.unitary(U, [qreg[i], qreg[i+1]], label='U')
        else:
            # Apply unitaries to pairs (1-2, 3-4, 5-0, ...)
            for i in range(1, self.L, 2):
                U = self.block_diagonal_unitary()
                if i == self.L-1:
                    qc.unitary(U, [qreg[i], qreg[0]], label='U')
                else:
                    qc.unitary(U, [qreg[i], qreg[i+1]], label='U')

    def measure_and_update_state(self, state, qreg):
        """Perform measurements probabilistically and update the state vector accordingly."""
        measured_indices = []
        measurements = {}

        for i in range(len(qreg)):
            if np.random.rand() < self.pm:
                measured_indices.append(i)
                outcome, new_state = state.measure([i])  # This returns a tuple (outcome, new_state)
                state = new_state
                # Extract measurement result for the qubit
                measurement = int(outcome)  # Directly use the measurement outcome string
                measurements[i] = measurement  # 0 for up spin, 1 for down spin

        return state, measured_indices, measurements

    def run_simulation_step(self, qreg, creg, step, state, qc):
        """Run a single step of the simulation, applying unitaries and measurements, and update the state."""
        # Apply unitaries
        unitary_qc = QuantumCircuit(qreg, creg)
        self.apply_custom_unitaries(unitary_qc, qreg, step)
        state = state.evolve(unitary_qc)
        qc.compose(unitary_qc, inplace=True)

        # Perform measurements
        measure_qc = QuantumCircuit(qreg, creg)
        state, measured_indices, measurements = self.measure_and_update_state(state, qreg)
        for i in measured_indices:
            measure_qc.measure(qreg[i], creg[i])
        qc.compose(measure_qc, inplace=True)

        return state, measured_indices, measurements

    def simulate_circuit_once(self):
        """Simulate the quantum circuit and apply feedback corrections iteratively for a single run."""
        qreg = QuantumRegister(self.L, 'q')
        creg = ClassicalRegister(self.L, 'c')
        qc = QuantumCircuit(qreg, creg)

        # Get the initial state vector
        state = Statevector.from_label('1' * self.L)  # Initialize in the all-down state
        trajectory = np.zeros((self.num_steps + 1, 2**self.L, 2**self.L), dtype=complex)
        trajectory[0] = state.to_operator().data  # Store initial state as density matrix

        for step in range(self.num_steps):
            # 1. Apply unitaries, perform measurements, and update the state
            state, measured_indices, measurements = self.run_simulation_step(qreg, creg, step, state, qc)
            
            # 2. Apply feedback corrections based on measurement results
            correction_qc = QuantumCircuit(qreg)
            for i in measured_indices:
                if measurements[i] == 1:  # If the measurement result is 1 (down spin)
                    correction_qc.reset(qreg[i])
                    if np.random.rand() < self.pf:  # Flip with probability pf
                        correction_qc.x(qreg[i])
                    state = state.evolve(correction_qc)
            qc.compose(correction_qc, inplace=True)  # Compose correction_qc into main qc
            trajectory[step + 1] = state.to_operator().data  # Store state as density matrix

        return state, qc, trajectory

    def simulate_circuit(self):
        """Simulate the quantum circuit multiple times and accumulate the results."""
        accumulated_counts = Counter()
        final_circuit = None
        all_trajectories = np.zeros((self.num_runs, self.num_steps + 1, 2**self.L, 2**self.L), dtype=complex)

        for run_index in range(self.num_runs):
            final_state, qc, trajectory = self.simulate_circuit_once()
            if final_circuit is None:
                final_circuit = qc  # Keep the final circuit from one of the runs
            counts = final_state.probabilities_dict(decimals=3)
            accumulated_counts.update(counts)
            all_trajectories[run_index] = trajectory

        # Compute the average density matrices directly using numpy.mean
        avg_density_matrices = np.mean(all_trajectories, axis=0)
        avg_density_matrices = [DensityMatrix(dm) for dm in avg_density_matrices]

        return dict(accumulated_counts), final_circuit, avg_density_matrices

    @staticmethod
    def second_renyi_entropy(rho):
        """
        Compute the second Rényi entropy for a given density matrix rho.

        Parameters:
        rho (np.ndarray): Density matrix

        Returns:
        float: Second Rényi entropy
        """
        rho_squared = np.dot(rho, rho)  # Square the density matrix
        trace_rho_squared = np.trace(rho_squared)  # Compute the trace of the squared density matrix
        renyi_entropy = -np.log2(trace_rho_squared)  # Compute the second Rényi entropy

        return renyi_entropy

    @classmethod
    def plot_renyi_entropy_vs_time(cls, system_sizes, pm, pf, num_steps, num_runs):
        """
        Plot the second Rényi entropy as a function of time for different system sizes.

        Parameters:
        system_sizes (list): List of system sizes
        pm (float): Measurement rate
        pf (float): Feedback rate
        num_steps (int): Number of steps in the simulation
        num_runs (int): Number of times to repeat the entire simulation
        """
        plt.figure(figsize=(10, 6))

        for L in system_sizes:
            # Create an instance of the class
            circuit_simulator = cls(L, pm, pf, num_steps, num_runs)

            # Run the simulation
            counts, final_circuit, avg_density_matrices = circuit_simulator.simulate_circuit()
            print(f"Final measurement results for L={L}:")
            print(counts)

            # Compute the second Rényi entropy as a function of time
            renyi_entropies = [cls.second_renyi_entropy(dm.data) for dm in avg_density_matrices]
            # Convert to numpy array
            renyi_entropies = np.array(renyi_entropies)
            # Remove the first data point and rescale by 1/L
            renyi_entropies_rescaled = renyi_entropies[1:] / L
            time_steps_rescaled = np.array(range(1, num_steps + 1)) / L

            # Plot the data
            plt.plot(time_steps_rescaled, renyi_entropies_rescaled, marker='o', label=f'L={L}')

        # Plot settings
        plt.xlabel('Time step / L')
        plt.ylabel('Second Rényi Entropy / L')
        plt.title('Second Rényi Entropy as a function of time (rescaled)')
        plt.legend()
        plt.grid(True)
        plt.show()

'''To use this module in Jupyter notebook to plot the second Renyi entropy vs time'''

# # Import the MonitoredRandomCircuit class
# from monitored_random_circuit import MonitoredRandomCircuit

# # Define parameters
# L = 6
# pm = 0.9
# pf = 0.1
# num_steps = 100
# num_runs = 200
# system_sizes = [4, 6, 8]

# # Plot the second Rényi entropy as a function of time for different system sizes
# MonitoredRandomCircuit.plot_renyi_entropy_vs_time(system_sizes, pm, pf, num_steps, num_runs)

# Example usage
if __name__ == "__main__":
    # Parameters
    system_sizes = [4, 6, 8]  # Different system sizes
    pm = 0.1  # Measurement rate
    pf = 0.1  # Feedback rate
    num_steps = 100  # Number of steps in the simulation
    num_runs = 200  # Number of times to repeat the entire simulation