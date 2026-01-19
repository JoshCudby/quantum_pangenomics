import pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from collections import Counter
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator, NullLocator)
from typing import Optional

from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


def plot_several_p_dist(
    axs: list[Axes], 
    filename, prob,
    db_fixed, dg_fixed, shots,
    ps, rescale,
    max_beta_T:Optional[float]=None, eps:Optional[float]=None, alpha:Optional[float]=None,
    iters=None, normalise=False
) -> list[Axes]:
    data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

    _, hamiltonian, _, ising_offset = get_Q_and_hamiltonian(data_file)
    num_qubits: int = hamiltonian.num_qubits
    base_file_name = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/nonvariational.{filename}.db{db_fixed}.dg{dg_fixed}.shots{shots}'
    append_str = (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + (f'.alpha{alpha}' if alpha is not None else '')
    with open(f'{base_file_name}{append_str}.pkl', 'rb') as f:
        res = pickle.load(f)
        
    sample_sequence = []
    samples_dict: dict[tuple[int, float], list[list[str]]] = res['samples_dict']
    keys = samples_dict.keys()
    energies = res['energies']
    if iters is None:
        iters = [0, 5, 9]
    energies = np.array(energies)
    if normalise:
        energies /= energies.max()
        # energies /= energies[0,0]
    
    cutoff=25
    ax = axs[0]
    rand_shots = min(shots*10, 40000)
    random_samples = np.random.choice(('0', '1'), (rand_shots, num_qubits), p=(1-prob,prob))
    rand_samples = [''.join(sample) for sample in random_samples]
    rand_vals = np.round(ising_offset + evaluate_sparse_pauli_samples(rand_samples, hamiltonian), 2)
    ax.hist(rand_vals, bins=range(cutoff+1), weights=[1/rand_shots]*len(rand_vals), rwidth=1, log=True, color='gray', label='Random')
    ax.set_xlim(0, cutoff)
    ax.set_ylim(10**-3, 10**0)
    

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.text(.95, .99, 'Iter = 0', ha='right', va='top', transform=ax.transAxes)
    # ax.legend(loc='best', bbox_to_anchor=(0.6, 0.6, 0.4, 0.4))
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.92),  # move legend down
        frameon=True
    )
    
    for i in range(1, len(axs)):
        ax = axs[i]
        sample_sequence = []
        for p in ps:
            rescale_value = None
            for key in keys:
                if key[0] == p and np.abs(key[1] - rescale)**2 < 0.0005:
                    rescale_value = key[1]
                    break
            if rescale_value is None:
                raise Exception('Could not rescale value')
            if len(samples_dict[(p, rescale_value)]) > 3:
                counter = Counter(samples_dict[(p, rescale_value)][iters[i-1]])
            else:
                counter = Counter(samples_dict[(p, rescale_value)][i-1])
            if i == len(axs) - 1:
                print(p, counter.most_common(2))
            evals = np.round(ising_offset + evaluate_sparse_pauli_samples(list(counter.keys()), hamiltonian), 2)
            
            # energies = [count * [evals[idx]] for idx, count in enumerate(counter.values()) if evals[idx] < cutoff]
            energies = [count * [evals[idx]] for idx, count in enumerate(counter.values())]
            sample_vals = np.array([x for xs in energies for x in xs])
            if alpha is not None:
                sample_vals = np.sort(sample_vals)[:int(alpha * len(sample_vals))]
            sample_sequence.append(sample_vals)

        ax.hist(sample_sequence, bins=range(cutoff+1), weights=[[1/len(sample_vals)]*len(sample_vals) for sample_vals in sample_sequence], rwidth=1, log=True, label=ps)
        ax.set_xlim(0, cutoff)
        ax.set_ylim(10**-3, 10**0)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=2)
        ax.text(.95, .99, f'Iter = {iters[i-1]+1}', ha='right', va='top', transform=ax.transAxes)
          
    # ax.legend(loc='best', bbox_to_anchor=(0.6, 0.6, 0.4, 0.4))
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.92),  # move legend down
        frameon=True
    )
    return axs


delta_b_fixed = 0.63
delta_g_fixed = 0.16
rescale = 10**(0.0)

ps = [1,3,5]
iters=range(0,5,2)
max_beta_T = 0.15
eps = 0.15

# n = 4000
# alpha = 0.1
# very_large = False
# large = True

for large, very_large in zip([True, False], [False, True]):
    for n, alpha in zip([4000, 4000, 40000], [None, 0.1, 0.1]):

        if very_large:
            filenames = ["test_N7_W5", "test_N8_W5"]
            qubits = [70, 80]
            Ns = [7, 8]
        elif large:
            filenames = ["test_N4_W6", "test_N5_W6", "test_N8_W4"]
            qubits = [48, 60, 64]
            Ns = [4, 5, 8]   
        else:
            filenames = ["test_N2_W2", "trivial", "test_N3_W4", "test_N7_W2", "test_N4_W5"]
            qubits = [8, 18, 24, 28, 40]
            Ns=[2, 3, 3, 7, 4]

        fig, axs = plt.subplots(len(filenames), len(iters) + 1, sharey='row', sharex='col')
        for i in range(len(filenames)):
            plot_several_p_dist(axs[i, :], filenames[i], (2*Ns[i])**-1, delta_b_fixed, delta_g_fixed, n, ps, rescale=rescale, max_beta_T=max_beta_T, iters=iters, eps=eps, alpha=alpha)

        for ax in axs[:, 0]:
            ax.set_ylabel('Sample density' if alpha is None or alpha == 1.0 else 'Sample quasi-density')
            
        for idx, ax in enumerate(axs[:, -1]):
            twin = ax.twinx()
            twin.set_yticks([])
            twin.set_ylabel(f'{qubits[idx]} qubits')
            
        for ax in axs[-1, :]:  
            ax.set_xlabel(r'Energy')  
            
        fig.suptitle(f'$\\Delta_\\beta = {delta_b_fixed}, \\Delta_\\gamma = {delta_g_fixed}, p = {", ".join([str(p) for p in ps])}, n = {n}$', fontsize=16)
        # fig.set_figheight(3 * len(axs[:, 0]))
        # fig.set_figwidth(3.5 * len(axs[0, :]))
        fig.set_figheight(1.5 + 3.3 * len(axs[:, 0]))
        fig.set_figwidth(22)
        plt.tight_layout()

        append_str = (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + (f'.alpha{alpha if alpha is not None else 1.0}')
        figname = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/out/iter_qaoa.{}.db{}.dg{}.n{}{}.png'.format(
            'very_large' if very_large else 'large' if large else 'small',
            delta_b_fixed,
            delta_g_fixed,
            n,
            append_str
        )
        fig.savefig(figname)

