import pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from collections import Counter
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


def plot_several_p_dist(
    axs: list[Axes], 
    filename, copy_numbers, prob,
    db_fixed, dg_fixed, shots,
    ps, rescale,
    max_beta_T=None, eps=None, alpha=None,
    iters=None
) -> list[Axes]:
    filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
    graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)
    hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
    hamiltonian = hamiltonian * norm
    num_qubits: int = hamiltonian.num_qubits
    
    try:
        with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/nonvariational.{filename}.db{db_fixed}.dg{dg_fixed}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'rb') as f:
            res = pickle.load(f)
    except Exception:
        with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/nonvariational.{filename}.db{db_fixed}.dg{dg_fixed}.ps{ps[-1]}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'rb') as f:
            res = pickle.load(f) 
    
    sample_sequence = []
    samples_dict: dict[tuple[int, float], list[list[str]]] = res['samples_dict']
    keys = samples_dict.keys()
    energies = res['energies']
    if iters is None:
        iters = [0, 5, 9]

    cutoff=25
    ax = axs[0]
    rand_shots = min(shots*10, 40000)
    random_samples = np.random.choice(('0', '1'), (rand_shots, num_qubits), p=(1-prob,prob))
    rand_samples = [''.join(sample) for sample in random_samples]
    rand_vals = np.round(evaluate_sparse_pauli_samples(rand_samples, hamiltonian), 2)
    ax.hist(rand_vals, bins=range(cutoff+1), weights=[1/rand_shots]*len(rand_vals), rwidth=1, log=True, color='gray')
    ax.set_xlim(0, cutoff)
    ax.set_ylim(1/shots, 10**0)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.text(.95, .99, 'Iter = 0', ha='right', va='top', transform=ax.transAxes)
    
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
            evals = np.round(evaluate_sparse_pauli_samples(list(counter.keys()), hamiltonian), 2)
            
            energies = [count * [evals[idx]] for idx, count in enumerate(counter.values()) if evals[idx] < cutoff]
            sample_vals = np.array([x for xs in energies for x in xs])
            sample_sequence.append(sample_vals)

        ax.hist(sample_sequence, bins=range(cutoff+1), weights=[[1/shots]*len(sample_vals) for sample_vals in sample_sequence], rwidth=1, log=True, label=ps)
        ax.set_xlim(0, cutoff)
        ax.set_ylim(1/shots, 10**0)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=2)
        ax.text(.95, .99, f'Iter = {iters[i-1] + 1}', ha='right', va='top', transform=ax.transAxes)
          
    ax.legend(loc='best', bbox_to_anchor=(0.8, 0.6, 0.2, 0.4))
    return axs


setting = 1

delta_b_fixed, delta_g_fixed = 0.75, 0.30
rescale = 10**(0.0)

n = 4000
ps = [1,3,5]
iters=range(0,5,2)
max_beta_T = 0.15
eps=0.15
alpha=0.1


match setting:
    case 0:
        filenames = ("test_N2_W2", "test_N7_W2", "trivial",  "test_N7_W3", "test_N3_W4")
        copy_numbers = ([1,1], [1,0,0,0,0,0,1], [1,1,1], [1,1,0,0,0,0,1],[2,1,1])
        qubits = (4, 8, 9, 12, 12)
    case 1:
        filenames = ('test_N4_W5', 'test_N7_W4', 'test_N8_W5', 'test_N8_W6')
        copy_numbers = ([2,1,1,1], [1,1,1,0,0,0,1], [1,1,1,1,0,0,0,1], [1,1,0,1,1,1,0,1],)
        qubits = (15, 16, 20, 24)
    # case 2:
    #     filenames = ("test_N2_W2", "test_N7_W2", "trivial",  "test_N7_W3", "test_N3_W4")
    #     copy_numbers = ([1,1], [1,0,0,0,0,0,1], [1,1,1], [1,1,0,0,0,0,1],[2,1,1])
    #     qubits = (4, 8, 9, 12, 12)
    case _:
        raise Exception('Bad setting')      



fig, axs = plt.subplots(len(filenames), len(iters) + 1, sharey='row', sharex='col')

for idx, (f, c) in enumerate(zip(filenames, copy_numbers)):
    plot_several_p_dist(axs[idx, :], f, c, 1/2, delta_b_fixed, delta_g_fixed, n, ps, rescale=rescale, eps=eps, max_beta_T=max_beta_T, iters=iters, alpha=alpha)


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
fig.set_figwidth(1.5 + 3.3 * len(axs[0, :]))
plt.tight_layout()

append_str = (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + (f'.alpha{alpha if alpha is not None else 1.0}')
figname = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/out/iter_qaoa.{}.db{}.dg{}.n{}{}.png'.format(
    'large' if setting == 1 else 'small',
    delta_b_fixed,
    delta_g_fixed,
    n,
    append_str
)
fig.savefig(figname)