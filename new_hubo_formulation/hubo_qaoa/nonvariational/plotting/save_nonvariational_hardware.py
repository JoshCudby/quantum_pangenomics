import pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from collections import Counter
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator, LogFormatterSciNotation)
from typing import Optional

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


colours = [
    "#FD8153",
    '#5366E0',
    '#4DB78C',
    '#911449',
]

def compare_hardware_and_noiseless(
    axs: list[Axes], 
    filename, copy_numbers, prob,
    db_fixed, dg_fixed, shots,
    p, maxp, rescale,
    max_beta_T:Optional[float]=None, eps:Optional[float]=None, alpha:Optional[float]=None, 
    error_mitigation:Optional[bool]=None, backend:Optional[str]=None,
    iters=None
) -> tuple[list[Axes], list, list[str]]:   
    filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
    graph, n, _, T = gfa_file_to_graph(filepath, copy_numbers)
    hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
    hamiltonian = hamiltonian * norm
    num_qubits: int = hamiltonian.num_qubits

    base_file_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/hardware/nonvariational.hardware.{filename}'
    append_str = (".error_mit" if error_mitigation else "")+(f".backend{backend}" if backend else "") + (f'.db{db_fixed}.dg{dg_fixed}.shots{shots}') + (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + (f'.alpha{alpha}' if alpha is not None else '')
    print(f'Hardware filepath: {base_file_name}{append_str}.pkl')
    with open(f'{base_file_name}{append_str}.pkl', 'rb') as f:
        res = pickle.load(f)

    base_file_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/nonvariational.{filename}.db{db_fixed}.dg{dg_fixed}.ps{maxp}.shots{int(shots*alpha)}'
    append_str = (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + ('.alpha1.0')
    print(f'Simulation filepath: {base_file_name}{append_str}.pkl')
    with open(f'{base_file_name}{append_str}.pkl', 'rb') as f:
        simulation_res = pickle.load(f)
        
    sample_sequence = []
    samples_dicts = {
        'hardware': res['samples_dict'],
        'simulation': simulation_res['samples_dict']
    }
    legend_artists = {}

    if iters is None:
        iters = [0, 2, 4]
    
    cutoff = 25
    ax = axs[0]
    rand_shots = min(shots*10, 40000)
    random_samples = np.random.choice(('0', '1'), (rand_shots, num_qubits), p=(1-prob,prob))
    rand_samples = [''.join(sample) for sample in random_samples]
    rand_vals = np.round(evaluate_sparse_pauli_samples(rand_samples, hamiltonian), 2)
    _,_, rand_patches = ax.hist(
        rand_vals,
        bins=range(cutoff+1), 
        weights=[1/rand_shots]*len(rand_vals), 
        rwidth=1, 
        log=True,
        color='#546072', 
        label='Random'
    )
    
    if rand_patches:
        legend_artists['Random warm-start'] = rand_patches[0]
        
        
    ax.set_xlim(0, cutoff)
    ax.set_ylim(1/shots, 10**0)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.text(.98, .97, 'Iter = 0', ha='right', va='top', transform=ax.transAxes)
    
    
    for i in range(1, len(axs)):
        ax = axs[i]
        sample_sequence = []
        labels_for_datasets = []
        for name in samples_dicts.keys():
            rescale_value = None
            for key in samples_dicts[name].keys():
                if key[0] == p and np.abs(key[1] - rescale)**2 < 0.0005:
                    rescale_value = key[1]
                    break
            if rescale_value is None:
                raise Exception('Could not rescale value')

            counter = Counter(samples_dicts[name][(p, rescale_value)][iters[i-1]])
            if i == len(axs) - 1:
                print(p, counter.most_common(2))
            evals = np.round(evaluate_sparse_pauli_samples(list(counter.keys()), hamiltonian), 2)
            
            energies = [count * [evals[idx]] for idx, count in enumerate(counter.values())]
            sample_vals = np.array([x for xs in energies for x in xs])
            if name == 'simulation':
                sample_sequence.append(sample_vals)
                labels_for_datasets.append('Simulation')
            if alpha is not None and name == 'hardware':
                quasi_sample_vals = np.sort(sample_vals)[:int(alpha * len(sample_vals))]
                sample_sequence.append(quasi_sample_vals)
                labels_for_datasets.append(f'E-M Hardware ($\\alpha = {alpha}$)')

        _, _, patches = ax.hist(
            sample_sequence, 
            bins=range(cutoff+1), 
            weights=[[1/len(sample_vals)]*len(sample_vals) for sample_vals in sample_sequence], 
            rwidth=1, 
            log=True, 
            color=[colours[i] for i in range(len(sample_sequence))]
        )
        for lab, patch in zip(labels_for_datasets, patches):
            if lab not in legend_artists:
                legend_artists[lab] = patch
                
        ax.set_xlim(0, cutoff)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        
        n_ticks = 1+np.floor(np.log10(shots))
        ax.set_ylim(1/shots, 10**0)
        ax.yaxis.set_major_locator(LogLocator(numticks=n_ticks))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        minor_subs = range(2, 10)
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=list(minor_subs), numticks=n_ticks * len(minor_subs))
        )
        
        # ax.tick_params(axis='x', which='major', length=6)
        # ax.tick_params(axis='x', which='minor', length=2)
        ax.text(.98, .97, f'Iter = {iters[i-1]+1}', ha='right', va='top', transform=ax.transAxes)
          
    handles = list(legend_artists.values())
    labels = list(legend_artists.keys())
    return axs, handles, labels




delta_b_fixed = 0.75
delta_g_fixed = 0.30

p = 1
maxp=5
iters=range(0,5,2)
max_beta_T = 0.25
eps = 0.25
backend='ibm_boston'


for filename, qubits, copy_numbers, n, alpha in zip(
    ["test_N3_W4", "test_N4_W5", "test_N4_W5"],
    [12, 15, 15],
    [[2,1,1], [2,1,1,1], [2,1,1,1]],
    [4000, 4000, 8000],
    [0.1, 0.1, 0.05]
):
    cols = int(np.ceil( (len(iters) + 1) ** 0.5 ))
    rows = int(np.floor( (len(iters) + 1) / cols ))
    fig, axs = plt.subplots(rows, cols, figsize=(6.27, 2*rows), sharey='row', sharex='col')
    axs_flat = np.asarray(axs).flat
    axs_flat, legend_handles, legend_labels = compare_hardware_and_noiseless(
        axs_flat, filename, copy_numbers, 1/2, 
        delta_b_fixed, delta_g_fixed,
        n, p, maxp, rescale=1,
        max_beta_T=max_beta_T, 
        iters=iters, eps=eps, alpha=alpha,
        error_mitigation=True, backend=backend,
    )

    fig.suptitle(f'${qubits}$ qubits', fontsize=16, x=0.555)

    fig.supylabel('Sample density', x=0.04, y=0.57)
        
    fig.supxlabel('Energy', y=0.12, x=0.555)
        
        
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),  
        frameon=True,
        bbox_to_anchor=(0.54, 0.02)  
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1.04])

    append_str = (f'.betaT{max_beta_T}' if max_beta_T is not None else '') + (f'.eps{eps}' if eps is not None else '') + (f'.alpha{alpha if alpha is not None else 1.0}')
    figname = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/out/save.iter_qaoa.hardware.{}.db{}.dg{}.p{}.n{}{}.png'.format(
        filename,
        delta_b_fixed,
        delta_g_fixed,
        p,
        n,
        append_str
    )
    fig.savefig(figname, dpi=300)

