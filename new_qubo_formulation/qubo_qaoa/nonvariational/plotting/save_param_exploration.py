import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

# -----------------------------
# Configuration
# -----------------------------
n_instances = 3          # number of problem instances
instance_names = ['test_N2_W2', 'trivial', 'test_N3_W4']
n_values = [8, 18, 24]
p_values = [1]
n_p = len(p_values)

nx, ny = 41, 41          # resolution of (Δβ, Δγ) grid
# delta_beta = np.logspace(-1, 0, nx, base=10)
# delta_gamma = np.logspace(-1, -0.5, ny, base=10)
delta_beta = np.logspace(-1.5, 0.5, 41, base=10)
delta_gamma = np.logspace(-1.5, -0.5, 41, base=10)

# X, Y = np.meshgrid(delta_beta, delta_gamma)
# extent = [
#     delta_beta.min(),  delta_beta.max(),
#     delta_gamma.min(), delta_gamma.max()
# ]

# -----------------------------
data = {}
for instance_index in range(n_instances):
    with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/param_exploration/LR_unequal.{instance_names[instance_index]}.db{np.round(delta_beta[-1], 2)}.dg{np.round(delta_gamma[-1], 2)}.p{p_values[-1]}.pkl', 'rb') as f:
        res = pickle.load(f)
    p_opts = res['p_opts']
    delta_beta = res['delta_bs']
    delta_gamma = res['delta_gs']
    for p_index in range(n_p):
        data[(instance_index, p_index)] = p_opts[p_index, :, :]




# -----------------------------
# Summary metric: best p_opt
# -----------------------------
best_vals = np.zeros((n_instances, n_p))
for inst in range(n_instances):
    for p_idx in range(n_p):
        best_vals[inst, p_idx] = data[(inst, p_idx)].max()

# -----------------------------
# Figure layout (fixed)
# -----------------------------
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(
    nrows=n_instances + 2,
    ncols=n_p,
    height_ratios=[1]*n_instances + [0.2, 0.5],
    hspace=0.25,
    wspace=0.12,
    figure=fig
)

# Color normalization (clip tiny or zero values for LogNorm)
eps = 1e-4
vmin = min(np.clip(arr, eps, None).min() for arr in data.values())
vmax = max(arr.max() for arr in data.values())
cmap = plt.get_cmap('viridis')
cmap.set_over('yellow')
cmap.set_under('black')

# # Formatter to convert log10 ticks back to readable numbers (scientific or simple)
# def tick_formatter(val, pos):
#     # val is log10(value)
#     real = 10.0 ** val
#     # use compact formatting: 1e-1, 3.16e-1, 1.0 etc.
#     if real >= 1:
#         return f"{real:.0f}"
#     else:
#         return f"{real:.0g}"

# fmt = FuncFormatter(tick_formatter)

# -----------------------------
# Heatmaps
# -----------------------------
pcm = None
for inst in range(n_instances):
    for p_idx, p in enumerate(p_values):
        ax = fig.add_subplot(gs[inst, p_idx])

        # We plot in log10 space for axis but keep the data in linear probability space
        # note: pcolormesh expects X edges (len = nx+1) and Y edges (ny+1).
        # using np.log10(delta_beta) and np.log10(delta_gamma) here (as in your code).
        pcm = ax.pcolormesh(
            np.log10(delta_beta),            # x (centers or edges - works for shading='auto')
            np.log10(delta_gamma),           # y
            data[(inst, p_idx)].T,          # shape (ny, nx) ; keep .T if that matched your data
            shading='auto',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
        )

        ix, iy = np.unravel_index(np.argmax(data[(inst, p_idx)]), data[(inst, p_idx)].shape)
        x_max = np.log10(delta_beta)[ix]
        y_max = np.log10(delta_gamma)[iy]
        ax.plot(
            x_max, y_max,
            marker='x',
            color='red',
            markersize=8,
            # markeredgecolor='white',
            zorder=10
        )

        # Choose tick locations in log10 space (coarse to avoid clutter)
        beta_ticks  = np.linspace(np.log10(delta_beta.min()),  np.log10(delta_beta.max()), 3)
        gamma_ticks = np.linspace(np.log10(delta_gamma.min()), np.log10(delta_gamma.max()), 3)
        # Only show ticks on outer panels; otherwise hide them completely
        ax.set_xticks(beta_ticks)
        if not inst == n_instances - 1:
            ax.set_xticklabels([])

        ax.set_yticks(gamma_ticks)
        if p_idx == 0:
            # ax.set_yticks(gamma_ticks)
            # ax.yaxis.set_major_formatter(fmt)
            # instance name as y-label but smaller so it doesn't clash with figure-level label
            # ax.set_ylabel(instance_names[inst], fontsize=9, labelpad=8)
            pass
        else:
            ax.set_yticklabels([])

        # Label columns with p on the bottom row (keeps top tidy)
        if inst == 0:
            # put a small title above each column to indicate p (optional)
            ax.set_title(f"p = {p}", fontsize=16, pad=6)




heatmap_grid = gs[0:n_instances, 0:n_p]                 # GridSpec slice for heatmaps
heatmap_bbox = heatmap_grid.get_position(fig)          # Bbox in figure coordinates

# Slight offsets for label placement (tweak if you want more/less margin)
xpad_label = 0.065   # horizontal offset for y-label from heatmap left edge
ypad_label = 0.01    # vertical offset for x-label below heatmaps
xpad_instance = 0.045 # horizontal offset for instance names left of heatmap rows

# 3) Put the Δβ label centered under the heatmap block but above the summary plot
x_center = heatmap_bbox.x0 + 0.5 * heatmap_bbox.width - 0.05
y_below_heatmaps = heatmap_bbox.y0 - ypad_label
fig.text(x_center, y_below_heatmaps, r'$\log_{10}(\Delta_\beta)$', ha='center', va='top', fontsize=16)

# 4) Put the Δγ label centered to the left of the heatmap block (vertical text)
x_left_of_heatmaps = heatmap_bbox.x0 - xpad_label
y_center = heatmap_bbox.y0 + 0.5 * heatmap_bbox.height
fig.text(x_left_of_heatmaps, y_center, r'$\log_{10}(\Delta_\gamma)$', ha='right', va='center',
         rotation='vertical', fontsize=16)

# 5) Place instance names to the left of each row using the row bbox (so they don't clash with the Δγ label)
for inst in range(n_instances):
    # Get bbox of the row (row index = inst)
    row_grid = gs[inst, 0:n_p]
    row_bbox = row_grid.get_position(fig)
    # Place the instance name slightly to the left of the heatmap row bbox
    instance_x = row_bbox.x0 - xpad_instance
    instance_y = row_bbox.y0 + 0.5 * row_bbox.height + 0.03
    fig.text(instance_x, instance_y, f'n = {n_values[inst]}',
             ha='right', va='center', fontsize=14, rotation='vertical')


# -----------------------------
# Shared colorbar (moved left a bit to make space for legend)
# -----------------------------
# reserve right margin for legend; colorbar inside that reserved area
cbar_ax = fig.add_axes((heatmap_bbox.x1-0.08, heatmap_bbox.y0+0.031, 0.02, heatmap_bbox.height))  # [left, bottom, width, height] in figure coords
cb = fig.colorbar(pcm, cax=cbar_ax)
cb.set_label(r"$p_{\mathrm{opt}}$", fontsize=14)

# -----------------------------
# Summary line plot
# -----------------------------
ax_summary = fig.add_subplot(gs[-1, :])
for inst in range(n_instances):
    ax_summary.plot(
        np.round(np.log10(np.array(p_values)), 1),
        best_vals[inst],
        marker="o",
        label=n_values[inst],
    )

ax_summary.set_xlabel(r"$\log_{10}(p)$", fontsize=14)
ax_summary.set_ylabel(r"Best $p_{\mathrm{opt}}$", fontsize=14)
ax_summary.set_ylim(0, 1.02)
ax_summary.set_xticks(np.round(np.log10(np.array(p_values)), 1))
ax_summary.set_xticklabels(np.array(p_values))

# Legend placed outside to the right of the figure (in the reserved right area)
ax_summary.legend(
    title="Number of qubits",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)


# -----------------------------
# Title, layout tweaks and save
# -----------------------------
fig.suptitle(
    r"Heatmaps of $p_{\mathrm{opt}}(\Delta_\beta, \Delta_\gamma)$",
    fontsize=18
)

# Make room: leave extra bottom space for the figure-level x label and summary plot,
# and extra right space for the legend + colorbar.
plt.subplots_adjust(left=0.12, right=0.80, top=0.92, bottom=0.14)


figname = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/out/param_exploration.db{np.round(delta_beta[-1], 2)}.dg{np.round(delta_gamma[-1],2)}.p{p_values[-1]}.png'
fig.savefig(figname, dpi=200, bbox_inches='tight')
plt.show()
