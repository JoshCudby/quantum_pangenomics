import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

# -----------------------------
# Configuration / data setup (unchanged)
# -----------------------------
instance_names = ["test_N2_W2", "test_N7_W2", "trivial", 'test_N3_W4', "test_N4_W5"]
n_instances = len(instance_names)
n_values = [4, 8, 9, 12, 15]
p_values = [1, 2, 3, 4, 5]
n_p = len(p_values)

delta_beta = None
delta_gamma = None
db_max = 1.0
dg_max = 2.0

data = {}
for instance_index in range(n_instances):
    with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/LR_unequal.{instance_names[instance_index]}.db{np.round(db_max, 2)}.dg{np.round(dg_max, 2)}.p{p_values[-1]}.pkl', 'rb') as f:
        res = pickle.load(f)
    p_opts = res['p_opts']
    delta_beta = res['delta_bs']
    delta_gamma = res['delta_gs']
    for p_index in range(n_p):
        data[(instance_index, p_index)] = p_opts[p_index, :, :]

if delta_beta is None or delta_gamma is None:
    raise Exception('No db or dg data')

# -----------------------------
# Summary metric: best p_opt (kept)
# -----------------------------
best_vals = np.zeros((n_instances, n_p))
for inst in range(n_instances):
    for p_idx in range(n_p):
        best_vals[inst, p_idx] = data[(inst, p_idx)].max()

# -----------------------------
# Figure layout (updated)
# -----------------------------
fig_width_in = 6.27  # publication width requested (inches)

# sensible per-row height + margins (tweak per journal if needed)
per_row_height = 0.95   # inches per heatmap row
top_margin = 0.6        # inches reserved above heatmaps for suptitle
bottom_margin = 0.6     # inches reserved below heatmaps for Δβ label
fig_height_in = per_row_height * n_instances + top_margin + bottom_margin

fig = plt.figure(figsize=(fig_width_in, fig_height_in))

# GridSpec now only for heatmaps (no summary plot)
gs = GridSpec(
    nrows=n_instances,
    ncols=n_p,
    height_ratios=[1] * n_instances,
    hspace=0.28,
    wspace=0.12,
    figure=fig
)

# Color normalization (clip tiny or zero values for LogNorm)
eps = 1e-4
vmin = min(np.clip(arr, eps, None).min() for arr in data.values())
vmax = 1
cmap = plt.get_cmap('viridis').copy()
cmap.set_over('yellow')
cmap.set_under('black')

# -----------------------------
# Heatmaps
# -----------------------------
pcm = None
for inst in range(n_instances):
    for p_idx, p in enumerate(p_values):
        ax = fig.add_subplot(gs[inst, p_idx])

        pcm = ax.pcolormesh(
            delta_beta,
            delta_gamma,
            data[(inst, p_idx)].T,
            shading='auto',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
        )

        # mark the maximum
        ix, iy = np.unravel_index(np.argmax(data[(inst, p_idx)]), data[(inst, p_idx)].shape)
        x_max = delta_beta[ix]
        y_max = delta_gamma[iy]
        ax.plot(x_max, y_max, marker='x', color='red', markersize=7, zorder=10)

        # ticks: show only outer ticks to avoid clutter
        beta_ticks = np.linspace(delta_beta.min(), delta_beta.max(), 3)
        gamma_ticks = np.linspace(delta_gamma.min(), delta_gamma.max(), 3)

        ax.set_xticks(beta_ticks)
        if inst != n_instances - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{v:.2g}" for v in beta_ticks], fontsize=7)

        ax.set_yticks(gamma_ticks)
        if p_idx == 0:
            ax.set_yticklabels([f"{v:.2g}" for v in gamma_ticks], fontsize=7)
        else:
            ax.set_yticklabels([])

        # Label columns with p - smaller font to avoid collision with suptitle
        if inst == 0:
            ax.set_title(f"p = {p}", fontsize=9, pad=6)

# -----------------------------
# Reserve margins so that labels & colorbar don't overlap:
# We'll adjust axis positions after plotting using the heatmap bbox
# -----------------------------
# initial margins via subplots_adjust (keeps space for instance labels & colorbar)
# right is intentionally less than 1.0 to leave room for colorbar
plt.subplots_adjust(left=0.16, right=0.86, top=0.92, bottom=0.08)

# get bbox of the entire heatmap block (all rows x all cols)
heatmap_grid = gs[0:n_instances, 0:n_p]
heatmap_bbox = heatmap_grid.get_position(fig)  # in figure coordinates

# -----------------------------
# Shared axis labels & instance labels (aligned to heatmap_bbox)
# -----------------------------
# Figure-coord offsets (bigger so labels sit outside heatmaps)
xpad_label = 0.06    # how far left Δγ sits from heatmaps (fig coords)
ypad_label = 0.035   # how far below Δβ sits from heatmaps (fig coords)
xpad_instance = 0.035

# Δβ label centered under the heatmap block (move further down)
x_center = heatmap_bbox.x0 + 0.5 * heatmap_bbox.width
y_below_heatmaps = heatmap_bbox.y0 - ypad_label
# Clamp so label doesn't go below figure
y_below_heatmaps = max(y_below_heatmaps, 0.02)
fig.text(x_center, y_below_heatmaps, r'$\Delta_\beta$', ha='center', va='top', fontsize=10)

# Δγ label centered vertically to the left of the heatmap block (move further left)
x_left_of_heatmaps = heatmap_bbox.x0 - xpad_label
x_left_of_heatmaps = max(x_left_of_heatmaps, 0.01)  # clamp inside figure
y_center = heatmap_bbox.y0 + 0.5 * heatmap_bbox.height
fig.text(x_left_of_heatmaps, y_center, r'$\Delta_\gamma$', ha='right', va='center',
         rotation='vertical', fontsize=10)

# Instance labels placed left of each row, vertically centered on the row bbox
for inst in range(n_instances):
    row_grid = gs[inst, 0:n_p]
    row_bbox = row_grid.get_position(fig)
    instance_x = row_bbox.x0 - xpad_instance
    instance_x = max(instance_x, 0.01)
    instance_y = row_bbox.y0 + 0.5 * row_bbox.height
    fig.text(instance_x, instance_y, f'n = {n_values[inst]}',
             ha='right', va='center', fontsize=9, rotation='vertical')

# -----------------------------
# Shared colorbar (to the right of the heatmap block), with safety clamp
# -----------------------------
cbar_gap = 0.02
cbar_width = 0.018

cbar_left = heatmap_bbox.x1 + cbar_gap
# safety clamp: ensure we don't put it off the figure or overlapping the heatmaps (if subplots_adjust was too tight)
max_allowed_left = 0.98 - cbar_width
if cbar_left + cbar_width > 0.98:
    cbar_left = max_allowed_left
if cbar_left <= heatmap_bbox.x1 - 1e-6:
    # if still problematic (highly unlikely), place it at a sane fixed pos to the right
    cbar_left = 0.88

cbar_bottom = heatmap_bbox.y0
cbar_height = heatmap_bbox.height
cbar_ax = fig.add_axes((cbar_left, cbar_bottom, cbar_width, cbar_height))
cb = fig.colorbar(pcm, cax=cbar_ax)
cb.set_label(r"$p_{\mathrm{opt}}$", fontsize=8)
cb.ax.tick_params()
cb.outline.set_linewidth(0.8)
cb.ax.tick_params(width=0.8, labelsize=6)

fig.suptitle(
    r"Heatmaps of $p_{\mathrm{opt}}(\Delta_\beta, \Delta_\gamma)$",
    fontsize=12,
    y=0.985   # near top of figure, independent of heatmap bbox
)

# -----------------------------
# Save at requested width and 300 dpi
# -----------------------------
figname = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/out/param_exploration.linear.db{np.round(delta_beta[-1], 2)}.dg{np.round(delta_gamma[-1],2)}.p{p_values[-1]}.png'
fig.savefig(figname, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
