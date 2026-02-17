import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# -----------------------------
# Configuration & data loading (adjusted to your paths)
# -----------------------------
n_instances = 3
instance_names = ['test_N2_W2', 'trivial', 'test_N3_W4']
n_values = [8, 18, 24]
p_values = [1, 2, 3, 4, 5]
n_p = len(p_values)

# initial placeholders (will be overwritten by files)
nx, ny = 41, 41
delta_beta = np.linspace(0.0, 1, nx)
delta_gamma = np.linspace(0.0, 1, ny)

data = {}
for instance_index in range(n_instances):
    with open(
        f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/param_exploration/'
        f'LR_unequal.{instance_names[instance_index]}.db{np.round(delta_beta[-1], 2)}.'
        f'dg{np.round(delta_gamma[-1], 2)}.p{p_values[-1]}.pkl', 'rb'
    ) as f:
        res = pickle.load(f)
    p_opts = res['p_opts']
    # update grid coordinates from file if present
    delta_beta = res.get('delta_bs', delta_beta)
    delta_gamma = res.get('delta_gs', delta_gamma)
    for p_index in range(n_p):
        data[(instance_index, p_index)] = p_opts[p_index, :, :]

# --- Plot params ---
fig_width_in = 6.27  # publication width requested (inches)

# sensible per-row height + margins (tweak per journal if needed)
per_row_height = 0.95   # inches per heatmap row
top_margin = 0.6        # inches reserved above heatmaps for suptitle
bottom_margin = 0.6     # inches reserved below heatmaps for Δβ label
fig_height_in = per_row_height * n_instances + top_margin + bottom_margin

fig = plt.figure(figsize=(fig_width_in, fig_height_in))

# GridSpec for just the heatmaps (no summary row)
gs = GridSpec(
    nrows=n_instances,
    ncols=n_p,
    height_ratios=[1] * n_instances,
    hspace=0.28,
    wspace=0.12,
    figure=fig
)


# color limits: floor at 1e-4
vmin = 1e-4
vmax = 1

cmap = plt.get_cmap('viridis').copy()
cmap.set_under('black')
cmap.set_over('yellow')

pcm = None
for inst in range(n_instances):
    for p_idx, p in enumerate(p_values):
        ax = fig.add_subplot(gs[inst, p_idx])

        arr = np.array(data[(inst, p_idx)])
        # arr_plot = np.clip(arr, vmin, None)

        pcm = ax.pcolormesh(
            delta_beta,
            delta_gamma,
            arr.T,
            shading='auto',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
        )

        # mark maximum (use original arr to find the true max location)
        ix, iy = np.unravel_index(np.argmax(arr), arr.shape)
        x_max = delta_beta[ix]
        y_max = delta_gamma[iy]
        ax.plot(x_max, y_max, marker='x', color='red', markersize=6, zorder=10)

        # ticks: only outer ticks shown to reduce clutter
        beta_ticks = np.linspace(delta_beta.min(), delta_beta.max(), 3)
        gamma_ticks = np.linspace(delta_gamma.min(), delta_gamma.max(), 3)

        ax.set_xticks(beta_ticks)
        if inst != n_instances - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{v:.2g}" for v in beta_ticks], fontsize=8)

        ax.set_yticks(gamma_ticks)
        if p_idx == 0:
            ax.set_yticklabels([f"{v:.2g}" for v in gamma_ticks], fontsize=8)
        else:
            ax.set_yticklabels([])

        # column titles on the top row: smaller font and slightly raised
        if inst == 0:
            ax.set_title(f"p = {p}", fontsize=9, pad=8)

        # thin axis spines for a clean look
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

# --- Reserve margins to prevent overlap (top kept generous) ---
plt.subplots_adjust(left=0.16, right=0.86, top=0.92, bottom=0.08)

# --- heatmap bbox for positioning shared labels/colorbar ---
heatmap_grid = gs[0:n_instances, 0:len(p_values)]
heatmap_bbox = heatmap_grid.get_position(fig)

# --- Shared axis labels (outside axes), tuned to avoid collisions ---
# move Δγ further left, instance labels a bit closer to heatmaps
xpad_label = 0.075    # how far left Δγ sits from heatmaps (fig coords)
ypad_label = 0.055   # how far below Δβ sits from heatmaps (fig coords)
xpad_instance = 0.050


# Δβ centered under the heatmaps (moved down a bit)
x_center = heatmap_bbox.x0 + 0.5 * heatmap_bbox.width
y_below_heatmaps = heatmap_bbox.y0 - ypad_label
y_below_heatmaps = max(y_below_heatmaps, 0.02)
fig.text(x_center, y_below_heatmaps, r'$\Delta_\beta$', ha='center', va='top', fontsize=10)

# Δγ vertical left of heatmaps (further left so instance labels don't collide)
x_left_of_heatmaps = heatmap_bbox.x0 - xpad_label
x_left_of_heatmaps = max(x_left_of_heatmaps, 0.01)
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

# --- Narrow colorbar, clamped to the right of heatmaps ---
cbar_gap = 0.02
cbar_width = 0.018
cbar_left = heatmap_bbox.x1 + cbar_gap
max_left = 0.98 - cbar_width
if cbar_left + cbar_width > 0.98:
    cbar_left = max_left

cbar_bottom = heatmap_bbox.y0
cbar_height = heatmap_bbox.height
cbar_ax = fig.add_axes((cbar_left, cbar_bottom, cbar_width, cbar_height))

# colorbar with log ticks at powers of ten (>= vmin)
cb = fig.colorbar(pcm, cax=cbar_ax, format=LogFormatterSciNotation())
cb.set_label(r"$p_{\mathrm{opt}}$", fontsize=9)
cb.outline.set_linewidth(0.8)
cb.ax.tick_params(width=0.8, labelsize=6)

# create tick list as powers of ten from vmin to vmax
locator = LogLocator(base=10.0)
ticks = locator.tick_values(vmin, vmax)
ticks = [t for t in ticks if t >= vmin and t <= vmax]
# keep just a few ticks if there are many
if len(ticks) > 6:
    ticks = ticks[:6]
cb.set_ticks(ticks)
cb.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in ticks])


# --- Suptitle well above the columns (no overlap) ---
fig.suptitle(
    r"Heatmaps of $p_{\mathrm{opt}}(\Delta_\beta, \Delta_\gamma)$",
    fontsize=12,
    y=1.03   # near top of figure, independent of heatmap bbox
)

# --- Save/show (300 dpi) ---
figname = (
    f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/out/'
    f'param_exploration.linear.db{np.round(delta_beta[-1], 2)}.dg{np.round(delta_gamma[-1],2)}.p{p_values[-1]}.png'
)
fig.savefig(figname, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()