import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# --- Styling for journal-ready figures ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def gradient_ascent_fixed_pstep(ps, rescaling, logp,
                                p0=1.0, r0=1.0,
                                alpha_r=0.25, max_steps=None, tol=1e-8):
    """
    Gradient ascent where p advances exactly +1 index in the ps array each step.
    - ps: 1D sorted array of p values (physical, >0)
    - rescaling: 1D array of rescaling values (physical, >0)
    - logp: 2D array shaped (len(rescaling), len(ps)) containing log10(probabilities).
    - p0, r0: starting physical coordinates (p0 should be in ps or near it).
    - alpha_r: step multiplier for the rescaling update in log-space.
    - max_steps: maximum steps (defaults to len(ps) - start_index).
    - tol: if the absolute change in log(r) is < tol, stop early.
    Returns (p_path, r_path) arrays (both physical-valued).
    """

    # precompute logs
    log_ps = np.log(ps)
    log_res = np.log(rescaling)

    # numeric gradients of logp w.r.t. log_res (axis0) and log_ps (axis1)
    grad_res, grad_p = np.gradient(logp, log_res, log_ps, edge_order=2)

    # find starting p index (closest)
    idx = int(np.argmin(np.abs(ps - p0)))
    # ensure start is not the last index (we need to be able to step forward)
    if idx >= len(ps) - 1:
        idx = len(ps) - 2

    # starting r in log-space (clamped)
    log_r = np.log(r0)
    log_r = np.clip(log_r, log_res[0], log_res[-1])

    # default max_steps
    if max_steps is None:
        max_steps = len(ps) - 1 - idx

    p_path = [ps[idx]]
    r_path = [np.exp(log_r)]

    # helper bilinear interpolator (re-use the one you have)
    def _bilinear(arr, x_vals, y_vals, x, y):
        i = np.searchsorted(x_vals, x) - 1
        j = np.searchsorted(y_vals, y) - 1
        i = np.clip(i, 0, len(x_vals) - 2)
        j = np.clip(j, 0, len(y_vals) - 2)
        x0, x1 = x_vals[i], x_vals[i + 1]
        y0, y1 = y_vals[j], y_vals[j + 1]
        if x1 == x0:
            wx1 = 0.0
        else:
            wx1 = (x - x0) / (x1 - x0)
        wx0 = 1.0 - wx1
        if y1 == y0:
            wy1 = 0.0
        else:
            wy1 = (y - y0) / (y1 - y0)
        wy0 = 1.0 - wy1
        f00 = arr[j, i]
        f10 = arr[j, i + 1]
        f01 = arr[j + 1, i]
        f11 = arr[j + 1, i + 1]
        return (f00 * wx0 * wy0 + f10 * wx1 * wy0 + f01 * wx0 * wy1 + f11 * wx1 * wy1)

    # iterate: at each step increase idx -> idx+1, update log_r using gradient gy interpolated at current (log_ps[idx], log_r)
    for step in range(int(max_steps)):
        if idx >= len(ps) - 1:
            break

        x = log_ps[idx]         # log p at current index
        y = log_r               # current log r

        # interpolate gradient d(logp)/d(log r) at (x,y)
        gy = _bilinear(grad_res, log_ps, log_res, x, y)

        # update log_r by alpha_r * gy
        dlogr = alpha_r * gy
        log_r_new = log_r + dlogr

        # clamp to domain
        log_r_new = np.clip(log_r_new, log_res[0], log_res[-1])

        # stop if movement is negligible
        if abs(log_r_new - log_r) < tol:
            # still advance p by one step if desired, or break.
            # Here we advance p once and then stop (so the path includes the next p)
            idx += 1
            p_path.append(ps[idx])
            r_path.append(np.exp(log_r_new))
            break

        # advance p index by one
        idx += 1
        p_path.append(ps[idx])
        r_path.append(np.exp(log_r_new))

        # commit update
        log_r = log_r_new

        # safety: if we've reached final p, break
        if idx >= len(ps) - 1:
            break

    return np.array(p_path), np.array(r_path)

        

def plot_log_to_ax(ax, filename, maxdb, maxdg, maxr, maxp):
    """
    Plot contourf(log10(probabilities)) on the provided Axes.
    Does NOT create a colorbar (caller will add a shared colorbar).
    """
    pkl_path = (
        f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/'
        f'LR_equal.performance.{filename}.db{maxdb}.dg{maxdg}.rescaling{maxr}.p{maxp}.pkl'
    )
    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)

    probabilities = np.asarray(res['probabilities'])   # shape: (p, rescaling) or similar
    rescaling = np.asarray(res['rescaling'])
    ps = np.asarray(res['ps'])

    # prepare log10 values (safe)
    eps = 1e-9
    logp = np.minimum(0, np.log10(probabilities.T + eps))   # shape (rescaling, p)
    
    print("shape logp:", logp.shape)
    print("any nan:", np.any(np.isnan(logp)))
    print("any inf:", np.any(~np.isfinite(logp)))
    print("min/max:", np.nanmin(logp), np.nanmax(logp))

    # color mapping: values under 1e-4 drawn black
    vmin = -4.0
    vmax = 0.0
    n_levels = 21
    levels = np.linspace(vmin, vmax, n_levels)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('black')   # below vmin -> black
    cmap.set_over('yellow')
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    # contourf: DO NOT clip `logp` here; extend='min' will show under color for < vmin
    im = ax.contourf(
        ps,
        rescaling,
        logp,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend='min'   # critical: values < vmin use cmap.set_under('black')
    )

    # axes scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # nice log tick locators/formatters
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    
    
    ymin = float(np.min(rescaling))
    ymax = float(np.max(rescaling))
    yticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

    # keep only ticks inside range
    yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    
    
    ax.minorticks_off()

    # labels (only x label on bottom row)
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$\Delta_\beta / \Delta_{\beta,\mathrm{fixed}} = \Delta_\gamma / \Delta_{\gamma,\mathrm{fixed}}$')

    # thin spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return im, ps, rescaling, logp


if __name__ == "__main__":
    filenames = ['test_N2_W2', 'trivial', 'test_N4_W5']
    subplot_titles = ['n = 4', 'n = 9', 'n = 15']
    db = 0.75
    dg = 0.30
    r = 3.16
    p = 316

    # Figure: 1 row x 3 cols, width fixed to 6.27 in
    fig_width = 6.27
    fig_height = 2.8  # feel free to tweak; this keeps a similar height as before
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)

    # If axes is length-1, wrap; here it's 3 so ok
    ims = None
    for i, (ax, fname, stitle) in enumerate(zip(axes, filenames, subplot_titles)):
        ims, ps, rescaling, logp = plot_log_to_ax(ax, fname, db, dg, r, p)
        
        p_path, r_path = gradient_ascent_fixed_pstep(ps, rescaling, logp,
                                                 p0=1.0, r0=1.0,
                                                 alpha_r=0.075, max_steps=200, tol=1e-6)
        # plot path: white line with small filled markers and black edge for contrast
        ax.plot(p_path, r_path, '--', color='red', linewidth=1.0, alpha=0.9, zorder=20)
        
        ax.set_title(stitle, fontsize=10, pad=6)

        if i > 0:
            ax.set_ylabel("")          # remove label
            ax.tick_params(labelleft=False)     # remove tick labels (keep ticks aligned)

    # layout: give room for shared colorbar at right and the suptitle
    plt.subplots_adjust(left=0.10, right=0.86, top=0.88, bottom=0.12, wspace=0.28)

    # Shared narrow colorbar to the right of all panels
    cbar = fig.colorbar(ims, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03, extend='min')
    # ticks as integer log10 positions between vmin and vmax
    log_ticks = np.arange(int(np.ceil(-4.0)), int(np.floor(0.0)) + 1).astype(int)  # [-4,-3,-2,-1,0]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(t)}}}$" for t in log_ticks])
    cbar.set_label(r'$p_{\mathrm{opt}}$', rotation=90, va='bottom', labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Overall figure title
    fig.suptitle(r'Performance diagrams for $\Delta_\beta = 0.75, \Delta_\gamma = 0.30$', fontsize=11, y=1.02)

    # Save
    outname = (
        f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/out/'
        f'performance_diagrams.multi.db{np.round(db, 2)}.dg{np.round(dg,2)}.maxp{p}.png'
    )
    fig.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
