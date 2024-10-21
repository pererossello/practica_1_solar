import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors



def scatter_density(fig, ax, x, y, step=1, n_bins=100, cmap='turbo', s=1., x_lim=None, y_lim=None):

    x = x.ravel()[::step]
    y = y.ravel()[::step]

    if x_lim is None:
        min_x, max_x = np.nanmin(x), np.nanmax(x)
    else: 
        min_x, max_x = x_lim
    if y_lim is None:
        min_y, max_y = np.nanmin(y), np.nanmax(y)
    else:
        min_y, max_y = y_lim
    
    hist, x_edges, y_edges = np.histogram2d(
        x, y, bins=n_bins, 
        range=[[min_x, max_x], [min_y, max_y]],
        density=False
    )
    hist = np.log10(hist+1)
    vmin, vmax = np.min(hist), np.max(hist)
    
    edges1 = np.linspace(min_x, max_x, n_bins)
    edges2 = np.linspace(min_y, max_y, n_bins)

    delta_edge1 = np.diff(edges1)[0]
    idxs1 = np.int32((x-min_x)/delta_edge1)

    delta_edge2 = np.diff(edges2)[0]
    idxs2 = np.int32((y-min_y)/delta_edge2)

    hist = hist[idxs1, idxs2]

    ax.scatter(x, y, s=s, lw=0., c=hist, cmap=cmap)


    cmap = plt.get_cmap(cmap)
    vlim = np.linspace(vmin, vmax, 10)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(vlim)  

    pos = ax.get_position()

    cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, 0.015])  

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')  # 

    cbar_ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, direction='in')


    cbar_ax.set_xlabel(r'$\log_{10}$ N')


    cbar_ax.xaxis.set_label_position("top")

def plot_cubes(
    list_of_arrays,
    cmap="seismic_r",
    figsize=5,
    idx=0,
    width=1,
    vlim=None,
    wspace=0.05,
    return_ims=False,
):

    if not isinstance(list_of_arrays, list):
        list_of_arrays = [list_of_arrays]
    M = len(list_of_arrays)

    if not isinstance(cmap, list):
        cmap = [cmap] * M

    if vlim is not None:
        if isinstance(vlim, (float, int)):
            vlim = [(-vlim, vlim)] * M
        elif isinstance(vlim, list):
            for i, vlim_item in enumerate(vlim):
                if isinstance(vlim_item, (int, float)):
                    vlim[i] = (-vlim_item, vlim_item)
        elif isinstance(vlim, tuple):
            vlim = [(vlim[0], vlim[1])] * M

    if M < 4:
        n_col = M
        n_row = 1
    elif M == 4:
        n_col = 2
        n_row = 2

    ratio = n_col / n_row
    figsize_fact = figsize / np.sqrt(ratio)
    fig, axs = plt.subplots(n_row, n_col, figsize=(figsize_fact * ratio, figsize_fact))
    plt.subplots_adjust(wspace=wspace, hspace=0.05)

    ims = []
    axs_flat = axs.flatten() if M > 1 else [axs]
    for i, ax in enumerate(axs_flat):
        ax.set_xticks([])
        ax.set_yticks([])

        arr = list_of_arrays[i]
        arr, min_val, max_val = get_projection(arr, idx)

        vmin = min_val if vlim is None else vlim[i][0]
        vmax = max_val if vlim is None else vlim[i][1]

        im = ax.imshow(arr.T, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap[i])
        ims.append(im)

    if return_ims:
        return fig, axs, ims

    return fig, axs


def get_projection(array, idx):

    N = array.shape[0]

    if idx > N - 1:
        raise ValueError("idx should be smaller than N-1")

    matrix = array[idx, :, :]

    min_val = np.min(matrix)
    max_val = np.max(matrix)
    # abs_max_val = np.max([np.abs(min_val), np.abs(max_val)])

    return matrix, min_val, max_val