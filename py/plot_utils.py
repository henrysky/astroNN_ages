import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy.stats import binned_statistic_2d
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_alpha_age(
    ax,
    fe_h,
    alpha,
    age,
    range=((-0.55, 0.45), (-0.1, 0.4)),
    vmin=0,
    vmax=13,
    bins=[100, 100],
    cbar_label="Age (Gyr)",
    orientation="vertical",
):
    """
    Plots [Fe/H]-[Alpha/M] colored by ages
    """
    assert len(fe_h) == len(alpha) == len(age)
    med, xe, ye, binnum = binned_statistic_2d(
        fe_h,
        alpha,
        age,
        bins=bins,
        range=range,
        statistic=np.median,
    )

    count, _, _, binnum = binned_statistic_2d(
        fe_h,
        alpha,
        age,
        bins=bins,
        range=range,
        statistic="count",
    )
    mappable = ax.pcolormesh(
        xe,
        ye,
        med.T,
        alpha=np.clip(count.T / np.percentile(count, 70), 0.0, 1.0),
        vmin=0,
        vmax=vmax,
        cmap="coolwarm",
        rasterized=True,
    )

    mappable = ax.pcolormesh(
        xe,
        ye,
        med.T,
        alpha=0.0,
        vmin=0,
        vmax=vmax,
        cmap="coolwarm",
    )
    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
    ax.set_ylabel(r"[$\alpha$/M] (dex)")
    ax.set_xlabel("[Fe/H] (dex)")
    divider = make_axes_locatable(ax)
    if orientation == "vertical":
        side = "right"
    elif orientation == "horizontal":
        side = "top"
    else:
        raise ValueError("Impossible orientation")
    cax = divider.append_axes(side, size="5%", pad=0.05)


    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation=orientation
    )
    if orientation == "horizontal":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    cbar.ax.tick_params()
    cbar.set_label(cbar_label)
    return ax


def plot_recon_comparison(ax):
    """
    Plots PSD reconstruction
    """
    return None
