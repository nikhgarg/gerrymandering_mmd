import pandas as pd
from visualization.settings import *
import pandas as pd
import numpy as np
from optimize.analyze_results import *
import seaborn as sns
from generic.latexify import *


# method_names = {"thiele_approvalindependent": "Winner takes all", "thiele_pav": "PAV", "thiele_squared": "Thiele squared", "stv": "STV"}
method_names = {
    "thiele_approvalindependent": "Winner takes all",
    "thiele_independent": "Winner takes all",
    "thiele_pav": "STV and PAV",
    "thiele_squared": "Thiele squared",
    "stv": "STV",
}


def plot_all_state_distribution_generic(
    distributions,
    prop_val=None,
    do_vertical_integers=False,
    ymin=None,
    ymax=None,
    xbins=None,
    legend=True,
    xlabel="Republican seat share",
    do_zoom=False,
    do_broken_axes=False,
    bbox_to_anchor=(0.95, 0),
    loc="lower left",
    ax=None,
    set_ylim=True,
    weight_by_state=None,
    party_colors=False,
    legendncol=1,
    legendfontsize=20,
    do_abs_after_combining=False,  # for the D advantage plot
):
    if xbins is None:
        xbins = list(np.linspace(1.0 / 53, 1, 100))  # np.arange(0, 1.001, .02)
        for k in range(2, 54):
            xbins.extend([float(l) / k for l in range(1, k)])
        xbins = list(sorted(set(xbins)))
    # print(len(xbins))
    full_distribution = np.zeros(
        (len(xbins), distributions[list(distributions.keys())[0]].shape[0])
    )
    pal = sns.color_palette()
    if party_colors:
        if distributions[list(distributions.keys())[0]].shape[0] == 4:
            pal = [
                sns.color_palette("RdYlBu", 10)[0],
                sns.color_palette("PRGn", 10)[2],
                sns.color_palette("PRGn", 10)[0 - 3],
                sns.color_palette("RdYlBu", 10)[-1],
            ]
        elif distributions[list(distributions.keys())[0]].shape[0] == 2:
            pal = [
                sns.color_palette("RdYlBu", 10)[0],
                sns.color_palette("RdYlBu", 10)[-1],
            ]
        elif distributions[list(distributions.keys())[0]].shape[0] == 3:
            pal = [
                sns.color_palette("RdYlBu", 10)[0],
                sns.color_palette("PRGn", 10)[0 - 3],
                sns.color_palette("RdYlBu", 10)[-1],
            ]
    total_seats = 0
    ints = list(range(1, 6))
    for state in distributions:
        if weight_by_state is None:
            weightstate = state_constants[state]["seats"]
        else:
            weightstate = weight_by_state[state]
        seat_fraction = np.array(distributions[state].columns) / weightstate
        state_stats = distributions[state].values.T
        if np.isnan(np.array(state_stats)).any():
            print("skipping state bc nan: ", state)
            continue
        state_distr = state_stats[
            np.argmin(np.abs(np.subtract.outer(xbins, seat_fraction)), axis=1)
        ]

        full_distribution += state_distr * weightstate
        total_seats += weightstate
    if do_abs_after_combining:
        full_distribution = np.abs(full_distribution)
    if ymin is None:
        ymin = np.min(full_distribution / total_seats)
    if ymax is None:
        ymax = np.max(full_distribution / total_seats)

    if ax is None:
        ax = (
            pd.DataFrame(
                full_distribution,
                columns=distributions[list(distributions.keys())[0]].index,
                index=xbins,
            )
            / total_seats
        ).plot(style=["-", "--", "-."], linewidth=2, color=pal)
        if set_ylim:
            ax.set_ylim((ymin * 0.9, ymax * 1.1))
    else:
        (
            pd.DataFrame(
                full_distribution,
                columns=distributions[list(distributions.keys())[0]].index,
                index=xbins,
            )
            / total_seats
        ).plot(style=["-", "--", "-."], linewidth=2, ax=ax, color=pal)

    if do_vertical_integers:
        for y in set(ints):
            linewidth = 1 if y <= 5 else 0  # 5/y
            ax.vlines(1.0 / y, ymin, ymax, linewidth=linewidth)

    if prop_val is not None:
        ax.hlines(prop_val, 0, 1, linewidth=2, linestyle="--")
    ax.set_ylabel(xlabel, fontsize=20)
    ax.set_xlabel("Districts / Seats", fontsize=20)
    sns.despine()
    if legend:
        ax.legend(
            frameon=False,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            ncol=legendncol,
            fontsize=legendfontsize,
        )
    else:
        ax.legend([], frameon=False)

    if do_zoom:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        axins = inset_axes(ax, width=3, height=2, loc="upper right")

        (
            pd.DataFrame(
                full_distribution,
                columns=distributions[list(distributions.keys())[0]].index,
                index=xbins,
            )
            / total_seats
        ).plot(style=["-", "--", "-."], linewidth=2, ax=axins)
        axins.set_xlim(0, 1)
        axins.set_ylim(0.02, 0.06)
        axins.legend([], frameon=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    return plt.gca()  # full_distribution
