import pandas as pd
from visualization.settings import *
import pandas as pd
import numpy as np
from optimize.analyze_results import *
import seaborn as sns
from generic.latexify import *


# method_names = {"thiele_approvalindependent": "Winner takes all", "thiele_pav": "PAV", "thiele_squared": "Thiele squared", "stv": "STV"}
method_names = {
    "thiele_approvalindependent": "Winner-take-all",
    "thiele_independent": "Winner-take-all",
    "thiele_pav": "STV and PAV",
    "thiele_squared": "Thiele squared",
    "stv": "STV",
}

#1/2/2026 to plot the difference between most rep and most dem across all states, for a given noise level
def plot_all_state_rep_dem_gap(
    distributions,
    rep_label="Most Republican",
    dem_label="Most Democratic",
    xbins=None,
    ymin=None,
    ymax=None,
    legend=True,
    label="Abs(Most Rep - Most Dem)",
    xlabel="Districts / Seats",
    ylabel="Abs(Most Republican - Most Democratic)",
    ax=None,
    weight_by_state=None,
    do_vertical_integers=False,
):
    if xbins is None:
        xbins = list(np.linspace(1.0 / 53, 1, 100))
        for k in range(2, 54):
            xbins.extend([float(l) / k for l in range(1, k)])
        xbins = list(sorted(set(xbins)))

    sample_state = list(distributions.keys())[0]
    row_index = list(distributions[sample_state].index)
    rep_i = row_index.index(rep_label)
    dem_i = row_index.index(dem_label)

    full_distribution = np.zeros((len(xbins), len(row_index)))
    total_seats = 0

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

    gap = np.abs(full_distribution[:, rep_i] - full_distribution[:, dem_i]) / total_seats

    if ymin is None:
        ymin = np.min(gap)
    if ymax is None:
        ymax = np.max(gap)

    if ax is None:
        ax = plt.gca()

    ax.plot(xbins, gap, linewidth=2, label=label)

    if do_vertical_integers:
        for y in set(range(1, 6)):
            linewidth = 1 if y <= 5 else 0
            ax.vlines(1.0 / y, ymin, ymax, linewidth=linewidth)

    ax.set_xlim(min(xbins), max(xbins))
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    sns.despine()

    if legend:
        ax.legend(frameon=False)

    return ax

#1/3/2026 boxplot of the gerrymandering gap per state
def plot_all_state_rep_dem_gap_boxplot(
    distributions,
    rep_label="Most Republican",
    dem_label="Most Democratic",
    xbins=None,
    ymin=None,
    ymax=None,
    xlabel="Districts / Seats",
    ylabel="Gerrymandering range",
    ax=None,
    weight_by_state=None,
    do_vertical_integers=False,
    showfliers=False,
    box_alpha=0.6,
    xbin_num = 100
):
    if xbins is None:
        xbins = list(np.linspace(1.0 / 53, 1, xbin_num))
        # for k in range(2, 6):
            # xbins.extend([float(1) / k])
            # xbins.extend([float(l) / k for l in range(1, k)])
        xbins = list(sorted(set(xbins)))

    sample_state = list(distributions.keys())[0]
    row_index = list(distributions[sample_state].index)
    rep_i = row_index.index(rep_label)
    dem_i = row_index.index(dem_label)

    gaps_by_xbin = [[] for _ in xbins]

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

        idx = np.argmin(np.abs(np.subtract.outer(xbins, seat_fraction)), axis=1)
        state_distr = state_stats[idx]
        state_gap = np.abs(state_distr[:, rep_i] - state_distr[:, dem_i])

        for i, gap_val in enumerate(state_gap):
            gaps_by_xbin[i].append(gap_val)

    if ax is None:
        ax = plt.gca()

    if len(xbins) > 1:
        width = .8 * np.min(np.diff(sorted(xbins)))
    else:
        width = 0.02

    # bp = ax.boxplot(
    #     gaps_by_xbin,
    #     positions=xbins,
    #     widths=width,
    #     patch_artist=True,
    #     showfliers=True,
    #     flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="black"),
    #     manage_ticks=False,
    # )

    # for box in bp["boxes"]:
    #     box.set_alpha(box_alpha)

    parts = ax.violinplot(
        gaps_by_xbin,
        positions=xbins,
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.6)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)


    if ymin is None:
        ymin = np.nanmin([np.nanmin(g) for g in gaps_by_xbin if len(g) > 0])
    if ymax is None:
        ymax = np.nanmax([np.nanmax(g) for g in gaps_by_xbin if len(g) > 0])

    if do_vertical_integers:
        for y in set(range(1, 6)):
            linewidth = 1 if y <= 5 else 0
            ax.vlines(1.0 / y, ymin, ymax, linewidth=linewidth)

    ax.set_xlim(min(xbins), max(xbins)+.05)
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    sns.despine()
    print(len(xbins))
    print(xbins)

    return ax


# 1/2/2026 plot difference for all noise levels on the same plot
from matplotlib import colors as mcolors

def _blend_with_white(color, t):
    # t in [0,1]; 0 = base color, 1 = white
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    t = min(max(t-.3, 0.0), 1.0)
    return tuple(base * (1 - t) + white * t)

def plot_all_state_rep_dem_gap_by_noise(
    distributions_by_noise,
    rep_label="Most Republican",
    dem_label="Most Democratic",
    xbins=None,
    ymin=None,
    ymax=None,
    legend=True,
    xlabel="Districts / Seats",
    ylabel="Gerrymandering range",
    weight_by_state=None,
    do_vertical_integers=False,
    figsize=(8, 6),
    base_color="#0B2A5B",  # dark blue
    do_abs=True,
    legendncol = 4,
):
    if xbins is None:
        xbins = list(np.linspace(1.0 / 53, 1, 100))
        for k in range(2, 54):
            xbins.extend([float(l) / k for l in range(1, k)])
        xbins = list(sorted(set(xbins)))

    fig, ax = plt.subplots(figsize=figsize)

    any_noise = list(distributions_by_noise.keys())[0]
    any_state = list(distributions_by_noise[any_noise].keys())[0]
    row_index = list(distributions_by_noise[any_noise][any_state].index)
    rep_i = row_index.index(rep_label)
    dem_i = row_index.index(dem_label)

    noise_levels = sorted(distributions_by_noise.keys())
    min_noise = min(noise_levels)
    max_noise = max(noise_levels)
    noise_span = np.log(max_noise + .01) - np.log(min_noise + .01) if max_noise != min_noise else 1

    for noise_level in noise_levels:
        distributions = distributions_by_noise[noise_level]
        full_distribution = np.zeros((len(xbins), len(row_index)))
        total_seats = 0

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

        gap = (full_distribution[:, rep_i] - full_distribution[:, dem_i]) / total_seats
        if do_abs:
            gap = np.abs(gap)
        print(f"Noise level {noise_level}, gap min {np.min(gap)}, max {np.max(gap)}")
        # Higher noise -> lighter
        noise_norm = (np.log(noise_level + .01) - np.log(min_noise + .01)) / noise_span
        color = _blend_with_white(base_color, noise_norm)

        ax.plot(
            xbins,
            gap,
            linewidth=2,
            color=color,
            label=str(int(noise_level)),
        )

        if ymin is None:
            ymin = np.min(gap) if ymin is None else min(ymin, np.min(gap))
        if ymax is None:
            ymax = np.max(gap) if ymax is None else max(ymax, np.max(gap))

    print("ymin, ymax: ", ymin, ymax)
    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = 1

    if do_vertical_integers:
        for y in set(range(1, 6)):
            linewidth = 1 if y <= 5 else 0
            ax.vlines(1.0 / y, ymin, ymax, linewidth=linewidth)

    ax.set_xlim(min(xbins), max(xbins))
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    sns.despine()

    #put the legend above the plot, in 4 columns
    if legend:
        ax.legend(frameon=False, title="Noise level", title_fontsize=16, ncol=legendncol, loc='upper center', bbox_to_anchor=(0.5, 1.15))

    return fig




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
