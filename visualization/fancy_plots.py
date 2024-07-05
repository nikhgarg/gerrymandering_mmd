import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from generic.latexify import *

from visualization.helpers import *
from visualization.settings import *


def boxplot_per_district_for_single_state_per_method(
    df,
    state,
    target_col="fraction_winners_Republican",
    do_extremes_and_prop_line=False,
    prop_col="fraction_voters_Republican",
    additional_filters={},
    ax=None,
):
    setting = {"state": state}
    setting.update(additional_filters)
    dfsingle = setting_to_filtered_df(df, setting)
    dfsingle.loc[:, "N_districts"] = dfsingle.loc[:, "N_districts"].astype(int)

    # for voting_method in dfsingle.voting_method.unique():
    # print(voting_method)
    more_filter = {"optimization": ["single_district_for_state", "subsampled"]}  # "voting_method": voting_method,
    dffilt = setting_to_filtered_df(dfsingle, more_filter)
    assert get_uniques_for_setting_columns(dffilt) == 0

    ndist = dffilt.N_districts.max()
    order = list(range(0, ndist + 1))
    # print(order)
    # print(sorted(dffilt.N_districts.unique()))

    sns.boxplot(
        y=target_col,
        x="N_districts",
        data=dffilt,
        fliersize=0,
        whis=1,
        saturation=0.1,
        width=1,
        color=sns.color_palette("Set2")[-1],
        order=order,
        ax=ax,
    )
    ordskip = list(range(1, ndist + 1, max(1, int(ndist / 5))))
    ax.set_xticks(ordskip)
    ax.set_xticklabels(ordskip)

    if do_extremes_and_prop_line:
        filt = {}  # {"voting_method": voting_method}  # , 'optimization':['unfair','fair','single_district_for_state','subsampled']
        # dffilt = setting_to_filtered_df(dfsingle, filt)
        dffilt = dfsingle
        dfextremes = dffilt.groupby("N_districts")[target_col].agg(["max", "min"]).reset_index()
        dfextremes = dfextremes.sort_values(by=["N_districts"])
        dfextremes["order"] = range(dfextremes.shape[0])
        sns.scatterplot(y="max", x="N_districts", data=dfextremes, s=60, color=sns.color_palette("RdYlBu", 10)[0], ax=ax)
        ax.set_xticks(ordskip)
        ax.set_xticklabels(ordskip)
        sns.scatterplot(
            y="min", x="N_districts", data=dfextremes, s=60, color=sns.color_palette("RdYlBu", 10)[-1], ax=ax
        )  # , color = sns.color_palette()[3])
        #             print(dfextremes)
        ax.set_xticks(ordskip)
        ax.set_xticklabels(ordskip)
        if len(dfsingle[prop_col].unique()) != 1:  # each state has a fixed voter set
            print(dfsingle[prop_col].unique())
        #                 assert len(dfsingle[prop_col].unique()) == 1
        prop_line = dfsingle[prop_col].mean()
        plt.hlines(prop_line, xmin=0, xmax=ndist)
    plt.xlabel("Districts", fontsize=20)
    plt.ylabel("")
    sns.despine()

    # plt.show()