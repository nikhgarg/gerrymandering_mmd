from visualization.settings import *
import pandas as pd
import numpy as np
from optimize.analyze_results import *

intcols = ["state_num", "N_districts", "total_voters", "total_winners"]
floatcols = [
    "cohesion_income_Republican",
    "fraction_voters_Republican",
    "fraction_winners_Republican",
    "cohesion_income_Democrat",
    "cohesion_income",
    "cohesion_partisan_score_Republican",
    "cohesion_partisan_score_Democrat",
    "cohesion_partisan_score",
    "cohesion_education_Republican",
    "cohesion_education_Democrat",
    "cohesion_education",
    "cohesion_racial_Republican",
    "cohesion_racial_Democrat",
    "cohesion_racial",
    "cohesion_geographic_Republican",
    "cohesion_geographic_Democrat",
    "cohesion_geographic",
]

from elections.median_voter import (
    thiele_pav,
    thiele_squared,
    thiele_approvalindependent,
    calculate_from_vote_share,
)

rule_map = {
    "thiele_pav": thiele_pav,
    "thiele_independent": thiele_approvalindependent,
    "thiele_squared": thiele_squared,
    "stv": thiele_pav,
}


def add_single_district_per_state_outputs(df, rule):
    cols = [
        "state",
        "N_districts",
        "total_winners",
        "optimization",
        "N_winners_Republican",
        "district_vote_shares",
        "district_n_winners",
        "fraction_voters_Republican",
        "district_n_winners_Republican",
        "fraction_winners_Republican",
    ]
    d = {col: [] for col in cols}
    for state in state_constants:
        d["state"].append(state)
        d["N_districts"].append(1)
        d["optimization"].append("single_district_for_state")
        d["total_winners"].append(state_constants[state]["seats"])
        rep_winners = calculate_from_vote_share(
            state_constants[state]["vote_share"],
            int(state_constants[state]["seats"]),
            rule_map[rule],
        )
        d["N_winners_Republican"].append(rep_winners)
        d["district_vote_shares"].append([state_constants[state]["vote_share"]])
        d["district_n_winners"].append([state_constants[state]["seats"]])
        d["district_n_winners_Republican"].append([rep_winners])
        d["fraction_voters_Republican"].append(state_constants[state]["vote_share"])
        d["fraction_winners_Republican"].append(
            rep_winners / state_constants[state]["seats"]
        )

    return pd.concat([df, pd.DataFrame(d)])


def load_organized_df(path, template, method, directfromoptimization=False):
    df = pd.read_csv("{}/{}_{}.csv".format(path, template, method))
    if not directfromoptimization:
        df = df.query('total_winners!="total_winners"')
        for col in intcols:
            df.loc[:, col] = df.loc[:, col].astype(int)
        for col in floatcols:
            df.loc[:, col] = df.loc[:, col].astype(float)
        df["N_winners_Republican"] = df.eval(
            "total_winners*fraction_winners_Republican"
        )
    else:
        states = list(state_constants.keys())
        stateconstants_df = pd.DataFrame(
            {
                "state": states,
                "fraction_voters_Republican": [
                    state_constants[state]["vote_share"] for state in states
                ],
            }
        )
        df = df.merge(stateconstants_df, how="left", on="state")
        if "N_winners_Republican" in df.columns:
            df["fraction_winners_Republican"] = df.eval(
                "N_winners_Republican/total_winners"
            )
        df = add_single_district_per_state_outputs(df, method)
    return df


def state_seat_share_distributions_nikhil(
    dfrule,
    col="N_winners_Republican",
    do_most_fair=True,
    divide=True,
    min_name="Most Democratic",
    max_name="Most Republican",
    frac_voters_dictionary=None,
    col_for_minmaxmostfair=None,
):
    if col_for_minmaxmostfair is None:
        col_for_minmaxmostfair = col
    distributions = {}
    print(dfrule.state.unique())
    for state in state_constants:
        n_seats = state_constants[state]["seats"]
        state_distribution = {}
        dfstate = dfrule.query("state==@state")
        if dfstate.shape[0] == 0:
            print("don't have state, skipping: ", state)
            continue
        for k in dfstate.N_districts.unique():
            dfk = dfstate.query("N_districts==@k")
            distributionsampled = dfk.query(
                'optimization=="subsampled" or optimization=="single_district_for_state"'
            )
            argmedianlist = list(distributionsampled[col_for_minmaxmostfair])
            argmedian = np.argsort(argmedianlist)[len(argmedianlist) // 2]

            median = list(distributionsampled[col])[argmedian]
            argminn = dfk[col_for_minmaxmostfair].argmin()
            argmaxx = dfk[col_for_minmaxmostfair].argmax()
            minn = list(dfk[col])[argminn]
            maxx = list(dfk[col])[argmaxx]
            #             print(k, mostfairind, mostfair, n_seats)
            summary_statistics = {max_name: maxx, "Median": median}
            if frac_voters_dictionary is None:
                fraction_voters_Republican = dfk.fraction_voters_Republican
            else:
                fraction_voters_Republican = frac_voters_dictionary[state]["vote_share"]
            if do_most_fair:
                mostfairind = np.argmin(
                    (dfk[col_for_minmaxmostfair] / n_seats - fraction_voters_Republican)
                    .abs()
                    .tolist()
                )
                mostfair = dfk.iloc[mostfairind, :][col]
                summary_statistics.update({"Most Fair in each state": mostfair})
            summary_statistics.update({min_name: minn})
            summary_statistics = pd.Series(summary_statistics)

            if divide:
                state_distribution[k] = summary_statistics / n_seats
            else:
                state_distribution[k] = summary_statistics
        state_distribution_df = pd.DataFrame(state_distribution).sort_index(axis=1)
        distributions[state] = state_distribution_df
    return distributions


def get_prop(df):
    dfff = (
        df.groupby("state")[["total_winners", "fraction_voters_Republican"]]
        .mean()
        .reset_index()
    )
    proportionality = (
        dfff.eval("total_winners*fraction_voters_Republican").sum()
        / dfff["total_winners"].sum()
    )
    return proportionality


def get_uniques_for_setting_columns(df, print_out=True):
    usually_same_columnsloc = set(usually_same_columns).intersection(df.columns)
    usually_different_columns_but_visual_on_maploc = set(
        usually_different_columns_but_visual_on_map
    ).intersection(df.columns)

    sameuniquemorethan1 = df[usually_same_columnsloc].nunique()
    if (sameuniquemorethan1[sameuniquemorethan1 > 1]).shape[0] > 0 and print_out:
        print(
            "WARNING: Make sure purposely want to have different values for the following in the same plot:"
        )
        print(sameuniquemorethan1[sameuniquemorethan1 > 1])
        for col in list(sameuniquemorethan1[sameuniquemorethan1 > 1].index):
            print(col, df[col].unique())
    differentmorethan1 = df[usually_different_columns_but_visual_on_maploc].nunique()
    if (differentmorethan1[differentmorethan1 > 1]).shape[0] > 0 and print_out:
        print(
            "The following are often expected to have multiple values in the same plot"
        )
        print(differentmorethan1[differentmorethan1 > 1])
    return (sameuniquemorethan1[sameuniquemorethan1 > 1]).shape[0]


def explode_optimization_characteristics(df, col="optimization_characteristics"):
    df.loc[:, col] = df.loc[:, col].replace(np.nan, "{}").apply(eval)
    df = pd.concat([df.drop([col], axis=1), df[col].apply(pd.Series)], axis=1)
    return df


def setting_to_query_string(setting):
    for x in setting:
        if type(setting[x]) != list:
            setting[x] = [setting[x]]
    ar = [
        "(" + " or ".join(['({}=="{}")'.format(x, val) for val in setting[x]]) + ")"
        for x in setting
    ]
    return " and ".join(ar)


def setting_to_filtered_df(df, setting):
    return df.query(setting_to_query_string(setting))


def load_outputs(output_files, nrows=None):
    dfs = [pd.read_csv(filee, nrows=nrows) for filee in output_files]
    df = pd.concat(dfs)
    df = explode_optimization_characteristics(df)
    get_uniques_for_setting_columns(df)
    df = df.drop_duplicates(subset=["overall_hash"])
    return df


#     print(df[output_columns].nunique())
