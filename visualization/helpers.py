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
    print(df.shape)
    print(df.columns)
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

# 1/3/2026 version that also returns the hashes composed of the argmin/etc so that we can track which maps were used and plot the same maps for the different noise levels
def state_seat_share_distributions_nikhil_with_hashes(
    dfrule,
    col="N_winners_Republican",
    do_most_fair=True,
    divide=True,
    min_name="Most Democratic",
    max_name="Most Republican",
    frac_voters_dictionary=None,
    col_for_minmaxmostfair=None,
    hash_col="overall_hash",
    percentile=75,
):
    if col_for_minmaxmostfair is None:
        col_for_minmaxmostfair = col

    distributions = {}
    distributions_indices = {}
    print(dfrule.state.unique())

    for state in state_constants:
        n_seats = state_constants[state]["seats"]
        state_distribution = {}
        state_distribution_indices = {}

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

            median_val = list(distributionsampled[col])[argmedian]
            median_key = list(distributionsampled[col_for_minmaxmostfair])[argmedian]
            median_hashes = list(
                distributionsampled.loc[
                    distributionsampled[col_for_minmaxmostfair] == median_key, hash_col
                ]
            )

            dfk_dem = dfk.query('party_optimized_for=="d_opt_solutions"')
            dfk_rep = dfk.query('party_optimized_for=="r_opt_solutions"')
            dfk_single = dfk.query('optimization=="single_district_for_state"')

            if dfk_dem.shape[0] == 0:
                dfk_dem = dfk_single
            if dfk_rep.shape[0] == 0:
                dfk_rep = dfk_single

            if dfk_dem.shape[0] == 0:
                min_key = np.nan
                minn = np.nan
                min_hashes = []
            else:
                min_key = np.percentile(dfk_dem[col_for_minmaxmostfair], 100 - percentile)
                dem_slice = dfk_dem[dfk_dem[col_for_minmaxmostfair] <= min_key]
                minn = dem_slice[col].mean()
                min_hashes = list(dem_slice[hash_col])

            if dfk_rep.shape[0] == 0:
                max_key = np.nan
                maxx = np.nan
                max_hashes = []
            else:
                max_key = np.percentile(dfk_rep[col_for_minmaxmostfair], percentile)
                rep_slice = dfk_rep[dfk_rep[col_for_minmaxmostfair] >= max_key]
                maxx = rep_slice[col].mean()
                max_hashes = list(rep_slice[hash_col])

            summary_statistics = {max_name: maxx, "Median": median_val}
            summary_hashes = {max_name: max_hashes, "Median": median_hashes}

            if frac_voters_dictionary is None:
                fraction_voters_Republican = dfk.fraction_voters_Republican
            else:
                fraction_voters_Republican = frac_voters_dictionary[state]["vote_share"]

            if do_most_fair:
                diffs = (
                    dfk[col_for_minmaxmostfair] / n_seats - fraction_voters_Republican
                ).abs()
                min_diff = diffs.min()
                mostfairind = diffs.argmin()
                mostfair = dfk.iloc[mostfairind, :][col]
                mostfair_hashes = list(dfk.loc[diffs == min_diff, hash_col])

                summary_statistics.update({"Most Fair in each state": mostfair})
                summary_hashes.update({"Most Fair in each state": mostfair_hashes})

            summary_statistics.update({min_name: minn})
            summary_hashes.update({min_name: min_hashes})

            summary_statistics = pd.Series(summary_statistics)

            if divide:
                state_distribution[k] = summary_statistics / n_seats
            else:
                state_distribution[k] = summary_statistics

            state_distribution_indices[k] = pd.Series(summary_hashes)

        distributions[state] = pd.DataFrame(state_distribution).sort_index(axis=1)
        distributions_indices[state] = pd.DataFrame(state_distribution_indices).sort_index(axis=1)

    return distributions, distributions_indices


# def state_seat_share_distributions_nikhil_with_hashes(
#     dfrule,
#     col="N_winners_Republican",
#     do_most_fair=True,
#     divide=True,
#     min_name="Most Democratic",
#     max_name="Most Republican",
#     frac_voters_dictionary=None,
#     col_for_minmaxmostfair=None,
#     hash_col="overall_hash",
# ):
#     if col_for_minmaxmostfair is None:
#         col_for_minmaxmostfair = col

#     distributions = {}
#     distributions_indices = {}
#     print(dfrule.state.unique())

#     for state in state_constants:
#         n_seats = state_constants[state]["seats"]
#         state_distribution = {}
#         state_distribution_indices = {}

#         dfstate = dfrule.query("state==@state")
#         if dfstate.shape[0] == 0:
#             print("don't have state, skipping: ", state)
#             continue

#         for k in dfstate.N_districts.unique():
#             dfk = dfstate.query("N_districts==@k")
#             distributionsampled = dfk.query(
#                 'optimization=="subsampled" or optimization=="single_district_for_state"'
#             )

#             argmedianlist = list(distributionsampled[col_for_minmaxmostfair])
#             argmedian = np.argsort(argmedianlist)[len(argmedianlist) // 2]

#             median_val = list(distributionsampled[col])[argmedian]
#             median_key = list(distributionsampled[col_for_minmaxmostfair])[argmedian]
#             median_hashes = list(
#                 distributionsampled.loc[
#                     distributionsampled[col_for_minmaxmostfair] == median_key, hash_col
#                 ]
#             )

#             min_key = dfk[col_for_minmaxmostfair].min()
#             max_key = dfk[col_for_minmaxmostfair].max()

#             argminn = dfk[col_for_minmaxmostfair].argmin()
#             argmaxx = dfk[col_for_minmaxmostfair].argmax()

#             minn = list(dfk[col])[argminn]
#             maxx = list(dfk[col])[argmaxx]

#             min_hashes = list(dfk.loc[dfk[col_for_minmaxmostfair] == min_key, hash_col])
#             max_hashes = list(dfk.loc[dfk[col_for_minmaxmostfair] == max_key, hash_col])

#             summary_statistics = {max_name: maxx, "Median": median_val}
#             summary_hashes = {max_name: max_hashes, "Median": median_hashes}

#             if frac_voters_dictionary is None:
#                 fraction_voters_Republican = dfk.fraction_voters_Republican
#             else:
#                 fraction_voters_Republican = frac_voters_dictionary[state]["vote_share"]

#             if do_most_fair:
#                 diffs = (dfk[col_for_minmaxmostfair] / n_seats - fraction_voters_Republican).abs()
#                 min_diff = diffs.min()
#                 mostfairind = diffs.argmin()
#                 mostfair = dfk.iloc[mostfairind, :][col]
#                 mostfair_hashes = list(dfk.loc[diffs == min_diff, hash_col])

#                 summary_statistics.update({"Most Fair in each state": mostfair})
#                 summary_hashes.update({"Most Fair in each state": mostfair_hashes})

#             summary_statistics.update({min_name: minn})
#             summary_hashes.update({min_name: min_hashes})

#             summary_statistics = pd.Series(summary_statistics)

#             if divide:
#                 state_distribution[k] = summary_statistics / n_seats
#             else:
#                 state_distribution[k] = summary_statistics

#             state_distribution_indices[k] = pd.Series(summary_hashes)

#         distributions[state] = pd.DataFrame(state_distribution).sort_index(axis=1)
#         distributions_indices[state] = pd.DataFrame(state_distribution_indices).sort_index(axis=1)

#     return distributions, distributions_indices

# 1/3/2026 version that then takes the hashes from above and gets the corresponding rows from a dataframe
def distributions_from_hash_means(
    dfrule,
    distributions_indices,
    col="N_winners_Republican",
    hash_col="overall_hash",
    divide=True,
):
    # Average duplicates per hash first
    hash_to_value = dfrule.groupby(hash_col)[col].mean()

    distributions = {}
    for state, state_idx_df in distributions_indices.items():
        n_seats = state_constants[state]["seats"]
        state_dist = {}

        for k in state_idx_df.columns:
            series = {}
            for stat_name, hash_list in state_idx_df[k].items():
                if hash_list is None or len(hash_list) == 0:
                    series[stat_name] = np.nan
                    continue

                values = hash_to_value.reindex(hash_list)
                series[stat_name] = values.mean()

            series = pd.Series(series)
            if divide:
                series = series / n_seats

            state_dist[k] = series

        distributions[state] = pd.DataFrame(state_dist).sort_index(axis=1)

    return distributions



#9/17/2025 version of state_seat_share_distributions so that I am taking the mean of the winners for the 
# most unfair ones, instead of the max/min
# this is because the max/min is very noisy when there are few samples, and so extremes are more intense
# and the purpose of the plots are to show the effectiveness of gerrymandering for min/min
def state_seat_share_distributions_nikhil_takemeansbyoptimization(
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


            if k > 1:
                distribution_forRepublicans = dfk.query(
                    'optimization=="unfair" and party_optimized_for=="r_opt_solutions"'
                )
                # print(state, k)
                # print(distribution_forRepublicans)
                
                argRepublicanlist = list(distribution_forRepublicans[col_for_minmaxmostfair])
                argRepublicanmedian = np.argsort(argRepublicanlist)[len(argRepublicanlist) // 2]
                republicanMedian = list(distribution_forRepublicans[col])[argRepublicanmedian]
                republicanMean = distribution_forRepublicans[col].mean()
                
                percentile = 70
                republicanPercentile = np.percentile(distribution_forRepublicans[col], percentile)
                
                distribution_forDemocrats = dfk.query(
                    'optimization=="unfair" and party_optimized_for=="d_opt_solutions"'
                )
                argDemocratlist = list(distribution_forDemocrats[col_for_minmaxmostfair])
                argDemocratmedian = np.argsort(argDemocratlist)[len(argDemocratlist) // 2]
                democratMedian = list(distribution_forDemocrats[col])[argDemocratmedian]
                democratMean = distribution_forDemocrats[col].mean()
                democratPercentile = np.percentile(distribution_forDemocrats[col], 100-percentile)
                
                # maxx = republicanMedian
                # minn = democratMedian
                maxx = republicanMean
                minn = democratMean
                
                # maxx = republicanPercentile
                # minn = democratPercentile
            else:
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
                # same thing here, take mean of most fair optimizations
                distribution_forMostFair = dfk.query(
                    'optimization=="fair"'
                )
                # argmostfairlist = list(distribution_forMostFair[col_for_minmaxmostfair])
                # argmostfairmedian = np.argsort(argmostfairlist)[len(argmostfairlist) // 2]
                # mean while allowing for nans
                
                mostfairMean = distribution_forMostFair[col_for_minmaxmostfair].mean(skipna = True)
                mostfair = mostfairMean
                
                #if distribution_forMostFair is empty since didn't do the optimization, then resort to argmin
                if distribution_forMostFair.shape[0] == 0:
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
    usually_same_columnsloc = list(set(usually_same_columns).intersection(df.columns))
    usually_different_columns_but_visual_on_maploc = list(set(
        usually_different_columns_but_visual_on_map
    ).intersection(df.columns))

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
