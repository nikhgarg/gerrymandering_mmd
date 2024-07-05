from evaluations.helpers import *
import numpy as np
from elections.cohesion_scores import geographic_cohesion
from elections.helpers import geometric_median
from scipy.spatial.distance import euclidean


def winner_location_variance(locations):
    numwinners = len(locations)
    weights = [1 for _ in range(numwinners)]
    median = geometric_median(locations, weights)
    return np.average(
        [euclidean(locations[i], median) for i in range(numwinners)], weights=weights,
    )


def fraction_each_group_and_medians(voters_df, competition_df, params):
    num_each_group_voters = {x: 0 for x in params["GROUPS"]}
    num_each_group_winners = {x: 0 for x in params["GROUPS"]}
    # medians = []
    winner_characteristics = {}
    # print(voters_df.head())
    # print(competition_df.query("N_VOTERS>0").head())

    for en, district_row in competition_df.query("N_VOTERS > 0").iterrows():
        district_num = district_row.district
        census_blocks = district_row.census_block
        voters = voters_df[voters_df.census_block.isin(census_blocks)]
        num_each_group_voters = add_two_dicts(
            num_each_group_voters, voters[params["GROUP_COL"]].value_counts()
        )
        num_each_group_winners = add_two_dicts(
            num_each_group_winners, district_row.winners
        )
        winner_characteristics = add_two_dicts(
            winner_characteristics, district_row.winner_characteristics, default=[]
        )

    total_voters = sum([num_each_group_voters[x] for x in num_each_group_voters])
    total_winners = sum([num_each_group_winners[x] for x in num_each_group_winners])
    met = {
        "fraction_voters_{}".format(x): num_each_group_voters[x] / max(1, total_voters)
        for x in num_each_group_voters
    }
    met.update(
        {
            "fraction_winners_{}".format(x): num_each_group_winners[x]
            / max(1, total_winners)
            for x in num_each_group_winners
        }
    )
    met.update(
        {
            "medians": winner_characteristics["medians"],
            # "locations": winner_characteristics.get("locations", np.nan),
            "total_voters": total_voters,
            "total_winners": total_winners,
        }
    )
    parties = competition_df.cohesion_scores.dropna().iloc[0].keys()
    if "locations" in winner_characteristics:
        met["winner_location_variance"] = winner_location_variance(
            winner_characteristics["locations"]
        )
        for party in parties:
            winnerslocations = [
                winner_characteristics["locations"][en]
                for en in range(len(winner_characteristics["locations"]))
                if winner_characteristics["parties"][en] == party
            ]
            if len(winnerslocations) > 0:
                met[
                    "winner_location_variance_{}".format(party)
                ] = winner_location_variance(winnerslocations)
    else:
        met["winner_location_variance"] = np.nan
        for party in parties:
            met["winner_location_variance_{}".format(party)] = np.nan

    return met, competition_df


def cohesion_scores(voters_df, competition_df, params):
    scorefuns = competition_df.cohesion_scores.dropna()
    if len(scorefuns) > 0:
        scorefuns = scorefuns.iloc[0]
    else:
        return {}, competition_df
    parties = scorefuns.keys()
    cohesion_functions = set(
        [x for party in parties for x in scorefuns[party].keys()]
    )  # scorefuns.keys()

    metrics = {}
    for fun in cohesion_functions:
        lab = "cohesion_" + fun
        partylabs = []
        for party in parties:
            partylab = "{}_{}".format(lab, party)
            partylabs.append(partylab)
            competition_df[partylab] = competition_df.cohesion_scores.apply(
                lambda x: x[party][fun]
                if type(x) == dict and party in x and fun in x[party]
                else []
            )
            # print(competition_df[partylab])
            metrics[partylab] = np.nanmean(
                [x for y in competition_df[partylab] for x in y]
            )
            # print(metrics[partylab])
        metrics[lab] = np.nanmean(
            [x for partylab in partylabs for y in competition_df[partylab] for x in y]
        )

        # print(competition_df[lab])
    return metrics, competition_df


metric_functions = [fraction_each_group_and_medians, cohesion_scores]


def calculate_all_metrics(voters_df, competition_df, params):
    mets = {}
    # create a dictionary of all the metrics that I care about
    for metric in metric_functions:
        metss, competition_df = metric(voters_df, competition_df, params)
        mets.update(metss)
    return mets, competition_df
