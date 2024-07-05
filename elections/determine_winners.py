import numpy as np
from elections.voting_methods import *
from collections import Counter


def determine_winners(voters_df, competition_df, params, candidates_df=None):
    competition_df["winners"] = np.nan
    competition_df.loc[:, "winners"] = competition_df["winners"].astype(object)
    competition_df["winner_characteristics"] = np.nan
    competition_df.loc[:, "winner_characteristics"] = competition_df[
        "winner_characteristics"
    ].astype(object)
    competition_df["cohesion_scores"] = np.nan
    competition_df.loc[:, "cohesion_scores"] = competition_df["cohesion_scores"].astype(
        object
    )

    for en, district_row in competition_df.iterrows():
        voters = voters_df[voters_df.census_block.isin(district_row.census_block)]
        competition_df.at[en, "N_VOTERS"] = len(voters)
        if competition_df.at[en, "N_VOTERS"] == 0:
            continue
        district_num = district_row.district
        state_num = district_row.state_num
        n_winners = district_row.N_WINNERS_PER_DISTRICT
        if district_row.voting_method == "stv":  # need candidates as argument
            ballots = list(voters["state_{}_ranking".format(state_num)].dropna())
            candidates_local = candidates_df[
                candidates_df.census_block.isin(district_row.census_block)
            ]
            candidlist = list(candidates_local.id)
            ballots = [[x for x in y if x in candidlist] for y in ballots]
            (
                winner_characteristics,
                winners_ids,
                cohesion_scores,
            ) = votingmethod_function_mapper[district_row.voting_method](
                ballots,
                candidates_local,  
                n_winners,
                voters,
                col="dem_partisan_score",
                party_col="party",
            )
            winners = Counter(
                candidates_local[candidates_local.id.isin(winners_ids)].party
            )
        else:
            (
                winner_characteristics,
                winners,
                cohesion_scores,
            ) = votingmethod_function_mapper[district_row.voting_method](
                voters, n_winners, col="dem_partisan_score", party_col="party"
            )
        competition_df.at[en, "winners"] = winners
        competition_df.at[en, "winner_characteristics"] = winner_characteristics
        competition_df.at[en, "cohesion_scores"] = cohesion_scores
    return competition_df