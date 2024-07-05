import numpy as np
import pandas as pd
from functools import lru_cache, partial
from elections.cohesion_scores import *


def get_median_partian_score_single(
    voters, col="dem_partisan_score", party_col="party"
):
    partywinner = voters[party_col].value_counts().reset_index()["index"][0]
    winnersbyparty = {partywinner: 1}
    winnersbyparty.update(
        {part: 0 for part in voters[party_col].unique() if part != partywinner}
    )
    return np.median(list(voters[col])), winnersbyparty


# @lru_cache
def thiele_pav(i):
    return 1.0 / (i + 1)


# @lru_cache
def thiele_approvalindependent(i):
    return 1.0


# @lru_cache
def thiele_squared(i):
    return 1.0 / ((i + 1) ** 2)


thiele_function_mapper = {
    x.__name__: x for x in [thiele_pav, thiele_approvalindependent, thiele_squared]
}


# @lru_cache
def calculate_objective(r, k, N, f):
    return r * sum([f(i) for i in range(0, k)]) + (1 - r) * sum(
        [f(i) for i in range(0, N - k)]
    )


def calculate_from_vote_share(rep_share, num_winners, f_thiele):
    vals = np.zeros(num_winners + 1)
    for k in range(num_winners + 1):
        vals[k] = calculate_objective(rep_share, k, num_winners, f_thiele)
    return np.argmax(vals)


def calculate_from_vote_share_stv(rep_share, num_winners):
    droop = 1.0 / (num_winners + 1) + 1e-10
    ans = int(np.floor(rep_share / droop))
    otherpart = int(np.floor((1 - rep_share) / droop))
    # print(rep_share, ans, otherpart, num_winners)
    assert ans + otherpart == num_winners
    return ans


def get_thiele_median_scores_2parties(
    voters,
    num_winners,
    thiele_fun=thiele_pav,
    col="dem_partisan_score",
    party_col="party",
):
    parties = list(voters[party_col].unique())
    assert len(parties) == 2  # this vote share method only works for 2 parties

    # calculate number_winners from just party vote share -- very fast
    num_voters, _ = voters.shape
    voters_by_party = {
        part: voters.query("{}==@part".format(party_col)) for part in parties
    }
    votes_by_party = {part: voters_by_party[part].shape[0] for part in parties}
    assert sum(votes_by_party.values()) == num_voters
    party0 = parties[0]
    voteshareparty0 = votes_by_party[party0] / num_voters

    winners_by_party = {}
    winners_by_party[party0] = calculate_from_vote_share(
        voteshareparty0, num_winners, thiele_fun
    )
    winners_by_party[parties[1]] = num_winners - winners_by_party[party0]

    # calculate cohesion scores within each party, and then just make as many copies as there are winners for that party
    # cohesion_scores = {}
    cohesion_scores = {
        part: {} for part in parties
    }  # doing cohesion scores by party now
    for party in parties:
        if winners_by_party[party] > 0:
            partyvoters = voters_by_party[party]
            newscores = calculate_all_cohesion_scores(
                partyvoters, np.ones(partyvoters.shape[0])
            )  # hard coded because everyone in same party always has the same score in current use.
            for coh in newscores:
                cohesion_scores[party][coh] = (
                    cohesion_scores[party].get(coh, [])
                    + [newscores[coh]] * winners_by_party[party]
                )
                
    # print(cohesion_scores)
    # Calculate marginal medians for each winner
    # Do as follows: calculate winners sequentially for 1 ... num_winners-1. For each, that gives me the weighage per party using the theile functions with the appropriate number of winners for the next winner. Calculate the median on this weighted set as before.
    # does assume that all the partisan scores are ordered by party (nonoverapping)
    # print("calculated all scores", cohesion_scores)
    partisan_scores_sorted_by_party = {
        party: sorted(list(voters_by_party[party][col])) for party in parties
    }  # want sorted least extreme to most extreme
    reverseparty0order = (
        partisan_scores_sorted_by_party[parties[0]][0]
        < partisan_scores_sorted_by_party[parties[1]][0]
    )
    if reverseparty0order:
        partisan_scores_sorted_by_party[parties[0]] = partisan_scores_sorted_by_party[
            parties[0]
        ][::-1]
    else:
        partisan_scores_sorted_by_party[parties[1]] = partisan_scores_sorted_by_party[
            parties[1]
        ][::-1]

    # print(partisan_scores_sorted_by_party)
    cur_winners_party0 = 0
    winner_characteristics = {}
    marginal_medians = []

    for i in range(1, num_winners + 1):
        party0score = thiele_fun(cur_winners_party0)
        party1score = thiele_fun(i - 1 - cur_winners_party0)

        party0sum = party0score * votes_by_party[parties[0]]
        party1sum = party1score * votes_by_party[parties[1]]

        # print(party0score, party1score)
        # clever trick to get the median here -- see how much weight is needed from the winning party, divide by the weight per voter, and then select that least extreme voter
        weight_left = abs(party0sum - party1sum)
        # print(party0sum, party1sum, weight_left)
        if party0sum < party1sum:
            median = partisan_scores_sorted_by_party[parties[1]][
                int((party1sum - party0sum) / 2.0 / party1score)
            ]
        else:
            median = partisan_scores_sorted_by_party[parties[0]][
                int((party0sum - party1sum) / 2.0 / party0score)
            ]
        marginal_medians.append(median)

        cur_winners_party0 = calculate_from_vote_share(voteshareparty0, i, thiele_fun)
    winner_characteristics["medians"] = marginal_medians
    return winner_characteristics, winners_by_party, cohesion_scores

thiele_voting_methods = {
    x: partial(get_thiele_median_scores_2parties, thiele_fun=thiele_function_mapper[x])
    for x in thiele_function_mapper
}
