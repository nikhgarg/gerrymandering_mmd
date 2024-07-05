import pandas as pd
import numpy as np
from elections.cohesion_scores import *
import math
import heapq
from elections.stv.heap_helpers import *

rounding = 1e-6


def get_votes_and_voters_per_candidate(ballots, weight_per_voter, idx_per_voter):
    vote_count = {}
    voters_per_candidate = {}
    for ind, b in enumerate(ballots):
        # if idx_per_voter[ind]
        cand = b[
            idx_per_voter[ind]
        ]  # ballot for ind voter, candidate at their appropriate index
        vote_count[cand] = vote_count.get(cand, 0) + weight_per_voter[ind]
        voters_per_candidate[cand] = voters_per_candidate.get(cand, []) + [ind]
    return vote_count, voters_per_candidate


def eliminate_candidate_and_transfer_votes(
    ballots,
    weight_per_voter,
    idx_per_voter,
    cand_to_eliminate,
    vote_count,
    voters_per_candidate,
    heap,
):
    updated_candidates = set()
    for voter in voters_per_candidate[
        cand_to_eliminate
    ]:  # transfer votes for candidate  who got elimiated
        assert (
            ballots[voter][idx_per_voter[voter]] == cand_to_eliminate
        )  # that's who they voted for last...
        idx_per_voter[voter] += 1
        while (
            ballots[voter][idx_per_voter[voter]] not in vote_count
        ):  # already eliminated candidates
            idx_per_voter[voter] += 1
        newcand = ballots[voter][idx_per_voter[voter]]
        vote_count[newcand] += weight_per_voter[voter]
        voters_per_candidate[newcand] += [voter]
        updated_candidates.add(newcand)
    for newcand in updated_candidates:
        heap.add_update_candidate(newcand, vote_count[newcand])
    del vote_count[cand_to_eliminate]
    del voters_per_candidate[cand_to_eliminate]
    heap.remove_candidate(cand_to_eliminate)
    return vote_count, voters_per_candidate, idx_per_voter


def find_winner(votes_needed_to_win, heap):  # using the maxheap
    votes, candid = heap.find_best_candidate()
    if votes > votes_needed_to_win - rounding:
        return candid


def redistribute_votes(
    winner,
    vote_count,
    voters_who_selected_winner,
    weight_per_voter,
    votes_needed_to_win,
):
    excess_votes_frac: float = float(vote_count[winner] - votes_needed_to_win) / float(
        vote_count[winner]
    )  # len(voters_who_selected_winner)
    # new weight for each voter is old weight times excess_votes_frac
    summ = 0

    for voter in voters_who_selected_winner:
        weight_per_voter[voter] *= excess_votes_frac
        summ += weight_per_voter[voter]

    return weight_per_voter


def stv(
    ballots,
    all_candidates,
    num_winners,
    voters_df,
    col="dem_partisan_score",
    party_col="party",
):
    n_voters = len(ballots)
    weight_per_voter = np.ones(n_voters)
    idx_per_voter = [0 for _ in range(n_voters)]
    # candidates =
    winners = []
    # eliminated = []
    voters_for_winners = {}
    num_left = num_winners
    vote_count, voters_per_candidate = get_votes_and_voters_per_candidate(
        ballots, weight_per_voter, idx_per_voter
    )

    winners_by_party = {part: 0 for part in all_candidates[party_col].unique()}
    winner_characteristics = {}
    marginal_medians = []
    cohesion_scores = {
        part: {} for part in winners_by_party
    }  # doing cohesion scores by party now
    cohesion_scores_by_winner = {}
    # build up cohesion scores by winner while getting winners, and then when have their parties, I can convert it to what I will return

    total_votes_left = n_voters
    # votes_needed_to_win: float = n_voters / float((num_winners + 1))  # Drop quota
    votes_needed_to_win = math.floor(n_voters / (num_winners + 1)) + 1

    # find either a winner or elimintate a candidate
    # if found winner, then add them to winner, do cohesion scores, and transfer voters excess (only take fraction necessary to elect them from each voter)
    # print(votes_needed_to_win)
    # print(vote_count)

    # What I am doing with this heap stuff
    # I want dictionary access (vote_count) to the votes of each candidate so I can quickly modify during the transfer of votes
    # I want to have quick access to mins and maxes to eliminate candidates or find a winner
    # For maxes, assuming that there are many to eliminate, I can just maintain a "current max"
    heap = max_min_heap(vote_count)

    # print(candidate_vote_heap)
    while num_left < len(vote_count) and num_left > 0:
        # print(len(vote_count), num_left, end=" ")
        winner = find_winner(votes_needed_to_win, heap)
        if winner is not None:
            # print("found winner: ", winner, vote_count[winner])
            winners.append(winner)
            num_left -= 1
            voters_who_selected_winner = voters_per_candidate[winner]
            relevant_voter_weights = [
                weight_per_voter[x] for x in voters_who_selected_winner
            ]  # need to grab for cohesion scores
            if num_left > 0:
                weight_per_voter = redistribute_votes(
                    winner,
                    vote_count,
                    voters_who_selected_winner,
                    weight_per_voter,
                    votes_needed_to_win,
                )
                (
                    vote_count,
                    voters_per_candidate,
                    idx_per_voter,
                ) = eliminate_candidate_and_transfer_votes(
                    ballots,
                    weight_per_voter,
                    idx_per_voter,
                    winner,
                    vote_count,
                    voters_per_candidate,
                    heap,
                )
            else:
                del vote_count[winner]
                del voters_per_candidate[winner]
            cohesion_scores_by_winner[winner] = calculate_all_cohesion_scores(
                voters_df.iloc[voters_who_selected_winner, :], relevant_voter_weights
            )
            # print("votes left: ", sum(vote_count.values()), votes_needed_to_win)
        else:  # need to eliminate someone
            candidate_to_eliminate = (
                heap.find_worst_candidate()
            )  # min(vote_count, key=vote_count.get)
            # print("eliminating: ", candidate_to_eliminate, vote_count[candidate_to_eliminate])
            (
                vote_count,
                voters_per_candidate,
                idx_per_voter,
            ) = eliminate_candidate_and_transfer_votes(
                ballots,
                weight_per_voter,
                idx_per_voter,
                candidate_to_eliminate,
                vote_count,
                voters_per_candidate,
                heap,
            )
        # print(vote_count)

    # everyone left is also a winner
    if num_left > 0:
        for winner in vote_count:
            winners.append(winner)
            voters_who_selected_winner = voters_per_candidate[winner]
            relevant_voter_weights = [
                weight_per_voter[x] for x in voters_who_selected_winner
            ]
            cohesion_scores_by_winner[winner] = calculate_all_cohesion_scores(
                voters_df.iloc[voters_who_selected_winner, :], relevant_voter_weights
            )

    # get winner stats
    winner_df = all_candidates[all_candidates.id.isin(winners)]
    marginal_medians = list(winner_df[col])
    winner_locations = list(zip(winner_df.x, winner_df.y))  # list(winner_df[])
    winner_parties = list(winner_df[party_col])
    for en, winner_row in winner_df.iterrows():
        party = winner_row[party_col]
        winners_by_party[party] += 1
        for coh in cohesion_scores_by_winner[winner_row.id]:
            cohesion_scores[party][coh] = cohesion_scores[party].get(coh, []) + [
                cohesion_scores_by_winner[winner_row.id][coh]
            ]

    # assert len(winners) == num_winners
    winner_characteristics["medians"] = marginal_medians
    winner_characteristics["locations"] = winner_locations
    winner_characteristics["parties"] = winner_parties

    return winner_characteristics, winners, cohesion_scores
