from elections.helpers import *

from elections.median_voter import *

from elections.stv.helpers import stv


def approval(votes, candidates, num_winners=1, k=None):
    if k is None:
        k = num_winners
    votes_per_candidate = {c: 0 for c in candidates}
    for vote in votes:
        for c in vote[0:k]:
            votes_per_candidate[c] += 1
    return get_winners_from_dictionary(votes_per_candidate, num_winners)

votingmethod_function_mapper = {x.__name__: x for x in [approval, stv]}
votingmethod_function_mapper.update(thiele_voting_methods)
