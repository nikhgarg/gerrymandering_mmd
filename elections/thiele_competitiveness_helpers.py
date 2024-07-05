from elections.median_voter import *
import numpy as np


def get_vote_thresholds_for_n_winners(rule, n_winners, gridnum=1139):
    voteshares = np.linspace(0, 1, gridnum)
    n_reps_list = [
        calculate_from_vote_share(x, int(n_winners), rule) for x in voteshares
    ]
    # print(n_reps_list)
    indices = np.searchsorted(n_reps_list, list(range(int(n_winners))), side="right")
    thresholds = [voteshares[ind] for ind in indices]
    return thresholds
