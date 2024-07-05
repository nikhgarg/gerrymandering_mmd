import pandas as pd


def get_winners_from_dictionary(votes_per_candidate, n_winners):
    tup = sorted([(c, votes_per_candidate[c]) for c in votes_per_candidate], key=lambda x: -x[1])
    return list([t[0] for t in tup[0:n_winners]])


# Geometric median
# From https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def geometric_median(X, weights, eps=1e-5, do_approx=True):

    if do_approx:
        # sum_weights = sum(weights)
        avg = np.average(X, weights=weights, axis=0)
        # assert len(avg) ==
        return avg

        pass
    else:  # not implemented with weights yet
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)

            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros / r
                y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

            if euclidean(y, y1) < eps:
                return y1

            y = y1
