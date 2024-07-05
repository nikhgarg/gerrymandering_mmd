from functools import lru_cache
import numpy as np
from scipy.stats import t

### Thiele voting rules ###
@lru_cache(1028)
def thiele_pav(i):
    return 1.0 / (i + 1)


@lru_cache(1028)
def thiele_approvalindependent(i):
    return 1.0


@lru_cache(1028)
def thiele_squared(i):
    return 1.0 / ((i + 1) ** 2)


@lru_cache(1028)
def calculate_objective(r, k, N, f):
    obj = r* sum([f(i) for i  in range(0, k)]) + (1-r)*sum([f(i) for i  in range(0, N-k)])
    return obj


@lru_cache(1028)
def calculate_n_winners(rep_share, N, f_thiele):
    vals = np.zeros(N+1)
    for k in range(N+1):
        vals[k] = calculate_objective(rep_share, k, N, f_thiele)
    return np.argmax(vals)


def calculate_n_winners_vectorized(rep_share, N, score_array):
    scores = rep_share * score_array[:N+1] + (1-rep_share) * score_array[:N+1][::-1]
    return np.argmax(scores)


def calculate_n_winners_uncertain(rep_share, std, DoF, N, score_array):
    sample_values = np.arange(-2, 2.25, 0.25)
    rep_share = rep_share +  sample_values * std
    scores = np.outer(rep_share, score_array[:N+1]) + np.outer((1-rep_share), score_array[:N+1][::-1])
    return np.average(np.argmax(scores, axis=1), weights=t.pdf(sample_values, df=DoF))

def calculate_thiele_n_winners_uncertain_batch(ddf, score_array):
    sample_values = np.arange(-2, 2.25, 0.25)
    weights = t.pdf(sample_values, df=round(ddf.DoF.values.mean()))

    def calc_n_winners(r_share, std, N, score_array):
        rep_share = r_share + sample_values * std
        scores = np.outer(rep_share, score_array[:N+1]) + \
                 np.outer((1-rep_share), score_array[:N+1][::-1])
        return np.average(np.argmax(scores, axis=1), weights=weights)

    n_winners = [calc_n_winners(r, s, n, score_array) for
                 r, s, n in zip(ddf['mean'], ddf.std_dev, ddf.n_seats)]

    return n_winners

def calculate_thiele_n_winners_uncertain_batch_third_party(ddf, score_array):
    # TODO (hwr26): previously was using numerical integration but found
    # that random sampling was more effective after *very* small experiments.
    # sample_values = np.arange(-2, 2.5, 0.25)
    # w = t.pdf(sample_values, df=round(ddf.DoF.values.mean()))
    # weights = np.outer(w, w).flatten()

    def calc_n_winners(r_share, r_std, d_share, d_std, t_share, t_std, N, score_array):
        # TODO (hwr26): n chosen to balance correctness and runtime considerations.
        # TODO (hwr26): chose not to use numpy.random.multivariate_normal with
        # nonzero off-diagonal elements in cov due to runtime considerations.
        n = 50
        x = np.array([np.random.normal(r_share, r_std, n),
                      np.random.normal(d_share, d_std, n),
                      np.random.normal(t_share, t_std, n)]).T
        party_shares = x / x.sum(axis=1,keepdims=1)

        seat_combos = np.array([[N-i, i-j, j] for i in range(N+1) for j in range(i+1)])
        score_matrix = np.apply_along_axis(lambda x: score_array[x], 0, seat_combos)
        win_indices = np.argmax(party_shares @ score_matrix.T, axis=1)
        win_combos = np.array([seat_combos[i] for i in win_indices])
        return list(np.average(win_combos, axis=0))

    n_winners = np.array(
                 [calc_n_winners(r_m, s_s, d_m, d_s, t_m, t_s, n, score_array) for
                 r_m, s_s, d_m, d_s, t_m, t_s, n in
                 zip(ddf.r_mean, ddf.r_std,
                     ddf.d_mean, ddf.d_std,
                     ddf.t_mean, ddf.t_std,
                     ddf.n_seats)]
                 ).T

    return n_winners
