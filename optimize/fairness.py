import numpy as np


def third_party_max_proportionality_gap(expected_seats, state_vote_share, seats):
    """Return the max proportionality gap.

    Args:
        expected_seats (Dict): Dictionary from party {"r", "d", "t"} to the \
            expected seat share.
        state_vote_share (np.ndarray): State-wide vote share.
        seats (np.ndarray): Number of seats.
    """
    gaps = np.array([(np.abs(expected_seats['r'] - (state_vote_share[0] * seats))),
                     (np.abs(expected_seats['d'] - (state_vote_share[1] * seats))),
                     (np.abs(expected_seats['t'] - (state_vote_share[2] * seats)))])
    return np.max(gaps, axis=0)


def third_party_mean_proportionality_gap(expected_seats, state_vote_share, seats):
    """Return the mean proportionality gap.

    Args:
        expected_seats (Dict): Dictionary from party {"r", "d", "t"} to the \
            expected seat share.
        state_vote_share (np.ndarray): State-wide vote share.
        seats (np.ndarray): Number of seats.
    """
    sum = np.abs(expected_seats['r'] - (state_vote_share[0] * seats)) + \
          np.abs(expected_seats['d'] - (state_vote_share[1] * seats)) + \
          np.abs(expected_seats['t'] - (state_vote_share[2] * seats))
    return sum / 3
