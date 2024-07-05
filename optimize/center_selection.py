import math
import random
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

def iterative_random(region_df, n_centers, pdists):
    blocks = list(region_df.index)
    centers = [np.random.choice(blocks)]
    while len(centers) < n_centers:
        weights = np.prod(pdists[np.ix_(centers, region_df.index)],
                          axis=0)
        centers.append(np.random.choice(blocks, p=weights / weights.sum()))
    return centers


def get_capacities(centers, child_sizes, region_df, config):
    """
    Implements capacity assigment methods (both computing and matching)

    Args:
        centers: (list) of block indices of the centers
        child_sizes: (list) of integers of the child node capacities
        region_df: (pd.DataFrame) state_df subset of the node region
        config: (dict) ColumnGenerator configuration

    Returns: (dict) {block index of center: capacity}

    """
    n_children = len(child_sizes)
    total_districts = int(sum([s for s, _ in child_sizes]))

    center_locs = region_df.loc[centers][['x', 'y']].values
    locs = region_df[['x', 'y']].values
    pop = region_df['population'].values

    dist_mat = cdist(locs, center_locs)
    if config['capacity_weights'] == 'fractional':
        dist_mat **= -2
        weights = dist_mat / np.sum(dist_mat, axis=1)[:, None]
    elif config['capacity_weights'] == 'voronoi':
        assignment = np.argmin(dist_mat, axis=1)
        weights = np.zeros((len(locs), len(centers)))
        weights[np.arange(len(assignment)), assignment] = 1
    else:
        raise ValueError('Invalid capacity weight method')

    center_assignment_score = np.sum(weights * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()

    center_order = center_assignment_score.argsort()
    capacities_order = [ix for ix, _ in sorted(enumerate(child_sizes), key=lambda x: x[1][1])]

    return {centers[cen_ix]: child_sizes[cap_ix] for cen_ix, cap_ix
            in zip(center_order, capacities_order)}



