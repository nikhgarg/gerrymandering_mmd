import numpy as np
import networkx as nx
import itertools
from load import *


def make_bdm(leaf_nodes, n_blocks=None):
    """
    Generate the block district matrix given by a sample trees leaf nodes.
    Args:
        leaf_nodes: SHPNode list, output of the generation routine
        n_blocks: (int) number of blocks in the state

    Returns: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.

    """
    districts = [d.area for d in sorted(leaf_nodes.values(),
                 key=lambda x: x.id)]
    if n_blocks is None:
        n_blocks = max([max(d) for d in districts]) + 1
    block_district_matrix = np.zeros((n_blocks, len(districts)))
    for ix, d in enumerate(districts):
        block_district_matrix[d, ix] = 1
    return block_district_matrix


def vectorized_edge_cuts(bdm, G):
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G)
    degree_vector = adjacency_matrix.sum(axis=1).flatten()
    all_edges = degree_vector @ bdm
    district_edges = ((adjacency_matrix @ bdm) * bdm).sum(axis=0)
    return np.asarray(all_edges - district_edges)[0]


def vectorized_dispersion(bdm, state_df):
    x_locs = state_df['x'].values
    y_locs = state_df['y'].values
    population = state_df.population.values

    district_pop = bdm.T @ population
    bdm_p = bdm.T * population

    district_centroid_x = bdm_p @ x_locs / district_pop
    district_centroid_y = bdm_p @ y_locs / district_pop

    centroid_distance_matrix = np.sqrt((((bdm.T * x_locs).T - district_centroid_x)**2 +
                                        ((bdm.T * y_locs).T - district_centroid_y)**2) * bdm)
    return (centroid_distance_matrix.T @ population) / district_pop / 1000


def aggregate_district_election_results(bdm, election_df):
    election_columns = {e: ix for ix, e in enumerate(election_df.columns)}
    elections = list(set([e[2:] for e in election_columns]))
    election_ixs = {e: {'D': election_columns['D_' + e], 'R': election_columns['R_' + e]}
                    for e in elections}

    election_vote_totals = bdm.T @ election_df.values

    result_df = pd.DataFrame({
        election: election_vote_totals[:, column_ixs['R']] /
                  (election_vote_totals[:, column_ixs['R']] + 
                  election_vote_totals[:, column_ixs['D']])
        for election, column_ixs in election_ixs.items()
    })
    return result_df


def aggregate_sum_metrics(bdm, metric_df):
    return pd.DataFrame(
        bdm.T @ metric_df.values,
        columns=metric_df.columns
    )


def aggregate_average_metrics(bdm, metric_df, weights):
    district_weight_normalizer = bdm.T @ weights
    return pd.DataFrame(
        (((bdm.T * weights) @ metric_df.values).T / district_weight_normalizer).T,
        columns=metric_df.columns
    )


def create_district_df(state, bdm):
    election_df = load_election_df(state)
    state_df, G, _, _ = load_opt_data(state)

    sum_metrics = state_df[['area', 'population']]
    average_metrics = state_df.drop(columns=['area', 'population', 'GEOID'])

    sum_metric_df = aggregate_sum_metrics(bdm, sum_metrics)
    average_metric_df = aggregate_average_metrics(bdm, average_metrics,
                                                  state_df.population.values)
    vote_share_df = aggregate_district_election_results(bdm, election_df)

    compactness_df = pd.DataFrame({
        'edge_cuts': vectorized_edge_cuts(bdm, G),
        'dispersion': vectorized_dispersion(bdm, state_df)
    })

    vote_share_distribution_df = pd.DataFrame({
        'mean': vote_share_df.mean(axis=1),
        'std_dev': vote_share_df.std(ddof=1, axis=1),
        'DoF': len(vote_share_df.columns) - 1
    })

    return pd.concat([
        sum_metric_df,
        average_metric_df,
        compactness_df,
        vote_share_df,
        vote_share_distribution_df
    ], axis=1)
