import numpy as np
import pandas as pd
import time
import math
import os
import pickle

import master
import tree_dp
import objectives
import districts
import load
import subsample
import fairness


def optimize_fairness(bdm, cost_coeffs, root_map, k):
    sol_dict = {}
    for partition_ix, leaf_slice in root_map.items():
        if len(leaf_slice) == k:  # Trivial optimization
            sol_dict[partition_ix] = {
                'n_leaves': len(leaf_slice),
                'solution_ixs': root_map[partition_ix],
                'objective_value': cost_coeffs[leaf_slice].sum()
            }
            continue
        bdm_slice = bdm[:, leaf_slice]
        coeff_slice = cost_coeffs[leaf_slice]
        model, dvars = master.make_master_vectorized(k, bdm_slice, coeff_slice)

        model.Params.LogToConsole = 0
        model.Params.MIPGapAbs = 1e-4
        model.Params.TimeLimit = len(cost_coeffs) / 5
        model.optimize()
        opt_cols = [j for j, v in enumerate(dvars.tolist()) if v.X > .5]

        sol_dict[partition_ix] = {
            'n_leaves': len(leaf_slice),
            'solution_ixs': root_map[partition_ix][opt_cols],
            'objective_value': cost_coeffs[leaf_slice][opt_cols].sum()
        }

    return sol_dict


def optimize_unfairness(leaf_nodes, internal_nodes, expected_seats, party):
    # TODO (hwr26): no longer have the duality of two party system
    # will call this function for each voting rule and party pair
    values, solutions = tree_dp.query_per_root_partition(leaf_nodes, internal_nodes, expected_seats)
    #d_values, d_solutions = tree_dp.query_per_root_partition(leaf_nodes, internal_nodes, -expected_seats)

    # d_value is lowest r_value negated
    #d_values = list(internal_nodes[0].n_seats + np.array(d_values))

    return {
        f"{party}_opt_vals": values,
        f"{party}_opt_solutions": solutions
        # 'd_opt_vals': d_values,
        # 'd_opt_solutions': d_solutions,
    }


def enumerate_subsample(leaf_nodes, internal_nodes):
    k = internal_nodes[0].n_districts
    subsample_constant = 1500 * k**.5
    solution_count, parent_nodes = subsample.get_node_info(leaf_nodes, internal_nodes)
    pruned_internal_nodes = subsample.prune_sample_space(internal_nodes,
                                                         solution_count,
                                                         parent_nodes,
                                                         subsample_constant)
    subsampled_partitions = tree_dp.enumerate_partitions(leaf_nodes, pruned_internal_nodes)
    return subsampled_partitions


def calculate_statewide_average_voteshare(state):
    election_df = load.load_election_df(state)
    partisan_totals = election_df.sum(axis=0).to_dict()
    elections = set(e[2:] for e in partisan_totals)
    election_results = {e: partisan_totals['R_' + e] /
                        (partisan_totals['R_' + e] + partisan_totals['D_' + e])
                    for e in elections}
    return np.mean(np.array(list(election_results.values())))


# TODO (hwr26): this is a bit of a cheat to get around introducing the third
# party at the census tract level which would result in re-generating the
# districts which is the most computationally heavy task
# This number (it appears 2 districts is best) does a sufficent job at estimating
# the average voteshare by party
def calculate_statewide_average_voteshare_with_third_party(state, ddf_save_path):
    df = pd.read_csv(os.path.join(ddf_save_path, f"{state}_2_district_df.csv"), index_col=0)
    cols = [f"{p}_mean" for p in ["r", "d", "t"]]
    return np.average(np.array(df[cols]), axis=0, weights=df["population"])


def process_trial(state, k, internal_nodes, leaf_nodes, experiment_dir, district_dfs, score_df,
    fairness_metric, skip_fair):
    ddf_save_path = os.path.join(experiment_dir, district_dfs)
    state_vote_share = calculate_statewide_average_voteshare_with_third_party(state, ddf_save_path)

    bdm = districts.make_bdm(leaf_nodes, len(internal_nodes[0].area))
    seats_array = np.array([d.n_seats for d in sorted(leaf_nodes.values(),
                                                      key=lambda x: x.id)])
    partition_map = master.make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)

    msp_solutions = {}
    tree_solutions = {}

    # TODO (hwr26): explicitly give the voting rules and parties
    voting_rules = ["thiele_pav", "thiele_squared", "thiele_independent"]
    parties = ["r", "d", "t"]

    for voting_rule in voting_rules:
        tree_solutions[voting_rule] = {}
        expected_seats = {}
        for party in parties:
            expected_seats[party] = score_df[f"{party}_{voting_rule}"].values
            solutions = optimize_unfairness(leaf_nodes, internal_nodes, expected_seats[party], party)
            tree_solutions[voting_rule].update(solutions)

        if not skip_fair:
            cost_coeffs = fairness_metric(expected_seats, state_vote_share, seats_array)
            solutions = optimize_fairness(bdm, cost_coeffs, partition_map, k)
            msp_solutions[voting_rule] = solutions

    subsample_plans = enumerate_subsample(leaf_nodes, internal_nodes)

    return msp_solutions, tree_solutions, subsample_plans


def process_experiment(states, experiment_dir, district_dfs, scores, opt_save_name, fairness_metric, skip_fair):
    opt_save_dir = os.path.join(experiment_dir, opt_save_name)
    subsample_save_dir = os.path.join(experiment_dir, 'subsampled_plans')
    os.makedirs(opt_save_dir, exist_ok=True)
    os.makedirs(subsample_save_dir, exist_ok=True)

    for state in states:
        for trial_file in sorted(os.listdir(os.path.join(experiment_dir, "sample_trees", state))):
            if trial_file[-2:] != '.p':
                continue

            trial = trial_file[:-2]
            trial_path = os.path.join(experiment_dir, "sample_trees", trial_file[:2], trial_file)
            ddf_path = os.path.join(experiment_dir, 'district_dfs', f'{trial}_district_df.csv')
            score_path = os.path.join(experiment_dir, scores, f'{trial}_score_df.csv')

            print(f'Starting processing of trial {trial}')
            start_t = time.time()
            tree = pickle.load(open(trial_path, 'rb'))
            # district_df = pd.read_csv(ddf_path, index_col='node_id')
            score_df = pd.read_csv(score_path, index_col='node_id')
            leaf_nodes = tree['leaf_nodes']
            internal_nodes = tree['internal_nodes']
            state, k = trial.split('_')
            k = int(k)
            if experiment_dir.split('/')[-1].startswith('hr4000'):
                k = -1

            results = process_trial(state, k, internal_nodes, leaf_nodes, experiment_dir,
                district_dfs, score_df, fairness_metric, skip_fair)
            msp_solutions, tree_solutions, subsample_plans = results

            optimization_results = {
                'fair': msp_solutions,
                'unfair': tree_solutions
            }

            opt_save_path = os.path.join(opt_save_dir, f'{trial}_opt_results.p')
            subsample_save_path = os.path.join(subsample_save_dir, f'{trial}_plans.p')

            pickle.dump(optimization_results, open(opt_save_path, 'wb'))
            pickle.dump(subsample_plans, open(subsample_save_path, 'wb'))

            run_t = round((time.time() - start_t) / 60, 2)
            print(f'Processed and saved {trial} in {run_t} mins')


# if __name__ == '__main__':
#     experiment_dir = os.path.join('optimization_results', 'third_party')
#     process_experiment(experiment_dir)
