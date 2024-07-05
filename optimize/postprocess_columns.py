import os
import time
import pickle
import pandas as pd
from districts import *
from objectives import *


def columns_to_district_df(file, experiment_dir, dff_save_path):
    ddf_start_t = time.time()
    state = file[:2]
    tree = pickle.load(open(os.path.join(experiment_dir, file), 'rb'))
    bdm = make_bdm(tree['leaf_nodes'],
                    len(tree['internal_nodes'][0].area))

    ddf = create_district_df(state, bdm)
    node_seats = [n.n_seats for n in sorted(tree['leaf_nodes'].values(),
                 key=lambda x: x.id)]
    ddf['n_seats'] = node_seats
    ddf.index = sorted(tree['leaf_nodes'].keys())
    ddf.index.name = 'node_id'

    ddf.to_csv(os.path.join(dff_save_path, f'{file[:-2]}_district_df.csv'))
    runtime = round(time.time() - ddf_start_t, 2)
    print(f'District df processing for {file} finished in {runtime} seconds')
    return ddf


def score_districts(ddf, file, scores_save_path):
    # Score the districts
    score_start_t = time.time()
    max_N = max(ddf.n_seats)
    create_score_array = lambda f: np.insert(np.cumsum(np.array([f(n) for n in range(max_N)])), 0, 0)
    thiele_pav_score_array = create_score_array(thiele_pav)
    thiele_squared_score_array = create_score_array(thiele_squared)
    thiele_independent_score_array = create_score_array(thiele_approvalindependent)

    # TODO (hwr26): expected seats won by party for third party analysis
    score_dict = {}
    thiele_pav_score = calculate_thiele_n_winners_uncertain_batch_third_party(ddf, thiele_pav_score_array)
    thiele_squared_score = calculate_thiele_n_winners_uncertain_batch_third_party(ddf, thiele_squared_score_array)
    thiele_independent_score = calculate_thiele_n_winners_uncertain_batch_third_party(ddf, thiele_independent_score_array)
    scores = [("thiele_pav", thiele_pav_score),
              ("thiele_squared", thiele_squared_score),
              ("thiele_independent", thiele_independent_score)]
    for name, score in scores:
        score_dict[f"r_{name}"] = score[0]
        score_dict[f"d_{name}"] = score[1]
        score_dict[f"t_{name}"] = score[2]

    score_df = pd.DataFrame(score_dict, index=ddf.index)
    score_df.to_csv(os.path.join(scores_save_path, f'{file[:-2]}_score_df.csv'))

    score_runtime = round(time.time() - score_start_t, 2)
    print(f'Scoring for {file} finished in {score_runtime} seconds')


def save_block_assignments_as_dict(file, save_path, experiment_dir):
    start_t = time.time()
    state = file[:2]
    geoid_map = load_state_df(state)['GEOID'].to_dict()
    tree = pickle.load(open(os.path.join(experiment_dir, file), 'rb'))
    assignement = {
        nid: [geoid_map[block] for block in node.area]
         for nid, node in tree['leaf_nodes'].items()
    }
    file_path = os.path.join(save_path, f'{file[:-2]}_block_assignment.p')
    pickle.dump(assignement, open(file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    runtime = round(time.time() - start_t, 2)
    print(f'Saving assignment for {file} finished in {runtime} seconds')


def main(states, experiment_name, ddf_save_directory, scores_save_directory):
    # TODO (hwr26): For initial third-party analysis, CREATE_DDFS to False as
    # we will use pre-generated districts and SAVE_BLOCK_ASSIGNMENTS to False
    # as that is not needed for analysis and requires the state_df files.
    CREATE_DDFS = False
    RUN_SCORING = True
    SAVE_BLOCK_ASSIGNMENTS = False

    EXPERIMENT_DIR = os.path.join('optimization_results', experiment_name)
    DDF_SAVE_PATH = os.path.join(EXPERIMENT_DIR, ddf_save_directory)
    SCORES_SAVE_PATH = os.path.join(EXPERIMENT_DIR, scores_save_directory)
    ASSIGNMENT_PATH = os.path.join(EXPERIMENT_DIR, 'block_assignments')

    os.makedirs(DDF_SAVE_PATH, exist_ok=True)
    os.makedirs(SCORES_SAVE_PATH, exist_ok=True)
    os.makedirs(ASSIGNMENT_PATH, exist_ok=True)


    for state in states:
        for file in sorted(os.listdir(os.path.join(EXPERIMENT_DIR, "sample_trees", state))):
            if file[-2:] != '.p':
                continue
            # Process the generated columns by aggregating statistics to district level.
            if CREATE_DDFS:
                ddf = columns_to_district_df(file, EXPERIMENT_DIR, DDF_SAVE_PATH)
            else:
                ddf = pd.read_csv(os.path.join(DDF_SAVE_PATH, f'{file[:-2]}_district_df.csv'),
                                    index_col='node_id')

            if RUN_SCORING:
                score_districts(ddf, file, SCORES_SAVE_PATH)

            if SAVE_BLOCK_ASSIGNMENTS:
                save_block_assignments_as_dict(file, ASSIGNMENT_PATH, EXPERIMENT_DIR)


# if __name__ == '__main__':
#     main("experiment", "district_dfs")
