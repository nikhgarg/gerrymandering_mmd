import os
import time
import numpy as np
import pandas as pd
import pickle

# TODO (hwr26): chosen constants for experiments
THIRD_PARTY_SHARE = 0.2
THIRD_PARTY_STD = 0.02
PARTIES = ["r", "d", "t"]

EXPERIMENT_DIR = os.path.join('optimization_results', 'third_party')
DDF_SAVE_PATH = os.path.join(EXPERIMENT_DIR, 'district_dfs')


def involve_third_party(r_share, alpha, t_share):
    # "stolen" third party votes proportional to alignment and population
    norm = alpha * r_share + (1 - alpha) * (1 - r_share)
    new_r_share = r_share - (t_share * (alpha * r_share / norm))
    new_d_share = (1 - r_share) - (t_share * ((1 - alpha) * (1 - r_share) / norm))

    return new_r_share, new_d_share, t_share


# TODO (hwr26): recurses through the hierarchical sampling so that third party
# share has geographical correlation
def sample_third_party_share(file):
    tree = pickle.load(open(os.path.join(EXPERIMENT_DIR, "sample_trees", file[:2], file), "rb"))
    nodes = tree['internal_nodes']
    nodes.update(tree['leaf_nodes'])
    districts_t_shares = {}

    def sample(node_id, t_share, t_std):
        node = nodes[node_id]
        if not node.children_ids:
            districts_t_shares[node_id] = np.random.normal(t_share, t_std)
        else:
            for child in [c for part in node.children_ids for c in part]:
                sample(child, np.random.normal(t_share, t_std), t_std)

    sample(0, THIRD_PARTY_SHARE, THIRD_PARTY_STD)  # node_id 0 is root
    return [v for _,v in sorted(districts_t_shares.items(), key=lambda x: x[0])]


def main(states, alpha=0.5):
    synthetic_ddf_save_path = os.path.join(EXPERIMENT_DIR, f"synthetic_{alpha}_district_dfs")
    os.makedirs(synthetic_ddf_save_path, exist_ok=True)

    for state in states:
        for file in sorted(os.listdir(os.path.join(EXPERIMENT_DIR, "sample_trees", state))):
            if file[-2:] != '.p':
                continue

            synthetic_start_t = time.time()

            df = pd.read_csv(os.path.join(DDF_SAVE_PATH, f"{file[:-2]}_district_df.csv"))

            # clever trick to back out election columns from DoF
            election_columns = df.columns[-(df.DoF[0] + 5): -4]

            # synthesize third party for each election
            elections = {party : [] for party in PARTIES}
            for col in election_columns:
                r, d, t = involve_third_party(r_share = df[col],
                                            alpha=alpha,
                                            t_share=sample_third_party_share(file))
                elections["r"].append(r)
                elections["d"].append(d)
                elections["t"].append(t)

            # omit all individual election results from resulting district_df
            df = df.drop(columns=election_columns)

            # compute mean and std for each party
            for party in PARTIES:
                df[f"{party}_mean"] = np.mean(elections[party], axis=0)
                df[f"{party}_std"] = np.std(elections[party], axis=0)

            df.to_csv(os.path.join(synthetic_ddf_save_path, f"{file[:-2]}_district_df.csv"))

            synthetic_runtime = round(time.time() - synthetic_start_t, 2)
            print(f"Third party added to {file} in {synthetic_runtime} seconds.")


# if __name__ == '__main__':
#     main()