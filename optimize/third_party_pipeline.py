import os
import synthetic
import postprocess_columns
import optimize_pipeline
import fairness
import time

STATES = ["AL", "AR", "AZ", "GA", "IL", "IN", "MI", "NC", "OH", "PA", "VA"]
ALPHAS = [0.2, 0.5, 0.8]
FAIRNESS_METRICS = {"MEAN": fairness.third_party_mean_proportionality_gap,
                    "MAX": fairness.third_party_max_proportionality_gap}
# TODO (hwr26): optimize off as working on gurobi on cluster
ADD_THIRD_PARTY = False
SCORING = False
OPTIMIZE = True

start = time.time()
for alpha in ALPHAS:

    print(f"=========================================")
    print(f"          STARTING ALPHA {alpha}         ")
    print(f"=========================================\n")

    if ADD_THIRD_PARTY:
        print("======== ADDING THIRD PARTY =========")
        synthetic.main(states=STATES, alpha=alpha)
        print("=====================================\n")

    if SCORING:
        print("======== SCORING =========")
        postprocess_columns.main(states=STATES,
                                 experiment_name="third_party",
                                 ddf_save_directory=f"synthetic_{alpha}_district_dfs",
                                 scores_save_directory=f"synthetic_{alpha}_scores")
        print("==========================\n")

    if OPTIMIZE:
        for metric_name, metric in FAIRNESS_METRICS.items():

            print(f"======== OPTIMIZING w/ METRIC {metric_name} =========")
            experiment_dir = os.path.join("optimization_results", "third_party")
            optimize_pipeline.process_experiment(states=STATES,
                                                 experiment_dir=experiment_dir,
                                                 district_dfs=f"synthetic_{alpha}_district_dfs",
                                                 scores=f"synthetic_{alpha}_scores",
                                                 opt_save_name=f"opt_results_{alpha}_{metric_name}",
                                                 fairness_metric=metric,
                                                 skip_fair=False)
            print(f"=====================================================\n")

runtime = round((time.time() - start) / 60, 2)
print(f"TOTAL RUNTIME: {runtime} min")
