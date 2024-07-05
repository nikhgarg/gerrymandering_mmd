# Optimize
This directory contains all code for generating districts using stochastic hierarchical partitioning (SHP), evaluting, and then optimizing over these districts for multiple voter rules.

Gurobi is a required package to generate or optimize districts.

## Reproducing results
```
# Run generation
# This takes about a month to run in series. Recommended to run in parellel
# by running multible jobs with different trial_ix parameter.
python generate_multimember.py

# Create district dfs and score districts using different voting rules.
python postprocess_columns.py

# Run MSP, tree dynamic programming, and subsampling.
python optimize_pipeline.py
```

## Analyzing results
Full experiment results can be downloaded from our [box](https://cornell.app.box.com/folder/137242219019?s=xrzcblq4wg7bpe51q26cgpklwjuwz3ph) repository.

### Result contents
The `district_dfs` directory contains a csv for each trial where a trial is a (state, k) pair encoded in the file name. Each row of a district_df corresponds to a distinct district where the columns give aggregated census metrics for every district. The `node_id` column is the unique index which can be used to join this information to other result files.

The `district_scores` directory is much the same with a csv per trial where each row corresponds to a district. Each column corresponds to the number of seats Republicans win in expectation for a particular voting rule. `node_id` is the primary key.

The `optimization_results` directory contains a pickle file per trial each containing a dictionary with both the fairness and unfairness optimization results per voting rule. Specifically, let `results = pickle.load(open('STATE_SEATS.p', 'rb'))`. `results` has two top level keys: "fairness" and "unfairness". Each of `results['fairness']` and `results['unfairness']` has a key per voting rule. `results['fairness']['voting_rule']` is a dictionary with key equal to the root partition index `i`, and value as a dictionary with key "solution_ixs" and "objective_value" for the optimal solution and objective to the MSP using nodes from root partition `i`. `results['unfairness']['voting_rule']` is a dictionary with keys "r_opt_values", "r_opt_solutions", "d_opt_values", and "d_opt_solutions". Each is a list of the plan/objective value which maximizes the expected seats share for Democrats (d_\*) or Republicans (r_\*), with one element per root partition. 


The `subsample_results` directory contains a pickle file per trial each containing a list of subset of plans from the full sample tree enumeration. The exact structure is a list of lists, where each element in the inner list is a `node_id` index and each element in the outer list is a legal plan. Note, plans are not independent, so to take a subsample of the subsample, first randomize the order and then take the first `n` maps.

Finally the `block_assignment` directory contains a pickle file per trial each containing the full mapping of districts to blocks. Each file contains a dictionary with key equal to the node_id of the leaf node, and with value equal to a list of census tract GEOIDs. Note, these are encoded as `int`s not `str`s so in states with a single digit state fips code, it is a 10 digit number not the usual 11 digits.
