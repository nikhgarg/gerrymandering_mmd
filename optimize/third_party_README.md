The `district_dfs` and (currently a subset of) raw trial files (in `raw_trials`)
are available on the Box. Download both sets of files for the (state, k) pairs
of interest placing the `district_dfs` in
`optimization_results/third_party/district_dfs` and the raw trial files in
`optimization_results/`.

Then, run `synthetic.py` to create the `synthetic_district_dfs` directory with
a third party synthetically added into the data. You can then run
`postprocess_columns.py` which is set to skip generating `district_dfs`
(as we use pre-computed ones) and just serves to generate the
`district_scores`. Finally, run the `optimization_pipeline.py` script
to generate the final results under `optimization_results/optimization_results`