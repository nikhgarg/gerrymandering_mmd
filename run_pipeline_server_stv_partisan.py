import pickle
import pandas as pd
import numpy as np

import pipeline_multiple_states

custom_params = {
    "VOTING_METHODS": ["stv"],  # "stv",
    "MAP_GENERATOR": "from_optimization_and_sampling_runs_per_state",
}

generator_params = {
    "maps_per_setting_num": 100000,
    "maps_per_setting_num_order": [500, 1000],
    # "states_todo": ["NC", "CA", "MA", "TX"],
    "district_directory": "/home/ng343/gerrymandering_and_social_choice/",
    "minutes_freq_to_save": 30
    # "folder_save": "/home/ng343/cached_values/",
}

data_params = {
    "LOADING_FROM_DATA": True,
    "VOTER_RAW_FILENAME": "data/fullUS_individuals_with_FIP_6M_balanced_allelections.csv",
    "FEATURE_COLS": ["dem_partisan_score"],
    "GROUP_COL": "party",
    "N_VOTERS": 20000,
    "N_VOTERS_STV": 5000,
    "N_CANDIDATES_PER_DISTRICT_GROUP": 1,
    "N_STV_CANDIDATES_MAX": 1000,
    "DISTANCE_FUNCTION": "party_first_then_partisanscore",  #'party_first_then_partisanscore'#
    "CENSUS_TRACT_INFO_FILENAME": "data/state_df_tractinfo_combined.csv",
}
data_params.update(generator_params)
data_params.update(custom_params)
data_params.update({"label": "rebalanced20210530"})

output = pipeline_multiple_states.meta_pipeline_states(
    data_params,
    save_file_template="cached_values/outputs/outputs_partisan_stv_20210811_{}.csv",
)
