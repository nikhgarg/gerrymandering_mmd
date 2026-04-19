import pickle
import pandas as pd
import numpy as np

import pipeline_multiple_states

from helpers import process_csv_files_to_combined_into_organized_df

custom_params = {
    "VOTING_METHODS": ["stv"],  # "stv",
    "MAP_GENERATOR": "from_optimization_and_sampling_runs_per_state",
}

generator_params = {
    "maps_per_setting_num": 100,
    "maps_per_setting_num_order": [5, 50],
    # "states_todo": ["AL","TX","NC"],
    "district_directory": "/home/ng343/gerrymandering_and_social_choice/",
    "minutes_freq_to_save": 30
    # "folder_save": "/home/ng343/cached_values/",
}

data_params = {
    "LOADING_FROM_DATA": True,
    "VOTER_RAW_FILENAME": "data/fullUS_individuals_with_FIP_6M_balanced_allelections.csv",
    "FEATURE_COLS": ["dem_partisan_score"],
    "GROUP_COL": "party",
    "N_VOTERS": 50000,
    "N_VOTERS_STV": 50000,
    "N_CANDIDATES_PER_DISTRICT_GROUP": 1,
    "N_STV_CANDIDATES_MAX": 1000,
    "DISTANCE_FUNCTION": "party_first_then_partisanscore",  #'party_first_then_partisanscore'#
    "CENSUS_TRACT_INFO_FILENAME": "data/state_df_tractinfo_combined.csv",
}
data_params.update(generator_params)
data_params.update(custom_params)
data_params.update({"label": "rebalanced20210530"})

output_template = 'outputs_20250921_fewercandidates_morevoters_gumbelnoise'


noise_parameters = {
    'ranking_method' : 'add_noise',
    'noise_distribution' : 'gumbel', # normal, gumbel
    'DISTANCE_FUNCTION' : 'just_partisanscore',
    'noise_parameters' : {
    'loc' : 0,
    'scale' : 0, 
    }
}

for scales in [0, 5, 50, 100, 250, 500]:
    noise_parameters['noise_parameters']['scale'] = scales
    data_params.update(noise_parameters)
    print("Running with noise scale:", scales)
    
    output = pipeline_multiple_states.meta_pipeline_states(
        data_params,
        save_file_template="cached_values/outputs/{}".format(output_template) + "_{}_.csv",
    )

    # Combine all the per state output files into one organized dataframe and save as csv
    # Code written 2025/09/06 as part of revision for paper
    process_csv_files_to_combined_into_organized_df(
        "cached_values/outputs/",
        r"^{}_.*\.csv$".format(output_template),
        "organized/combined_{}_stv.csv".format(output_template)
    )