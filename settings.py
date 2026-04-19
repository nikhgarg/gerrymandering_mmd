default_parameters = {
    "N_DISTRICTS": 2,
    "N_GROUPS": 2,  # party, ethnicity, etc
    "N_FEATURES": 1,
    "FEATURE_DIST": ("UNIF", 0, 100),
    "FEATURE_DIST_A0": ("UNIF", 50, 100),
    "FEATURE_DIST_B0": ("UNIF", 0, 50),
    "district_directory": "C:/Users/Nikhi/Box/Gerrymandering_and_Social_Choice/",
}

foldersaves = "cached_values/"

relevant_params_for_cache_all = ["VOTER_RAW_FILENAME", "N_VOTERS", "FEATURE_COLS", "GROUP_COL", "LOADING_FROM_DATA"]
relevant_params_for_cache_per_voting_method = {
    "stv": [
        "N_VOTERS_STV",
        "N_CANDIDATES_PER_DISTRICT_GROUP",
        "DISTANCE_FUNCTION",
        "N_STV_CANDIDATES_MAX",
        'ranking_method',
        'noise_distribution',
        'noise_parameters'
    ]
}

voting_function_multiple_names_map = {
    "stv": "thiele_pav",
    "thiele_pav": "thiele_pav",
    "thiele_approvalindependent": "thiele_independent",
    "thiele_squared": "thiele_squared",
}


# distributions for voter ranking noise
settings_normal_noise = {
    "noise_distribution": "normal",
    "noise_parameters": {"loc": 0, "scale": 1},
}

settings_gumbel_noise = {
    "noise_distribution": "gumbel",
    "noise_parameters": {"loc": 0, "scale": 1},
}