default_parameters = {"DISTRICT_MAP": {0: [0, 1], 1: [2, 3]}, "N_WINNERS_PER_DISTRICT": 1, "VOTING_METHOD": "approval"}

data_params = {
    "LOADING_FROM_DATA": True,
    "VOTER_RAW_FILENAME": "data/NC_individuals_with_FIP_more.csv",
    "DISTRICT_MAP": "data/us_district_to_tract_geoid_map.p",
    "FEATURE_COLS": ["dem_partisan_score"],
    "GROUP_COL": "party",
    "N_VOTERS": None,
}
