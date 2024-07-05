from elections.helpers import *
from maps.district_map_helpers import *


def create_elections(parameters):

    competition_df = load_district_map(parameters)
    competition_df = create_competition_units(parameters, competition_df)

    return competition_df, parameters


def create_competition_units(parameters, competition_df):
    # for now, just add voting methods and number of winners in each unit to the dataframe
    for col in ["N_WINNERS_PER_DISTRICT"]:  # , "N_CANDIDATES_PER_DISTRICT_GROUP"]:
        if col not in competition_df.columns:
            competition_df.loc[:, col] = parameters[col]
        else:
            competition_df.loc[:, col] = competition_df.loc[:, col].replace(np.nan, parameters[col])
    competition_df.loc[:, "voting_method"] = parameters["VOTING_METHOD"]

    return competition_df
