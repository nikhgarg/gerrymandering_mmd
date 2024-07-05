import copy
from generic.latexify import *
from generic.pandas_apply_parallel import *
import multiprocessing


import settings
import voters.settings
from voters.create_voters import create_voters_simulate, create_voters_data
import voters.calculate_voter_preferences

import candidates.settings
from candidates.create_candidates import create_candidates_data
import elections.settings
from maps.create_elections import create_elections
import elections.determine_winners

import evaluations.settings
from evaluations import metrics_median
from evaluations.validation import create_validation


def pipeline(custom_parameters):
    params = copy.deepcopy(settings.default_parameters)
    params.update(voters.settings.default_parameters)
    params.update(elections.settings.default_parameters)
    if params["VOTING_METHOD"] in ["stv"]:
        params.update(candidates.settings.default_parameters)
    params.update(evaluations.settings.default_parameters)

    # params.update(elections.settings.default_parameters)
    params.update(custom_parameters)

    competition_df, params = create_elections(parameters=params)
    if params.get("do_validation", False):
        validation_df, params = create_validation(params, competition_df)
    else:
        validation_df = None
    # return validation_df, competition_df

    if params.get("LOADING_FROM_DATA", False):
        voters_df, params = create_voters_data(parameters=params)
    else:
        voters_df, params = create_voters_simulate(parameters=params)

    if params["VOTING_METHOD"] in ["stv"]:
        print("creating candidates and calculating voter rankings")
        candidates_df, params = create_candidates_data(params, voters_df)
        print("finished creating candidates: ", candidates_df.shape)
        # print(candidates_df.state.value_counts())
        voters_df = voters.calculate_voter_preferences.add_candidate_rankings(voters_df, candidates_df, params)
        print("finished calculating voter rankings: ", candidates_df.shape)

    else:
        candidates_df = None
    # return voters_df, candidates_df
    # return voters_df, candidates_df, competition_df, validation_df, params

    # print(params)

    print("calculating winners")
    competition_df = elections.determine_winners.determine_winners(voters_df, competition_df, params, candidates_df)

    print("calculating metrics")
    metric_calculations, competition_df = metrics_median.calculate_all_metrics(voters_df, competition_df, params)
    return voters_df, competition_df, validation_df, metric_calculations, candidates, params

    # return voters_df, candidates_df, competition_df, validation_df, metric_calculations, params
