import copy
from generic.latexify import *
from generic.pandas_apply_parallel import *
import multiprocessing
import random

import settings
import voters.settings
from voters.create_voters import create_voters_simulate, create_voters_data
import voters.calculate_voter_preferences

from maps.map_generators import *

import elections.settings
import elections.determine_winners

import evaluations.settings
from evaluations import metrics_median
from evaluations.validation import create_validation

import candidates.settings
from candidates.create_candidates import create_candidates_data

from helpers import *

import multiprocessing
from functools import partial

foldersaves = "cached_values/"

relevant_params_for_cache_all = ["VOTER_RAW_FILENAME", "N_VOTERS", "FEATURE_COLS", "GROUP_COL", "LOADING_FROM_DATA"]
relevant_params_for_cache_per_voting_method = {
    "stv": [
        "N_VOTERS_STV",
        "N_CANDIDATES_PER_DISTRICT_GROUP",
        "DISTANCE_FUNCTION",
        "N_STV_CANDIDATES_MAX",
    ]
}


def voting_function_map_generator(voting_functions, map_generator, done_outputs, params):
    for maphash, competition_df, map_characteristics in map_generator:
        for voting_function in voting_functions:
            method_hash_cols = relevant_params_for_cache_per_voting_method.get(voting_function, []) + relevant_params_for_cache_all
            methodsettingshash = get_param_str_from_dict({x: params[x] for x in method_hash_cols})
            overallhash = methodsettingshash + maphash + voting_function
            # print("skipping because already done: ", overallhash)
            if not overallhash in done_outputs:
                paramsloc = copy.copy(params)
                paramsloc["VOTING_METHOD"] = voting_function
                competition_df["voting_method"] = voting_function
                output = {"overall_hash": overallhash, "map_hash": maphash, "settings_hash": methodsettingshash}
                output.update(map_characteristics)
                output.update({x: params[x] for x in method_hash_cols})

                yield (competition_df, paramsloc, output)


def meta_pipeline(custom_parameters, save_file="cached_values/run_outputs.csv"):
    params = copy.deepcopy(settings.default_parameters)
    params.update(voters.settings.default_parameters)
    params.update(elections.settings.default_parameters)
    params.update(evaluations.settings.default_parameters)
    params.update(custom_parameters)
    if "stv" in params["VOTING_METHODS"]:
        params.update(candidates.settings.default_parameters)
        params.update(custom_parameters)

    if params.get("LOADING_FROM_DATA", False):
        voters_df, params = create_voters_data(parameters=params)
    else:
        voters_df, params = create_voters_simulate(parameters=params)

    print(voters_df.shape)
    voting_functions = params["VOTING_METHODS"]

    print("creating candidates and calculating voter rankings")
    if "stv" in voting_functions:

        relevant_params_for_stv_cache = relevant_params_for_cache_per_voting_method["stv"] + relevant_params_for_cache_all
        paramcachestr = get_param_str_from_dict({x: params[x] for x in relevant_params_for_stv_cache})

        if path.exists(get_pickle_name(foldersaves, paramcachestr, "candidates")):
            print("loading STV candidates and voters from cache")
            candidates_df = pickleload(foldersaves, paramcachestr, "candidates")
            voters_df = pickleload(foldersaves, paramcachestr, "voters")
            params = pickleload(foldersaves, paramcachestr, "params")
            candidates_df.loc[:, "id"] = candidates_df.loc[:, "id"].astype(str)
            # print(candidates_df.head())
            # print(voters_df.head())

        else:
            sample = random.sample(
                list(range(voters_df.shape[0])), params.get("N_VOTERS_STV", params["N_VOTERS"])
            )  # get random voter indices
            others = list(set(range(voters_df.shape[0])) - set(sample))
            voters_to_get_rankings = voters_df.iloc[sample, :]

            candidates_df, params = create_candidates_data(params, voters_to_get_rankings)
            print(candidates_df.head())
            print(voters_df.head())
            # if params["N_STV_CANDIDATES_MAX"] is not None:
            # candidates_df = candidates_df.groupby("state").sample(params["N_STV_CANDIDATES_MAX"]).reset_index()

            candidates_df = candidates_df.sample(params["N_STV_CANDIDATES_MAX"])
            print("finished creating candidates: ", candidates_df.shape)
            # print(candidates_df.state.value_counts())

            voters_to_get_rankings = voters.calculate_voter_preferences.add_candidate_rankings(
                voters_to_get_rankings, candidates_df, params
            )
            other_voters = voters_df.iloc[others, :]
            voters_df = pd.concat([other_voters, voters_to_get_rankings])
            print(voters_to_get_rankings.shape, other_voters.shape, voters_df.shape)
            print("finished calculating voter rankings")
            pickledump(candidates_df, foldersaves, paramcachestr, "candidates")
            pickledump(voters_df, foldersaves, paramcachestr, "voters")
            pickledump(params, foldersaves, paramcachestr, "params")
    else:
        candidates_df = None

    # print(voters_df.head())

    numdone = 0
    if os.path.exists(save_file):
        output_df = pd.read_csv(save_file)
        done_outputs = set(output_df.overall_hash)
    else:
        output_df = pd.DataFrame()
        done_outputs = set()

    map_generator = map_generators[params["MAP_GENERATOR"]](maps_per_district_num=params["maps_per_district_num"])  # (params)
    overall_generator = voting_function_map_generator(voting_functions, map_generator, done_outputs, params)

    pool = multiprocessing.Pool()
    pipeline_with_voters = partial(pipeline, voters_df=voters_df, candidates_df=candidates_df)

    outputs = []
    count = 0
    for output in pool.imap_unordered(pipeline_with_voters, overall_generator):
        outputs.append(output)
        print(count, end=" ")
        if count % 5 == 1:
            output_df = pd.concat([output_df, pd.DataFrame(outputs)])
            output_df.to_csv(save_file, index=False)
            outputs = []
        count += 1
    output_df = pd.concat([output_df, pd.DataFrame(outputs)])
    output_df.to_csv(save_file, index=False)

    #
    # for maphash, competition_df, map_characteristics in map_generator:
    #     numdone += 1
    #     print(numdone, maphash, end=" ")
    #     for voting_function in voting_functions:
    #         method_hash_cols = relevant_params_for_cache_per_voting_method.get(voting_function, []) + relevant_params_for_cache_all
    #         methodsettingshash = get_param_str_from_dict({x: params[x] for x in method_hash_cols})
    #         overallhash = methodsettingshash + maphash + voting_function
    #         if overallhash in done_outputs:
    #             print("skipping because already done: ", overallhash)
    #         else:
    #             paramsloc = copy.copy(params)
    #             paramsloc["VOTING_METHOD"] = voting_function
    #             competition_df["voting_method"] = voting_function
    #             output = {"overall_hash": overallhash, "map_hash": maphash, "settings_hash": methodsettingshash}
    #             output.update(map_characteristics)
    #             output.update(pipeline(voters_df, competition_df, candidates_df, paramsloc))
    #             output.update({x: params[x] for x in method_hash_cols})
    #             outputs.append(output)
    return output_df


def pipeline(multiinput, voters_df=None, candidates_df=None):
    competition_df, params, output = multiinput
    competition_df = elections.determine_winners.determine_winners(voters_df, competition_df, params, candidates_df)
    metric_calculations, competition_df = metrics_median.calculate_all_metrics(voters_df, competition_df, params)
    metric_calculations.update({"run_label": params["label"], "voting_method": params["VOTING_METHOD"]})
    output.update(metric_calculations)
    return output
