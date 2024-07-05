import copy
from generic.latexify import *
from generic.pandas_apply_parallel import *
import multiprocessing
import random

# import settings
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
from candidates.create_candidates_and_voter_preferences_pipeline import *

import settings
from helpers import *
from settings import *

import multiprocessing
from functools import partial

import time

import generic.pandas_apply_parallel as pandas_apply_parallel


def voting_function_map_generator(voting_functions, map_generator, done_outputs, params, state):
    skipcount = 0
    for maphash, competition_df, map_characteristics in map_generator:
        for voting_function in voting_functions:
            if map_characteristics["optimization_characteristics"]["optimization"] in ["fair", "unfair"]:
                if not (
                    voting_function_multiple_names_map[voting_function]
                    == map_characteristics["optimization_characteristics"]["optimization_voting_method_for"]
                ):
                    continue  # if a map is an optimal map for some voting method, then only use that map for that voting method
            method_hash_cols = relevant_params_for_cache_per_voting_method.get(voting_function, []) + relevant_params_for_cache_all
            methodsettingshash = get_param_str_from_dict({x: params[x] for x in method_hash_cols})
            overallhash = methodsettingshash + maphash + voting_function
            # print("skipping because already done: ", overallhash)
            if not overallhash in done_outputs:
                paramsloc = copy.copy(params)
                paramsloc["VOTING_METHOD"] = voting_function
                competition_df["voting_method"] = voting_function
                output = {"overall_hash": overallhash, "map_hash": maphash, "settings_hash": methodsettingshash, "state": state}
                output.update(map_characteristics)
                output.update({x: params[x] for x in method_hash_cols})

                yield (competition_df, paramsloc, output)
            else:
                skipcount += 1
                if skipcount % 100 == 0:
                    print("(skipped: {})".format(skipcount), end=" ")


def meta_pipeline_states(custom_parameters, save_file_template="cached_values/run_outputs_{}.csv"):
    pool = multiprocessing.Pool()
    pandas_apply_parallel.set_number_of_processors(6)
    params = copy.deepcopy(settings.default_parameters)
    params.update(voters.settings.default_parameters)
    params.update(elections.settings.default_parameters)
    params.update(evaluations.settings.default_parameters)
    params.update(custom_parameters)
    # print(params)
    if "stv" in params["VOTING_METHODS"]:
        params.update(candidates.settings.default_parameters)
        params.update(custom_parameters)

    if params.get("LOADING_FROM_DATA", False):
        voters_df, params = create_voters_data(parameters=params)
    else:
        voters_df, params = create_voters_simulate(parameters=params)

    print(voters_df.shape)
    voting_functions = params["VOTING_METHODS"]

    numdone = 0
    foldersavesloc = foldersaves  # params.get("folder_save", foldersaves)

    # print(voters_df.head())
    # print(params)
    # print(params["maps_per_setting_num"])
    per_setting_num = params["maps_per_setting_num"]
    setting_nums = params.get("maps_per_setting_num_order", list(range(20, per_setting_num, 20))) + [per_setting_num]
    minutes_freq_to_save = params.get("minutes_freq_to_save", 1)
    print(setting_nums)
    for setting_num in setting_nums:
        for state in params.get("states_todo", voters_df.state_short.unique()):
            save_file = save_file_template.format(state)
            if os.path.exists(save_file):
                output_df = pd.read_csv(save_file)
                done_outputs = set(output_df.overall_hash)
            else:
                output_df = pd.DataFrame()
                done_outputs = set()
            print("state: ", state, "setting_num: ", setting_num)
            voters_df_loc = voters_df.query("state_short==@state")
            print("num_voters in state: ", voters_df_loc.shape[0])
            if "stv" in voting_functions:
                print("creating candidates and calculating voter rankings")
                relevant_params_for_stv_cache = relevant_params_for_cache_per_voting_method["stv"] + relevant_params_for_cache_all
                paramcachestr = get_param_str_from_dict({x: params[x] for x in relevant_params_for_stv_cache}) + state
                candidates_df, voters_df_loc, params = get_candidates_and_voter_preferences_for_stv(
                    voters_df_loc, params, paramcachestr, foldersavesloc
                )
                params.update(
                    custom_parameters
                )  # in loading old parameters that were added with candidates, some things might have been overwritten
            else:
                candidates_df = None
            # return
            # print(voters_df.head())
            # print(voters_df_loc.columns)
            # print(params["maps_per_setting_num"])

            map_generator = map_generators[params["MAP_GENERATOR"]](
                state,
                voters_df_loc.iloc[0].state,
                maps_per_district_num_setting=setting_num,
                district_directory=params["district_directory"],
            )  # (params)
            overall_generator = voting_function_map_generator(voting_functions, map_generator, done_outputs, params, state)

            pipeline_with_voters = partial(pipeline, voters_df=voters_df_loc, candidates_df=candidates_df)

            outputs = []
            count = 0
            print("starting imap with pipeline")
            ts = time.time()
            for output in pool.imap_unordered(pipeline_with_voters, overall_generator):
                outputs.append(output)
                curtime = time.time()
                if (curtime - ts) / 60 > minutes_freq_to_save:  # been more than 5 minute(s) since last save
                    print(count, end=" ")
                    output_df = pd.concat([output_df, pd.DataFrame(outputs)])
                    output_df.to_csv(save_file, index=False)
                    outputs = []
                    ts = curtime
                count += 1
            output_df = pd.concat([output_df, pd.DataFrame(outputs)])
            if output_df.shape[0] > 0:
                output_df.to_csv(save_file, index=False)

    return output_df


def pipeline(multiinput, voters_df=None, candidates_df=None):
    competition_df, params, output = multiinput

    ts = time.time()
    competition_df = elections.determine_winners.determine_winners(voters_df, competition_df, params, candidates_df)
    metric_calculations, competition_df = metrics_median.calculate_all_metrics(voters_df, competition_df, params)
    elapsed = time.time() - ts
    metric_calculations.update({"run_label": params["label"], "voting_method": params["VOTING_METHOD"], "runtime": elapsed})
    output.update(metric_calculations)
    return output
