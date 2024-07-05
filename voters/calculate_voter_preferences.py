import numpy as np
from voters.distance_functions import *
from functools import partial
from generic import pandas_apply_parallel


def _add_ranking(voter, candidates_df, distance_func, parameters):
    # for each candidate get distance between voter and candidate, and then generate ranking
    distances = []
    # print(voter)
    for en, can in candidates_df.iterrows():
        distances.append(distance_func(voter, can, parameters["GROUP_COL"]))
    # print(distances, np.argsort(distances))
    # print(distances)
    return candidates_df.iloc[np.argsort(distances)].id.tolist()


def add_candidate_rankings(voters_df, candidates_df, params, pool=None):
    # loop through districts
    # print(voters_df.columns, candidates_df.columns)
    distance_func = distance_function_mapper[params["DISTANCE_FUNCTION"]]
    for state in voters_df.state.unique():
        # print(state)
        # figure out which
        # for each voter, calculate distance to each candidate and then produce ranking -- add to the voter_df
        voters_in_state = voters_df.query("state==@state").index
        candidates_in_state = candidates_df.query("state==@state")  
        parfun = partial(_add_ranking, candidates_df=candidates_in_state, distance_func=distance_func, parameters=params)
        voters_df.loc[voters_in_state, "state_{}_ranking".format(state)] = pandas_apply_parallel.parallelize_on_rows(
            voters_df.loc[voters_in_state, :], parfun
        )

    return voters_df
