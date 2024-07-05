import numpy as np
from functools import partial
from elections.helpers import geometric_median
from scipy.spatial.distance import euclidean
import pandas as pd

"""
This file contains functions for how "cohesive" a set of voters who elected a winner are --- given voters who selected a candidate,
how cohesive are they along various dimensions (e.g., partisan score average distance, geographic distance)
"""


# Given a set of scores X, cohesion is defined as the standard deviation: (E(E[X] - X)^2)^.5
def generic_singledimensionscore_cohesion(voters_who_selected_winner, voter_weights, score_column, column_name=None):
    if column_name is None:
        column_name = score_column

    vals = voters_who_selected_winner[score_column]
    isnotnan = [~np.isnan(x) for x in vals]
    vals = np.array(vals)[isnotnan]
    voter_weights = np.array(voter_weights)[isnotnan]
    mean = np.average(vals, weights=voter_weights)
    mse = np.average([(val - mean) ** 2 for val in vals], weights=voter_weights)
    # return {column_name: -np.std(vals)}
    return {column_name: -np.sqrt(mse)}


partisan_score = partial(generic_singledimensionscore_cohesion, score_column="dem_partisan_score", column_name="partisan_score")
income = partial(generic_singledimensionscore_cohesion, score_column="income", column_name="income")
education = partial(generic_singledimensionscore_cohesion, score_column="education", column_name="education")


def geographic_cohesion(voters_who_selected_winner, voter_weights, lat="x", long="y"):
    xy = voters_who_selected_winner[[lat, long]].to_numpy()
    median = geometric_median(xy, voter_weights)
    cohesion = np.average([euclidean(xy[i, :], median) for i in range(np.shape(xy)[0])], weights=voter_weights)
    return {"geographic": -cohesion}


from scipy.stats import entropy


def racial_cohesion(voters_who_selected_winner, voter_weights):
    race = pd.get_dummies(voters_who_selected_winner.race)
    racecols = list(voters_who_selected_winner.race.unique())
    race["weight"] = np.array(voter_weights) / sum(voter_weights)
    for col in racecols:
        race.loc[:, col] = race.apply(lambda x: x[col] * x["weight"], axis=1)
    vals = list(race[racecols].sum())
    return {"racial": -entropy(vals)}


cohesion_functions = [partisan_score, geographic_cohesion, racial_cohesion, income, education]


def calculate_all_cohesion_scores(voters_who_selected_winner, voter_weights):
    scores = {}
    for fun in cohesion_functions:
        scores.update(fun(voters_who_selected_winner, voter_weights))
    return scores
