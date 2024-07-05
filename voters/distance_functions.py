from functools import lru_cache
from helpers import euclidean_cached
import numpy as np


def party_first_then_feature0(voter_row, candidate_row, feature0name, groupname):
    return -1000 * (voter_row[groupname] == candidate_row[groupname]) + abs(voter_row[feature0name] - candidate_row[feature0name])


def party_first_then_partisanscore(voter_row, candidate_row, groupname):
    return party_first_then_feature0(voter_row, candidate_row, "dem_partisan_score", groupname)


def party_first_then_geographicdistance(voter_row, candidate_row, groupname, lat="x", long="y"):
    distance = euclidean_cached((voter_row[lat], voter_row[long]), (candidate_row[lat], candidate_row[long]))
    # print(groupname)
    return -1000 * (voter_row[groupname] == candidate_row[groupname]) + np.log(distance + 1)


distance_function_mapper = {x.__name__: x for x in [party_first_then_partisanscore, party_first_then_geographicdistance]}
