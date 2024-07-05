import copy
import pandas as pd
import random

from candidates.settings import *
from candidates.helpers import *
from voters.helpers import *
from helpers import census_tract_string


def create_candidates_data(p, voters_df):
    # Create 1 candidate per census tract per party per partisan score quantile (also a parameter)
    candidates = []
    num_candidates_per_district_group = p["N_CANDIDATES_PER_DISTRICT_GROUP"]
    for census in voters_df.census_block.unique():
        statestr = census_tract_string(census)[0:2]
        voters_in_tract = voters_df.query("census_block==@census")
        for group in voters_in_tract[p["GROUP_COL"]].unique():
            votersparty = voters_in_tract.query("{}==@group".format(p["GROUP_COL"]))
            _, bins = pd.qcut(votersparty["dem_partisan_score"], num_candidates_per_district_group, retbins=True, duplicates="drop")
            # print(bins, len(bins))
            for i in bins:
                cand = {
                    "id": str(random.getrandbits(32)),
                    p["GROUP_COL"]: group,
                    "census_block": census,
                    "dem_partisan_score": i,
                    "state": statestr,
                }
                candidates.append(cand)
    candidates_df = pd.DataFrame(candidates)

    if "CENSUS_TRACT_INFO_FILENAME" in p:
        dfcensustract = pd.read_csv(p["CENSUS_TRACT_INFO_FILENAME"], usecols=["GEOID", "x", "y"])
        dfcensustract = dfcensustract.rename(columns={"GEOID": "census_tract"})
        dfcensustract.loc[:, "census_tract"] = dfcensustract.loc[:, "census_tract"].apply(census_tract_string)
        candidates_df["census_tract"] = candidates_df.census_block.apply(census_tract_string)
        candidates_df = candidates_df.merge(dfcensustract, on="census_tract", how="left")

    return candidates_df, p


def create_candidates_simulate(parameters, competition_df):
    p = parameters
    candidates = []
    grouplist = []
    for group in range(p["N_GROUPS"]):
        groupletter = p["GROUPS"][group]  # chr(ord("A") + group)  # ascii trick to convert group number to letter
        grouplist.append(groupletter)
        feature_samplings = []
        for feat in range(p["N_FEATURES"]):
            p["FEATURE_DIST_{}{}".format(groupletter, feat)] = p.get("FEATURE_DIST_{}{}".format(groupletter, feat), p["FEATURE_DIST"])
            feature_sampling = get_sampling_function(
                p["FEATURE_DIST_{}{}".format(groupletter, feat)]
            )  # get feature distribution for that groupfeature. If not present in the parameter dictionary, default back a generic FEATURE_DIST
            feature_samplings.append(feature_sampling)
        for en, district_row in competition_df.iterrows():
            district = district_row.district
            state = district_row.state
            for studnum in range(0, district_row["N_CANDIDATES_PER_DISTRICT_GROUP"]):
                stud = {"id": str(random.getrandbits(32)), p["GROUP_COL"]: groupletter, "district": district, "state": state}
                stud.update({p["FEATURE_COLS"][x]: feature_samplings[x]() for x in range(p["N_FEATURES"])})
                candidates.append(stud)
    return pd.DataFrame(candidates), p
