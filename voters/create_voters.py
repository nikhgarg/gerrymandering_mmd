import copy
import pandas as pd
import os

from voters.settings import *
from voters.helpers import *
from helpers import census_tract_string


def create_voters_simulate(parameters={}):
    p = parameters
    voters = []
    grouplist = []
    for group in range(p["N_GROUPS"]):
        groupletter = chr(ord("A") + group)  # ascii trick to convert group number to letter
        grouplist.append(groupletter)
        feature_samplings = []
        for feat in range(p["N_FEATURES"]):
            p["FEATURE_DIST_{}{}".format(groupletter, feat)] = p.get("FEATURE_DIST_{}{}".format(groupletter, feat), p["FEATURE_DIST"])
            feature_sampling = get_sampling_function(
                p["FEATURE_DIST_{}{}".format(groupletter, feat)]
            )  # get feature distribution for that groupfeature. If not present in the parameter dictionary, default back a generic FEATURE_DIST
            feature_samplings.append(feature_sampling)
        # print(feature_samplings
        for census_block in range(0, p["N_CENSUSBLOCKS"]):
            for studnum in range(0, int(p["N_VOTERS"] / p["N_CENSUSBLOCKS"] * p["CENSUSBLOCK_FRACTIONS_GROUPS"][census_block][group])):
                stud = {"group": groupletter, "census_block": census_block}
                stud.update({"feature_{}".format(x): feature_samplings[x]() for x in range(p["N_FEATURES"])})
                voters.append(stud)
    p["GROUPS"] = grouplist
    p["GROUP_COL"] = "group"
    p["FEATURE_COLS"] = ["feature_{}".format(x) for x in range(p["N_FEATURES"])]
    return pd.DataFrame(voters), p


def create_voters_data(parameters={}, state_fip_map_filename="data/state_fip_map.csv"):
    rawfilename = parameters["VOTER_RAW_FILENAME"]
    feature_cols = parameters["FEATURE_COLS"]
    group_col = parameters["GROUP_COL"]
    other_cols_to_keep = ["race", "education", "income"]
    parameters["NUM_FEATURES"] = len(feature_cols)
    state_fip_map = pd.read_csv(state_fip_map_filename)
    state_fip_map.loc[:, "state"] = state_fip_map.loc[:, "state"].apply(lambda x: "{:02d}".format(x))

    cleanfilename = "{}_{}.csv".format(rawfilename[:-4], "-".join(feature_cols + [group_col]))
    if os.path.exists(cleanfilename):
        print("Loading existing clean voter file", cleanfilename)
        df = pd.read_csv(cleanfilename)
    else:
        print("Creating clean voter file", cleanfilename)
        df = pd.read_csv(rawfilename)
        df = df.dropna(subset=feature_cols + ["dem_partisan_score", "census_FIP"])
        df.loc[df.dem_partisan_score <= 50, "party"] = "Republican"
        df.loc[df.dem_partisan_score >= 50, "party"] = "Democrat"

        rename_cols = {"census_FIP": "census_block"}
        df = df.rename(columns=rename_cols)
        if "customer_link" in df.columns:
            df = df.drop_duplicates(subset=["customer_link"])

        cols = ["census_block", group_col] + feature_cols + other_cols_to_keep
        df = df[cols]
        df.to_csv(cleanfilename, index=False)

    df.loc[:, "census_block"] = df.census_block.apply(census_tract_string)
    df.loc[:, "state"] = df.census_block.apply(census_tract_string).astype(str).str.slice(0, 2)
    # print(df.head())

    if "CENSUS_TRACT_INFO_FILENAME" in parameters:
        dfcensustract = pd.read_csv(parameters["CENSUS_TRACT_INFO_FILENAME"], usecols=["GEOID", "x", "y"])
        dfcensustract = dfcensustract.rename(columns={"GEOID": "census_tract"})
        dfcensustract.loc[:, "census_tract"] = dfcensustract.loc[:, "census_tract"].apply(census_tract_string)
        df["census_tract"] = df.census_block.apply(census_tract_string)
        df = df.merge(dfcensustract, on="census_tract", how="left")
        # print(df.head())
        df = df.dropna(subset=["x", "y"])
        # print(df.count())
        # print(df.head())

    if parameters["N_VOTERS"] is not None:
        df = (
            df.groupby("state")
            .apply(lambda x: x.iloc[np.random.choice(range(0, len(x)), size=min(len(x), parameters["N_VOTERS"]), replace=False)])
            .reset_index(drop=True)
        )

        # grab N_voters per state, or as many as there are

    # print(state_fip_map.head())
    # print(df.head())
    df = df.merge(state_fip_map[["state", "state_short"]], on="state", how="left")
    parameters["GROUPS"] = df[group_col].unique()
    # print("finished loading voters")
    return df, parameters
