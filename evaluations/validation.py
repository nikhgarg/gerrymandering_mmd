import pandas as pd
import os


def create_validation(parameters, competition_df):
    df = pd.read_csv(parameters["VALIDATION_DATA_FILENAME"])
    df.loc[:, "district"] = df[parameters["VALIDATION_DISTRICT_COL"]]
    df = df[[x for y in [["state", "district"], parameters["VALIDATION_SCORE_COLS"]] for x in y]]

    return df, parameters
