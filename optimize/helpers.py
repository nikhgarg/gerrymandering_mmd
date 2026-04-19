import os
import numpy as np
import pandas as pd

# TODO (hwr26): this is a bit of a cheat to get around introducing the third
# party at the census tract level which would result in re-generating the
# districts which is the most computationally heavy task
# This number (it appears 2 districts is best) does a sufficent job at estimating
# the average voteshare by party
def calculate_statewide_average_voteshare_with_third_party(state, ddf_save_path):
    df = pd.read_csv(os.path.join(ddf_save_path, f"{state}_2_district_df.csv"), index_col=0)
    cols = [f"{p}_mean" for p in ["r", "d", "t"]]
    return np.average(np.array(df[cols]), axis=0, weights=df["population"])