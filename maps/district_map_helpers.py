import pandas as pd
import pickle
import hashlib


def reformat_districts(x):
    l = x[-2:]
    try:
        return int(l)
    except Exception as e:
        return l


def get_map_hash_string(map):
    districts = list(map.census_block)
    strlong = "--".join(["-".join(sorted(x)) for x in districts])
    return hashlib.md5(strlong.encode()).hexdigest()


def load_district_map(parameters):
    # return a dataframe that maps censusblocks to districts
    # can read directly from parameters or from file

    param_dist_map = parameters["DISTRICT_MAP"]
    if type(param_dist_map) == str:
        df = load_district_map_from_file(param_dist_map)
        df.loc[:, "district"] = df.loc[:, "district"].apply(reformat_districts)
        df.loc[:, "state_num"] = df.loc[:, "census_block"].apply(lambda x: x[0][0:2])
        return df
        # return pd.read_csv(param_dist_map).groupby("district").agg(list).reset_index()
    else:  # convert dictionary to df  of appropriate form
        d = []
        for state in param_dist_map:
            for district in param_dist_map[state]:
                # for en in param_dist_map[district]:
                d.append({"state": state, "census_block": param_dist_map[state][district], "district": district})
        return pd.DataFrame(d)


def load_district_map_from_file(filename="data/us_district_to_tract_geoid_map.p"):
    if filename[-2:] == ".p":
        tract_mapping = pickle.load(open(filename, "rb"))
        c = []
        for state in tract_mapping:
            for district in tract_mapping[state]:
                c.append({"state": state, "district": district, "census_block": tract_mapping[state][district]})
        return pd.DataFrame(c)
    elif filename[-4:] == ".csv":
        return pd.read_csv(filename)
