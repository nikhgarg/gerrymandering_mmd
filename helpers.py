from functools import lru_cache
from scipy.spatial.distance import euclidean
import hashlib


@lru_cache
def euclidean_cached(ar1, ar2):
    return euclidean(ar1, ar2)


def census_tract_string(x):
    return str("{:011d}".format(int(x)) if len(str(int(x))) < 11 else str(int(x)))


def get_param_str_from_dict(params):
    paramstr = ["{}{}".format(x, params[x]) for x in params]
    paramstr = "".join(paramstr)
    hash_object = hashlib.md5(paramstr.encode())
    paramstr = hash_object.hexdigest()
    return paramstr


import os.path
from os import path
import pickle


def get_pickle_name(foldersaves, paramstr, filename):
    return "{}{}_{}.p".format(foldersaves, paramstr, filename)


def pickleload(foldersaves, paramstr, filename):
    return pickle.load(open(get_pickle_name(foldersaves, paramstr, filename), "rb"))


def pickledump(val, foldersaves, paramstr, filename):
    pickle.dump(val, open(get_pickle_name(foldersaves, paramstr, filename), "wb"))
