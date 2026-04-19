from helpers import *
from settings import *
from candidates.create_candidates import *
import voters.calculate_voter_preferences


def get_candidates_and_voter_preferences_for_stv(voters_df, params, paramcachestr, foldersaves):
    """
    Either loads from cache or creates and saves to cache the candidates and voter_df for the given parameters.

    :param voters_df: The original voter_df
    :param params: The parameters to use for creating the candidates and determining which voters to use
    :param paramcachestr: The string to use to identify the cache
    :param foldersaves: The folder to save the cache in
    :return: The candidates_df, the voters_df with rankings, and the params
    """
    print(params)
    if path.exists(get_pickle_name(foldersaves, paramcachestr, "candidates")):
        print("loading STV candidates and voters from cache")
        candidates_df = pickleload(foldersaves, paramcachestr, "candidates")
        voters_df = pickleload(foldersaves, paramcachestr, "voters")
        params = pickleload(foldersaves, paramcachestr, "params")
        candidates_df.loc[:, "id"] = candidates_df.loc[:, "id"].astype(str)
        # print(params)

        # print(candidates_df.head())
        # print(voters_df.head())

    else:
        num_voters_orig = voters_df.shape[0]
        sample = random.sample(
            list(range(num_voters_orig)), min(params.get("N_VOTERS_STV", params["N_VOTERS"]), num_voters_orig)
        )  # get random voter indices
        others = list(set(range(voters_df.shape[0])) - set(sample))
        voters_to_get_rankings = voters_df.iloc[sample, :]

        candidates_df, params = create_candidates_data(params, voters_to_get_rankings)
        print(candidates_df.party.value_counts())
        # print(candidates_df.head())
        # print(voters_df.head())
        # if params["N_STV_CANDIDATES_MAX"] is not None:
        # candidates_df = candidates_df.groupby("state").sample(params["N_STV_CANDIDATES_MAX"]).reset_index()


        #sample same number of candidates per params["GROUP_COL"]
        number_of_groups = candidates_df[params["GROUP_COL"]].nunique()
        candidates_df = candidates_df.groupby(params["GROUP_COL"]).apply(
            lambda x: x.sample(min(int(params["N_STV_CANDIDATES_MAX"]/number_of_groups), x.shape[0]))
        ).reset_index(drop=True)

        # candidates_df = candidates_df.sample(min(params["N_STV_CANDIDATES_MAX"], candidates_df.shape[0]))
        # print(candidates_df.party.value_counts())

        print("finished creating candidates: ", candidates_df.shape)
        # print(candidates_df.state.value_counts())

        voters_to_get_rankings = voters.calculate_voter_preferences.add_candidate_rankings(voters_to_get_rankings, candidates_df, params)
        other_voters = voters_df.iloc[others, :]
        voters_df = pd.concat([other_voters, voters_to_get_rankings])
        print(voters_to_get_rankings.shape, other_voters.shape, voters_df.shape)
        print("finished calculating voter rankings")
        pickledump(candidates_df, foldersaves, paramcachestr, "candidates")
        pickledump(voters_df, foldersaves, paramcachestr, "voters")
        pickledump(params, foldersaves, paramcachestr, "params")
    return candidates_df, voters_df, params
