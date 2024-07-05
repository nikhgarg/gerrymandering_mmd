import pickle
import pandas as pd
from helpers import *
from maps.district_map_helpers import *
import os


def from_optimization_and_sampling_runs_per_state(
    state, state_num, maps_per_district_num_setting=None, district_directory="C:/Users/Nikhi/Box/Gerrymandering_and_Social_Choice/"
):
    # get all filenames in directory that have the right state (unique number of districts)
    print(maps_per_district_num_setting)
    files_for_state = [
        x for x in os.listdir("{}multimember_generation/district_dfs/".format(district_directory)) if "{}_".format(state) in x
    ]
    print(files_for_state)
    all_blocks_in_state = None
    num_seats_in_state = None

    # loop through the filenames (state/unique number of districts)
    for filee in files_for_state:
        print(filee)
        # load the block assignments and the distict_df for this
        statedistrict = filee.replace("_district_df.csv", "")
        num_districts = int(filee.split("_")[1])
        district_df = pd.read_csv(
            "{}{}_district_df.csv".format("{}multimember_generation/district_dfs/".format(district_directory), statedistrict)
        )
        district_df.set_index("node_id", drop=False, inplace=True, verify_integrity=True)
        block_assignments = pickle.load(
            open(
                "{}{}_block_assignment.p".format("{}multimember_generation/block_assignments/".format(district_directory), statedistrict),
                "rb",
            )
        )
        subsampleplanfile = "{}{}_plans.p".format("{}multimember_generation/subsampled_plans/".format(district_directory), statedistrict)
        if os.path.exists(subsampleplanfile):
            # print(subsampleplanfile)
            print("subsampled", end=" ")
            subsampled_plans = pickle.load(open(subsampleplanfile, "rb"))
            # create a single district map, yield it. Also compare its census blocks to the rest later to ensure valid map
            if all_blocks_in_state is None:
                all_blocks_in_state, num_seats_in_state = create_single_district_in_state_files(
                    subsampled_plans[0], block_assignments, district_df
                )
                d = [
                    {
                        "state": state,
                        "state_num": state_num,
                        "census_block": list(all_blocks_in_state),
                        "district": "singledistrictmap",
                        "N_WINNERS_PER_DISTRICT": int(num_seats_in_state),
                    }
                ]
                # map_label = "{}".format(num_districts)
                map = pd.DataFrame(d)
                maphash = get_map_hash_string(map)  # get the maphash based on the map
                optimization_characteristics = {
                    "optimization": "single_district_for_state",
                    "number": 0,
                }  # also get a setting for the map that is based on state, district_num, optimized string, and map number (ordering in the file)
                # the setting thing can be a dictionary that I just save itself as a column in outputs, and then eval it when reading from it
                map_characteristics = {
                    "state": state,
                    "state_num": state_num,
                    "N_districts": 1,
                    "optimization_characteristics": optimization_characteristics,
                }
                yield maphash, map, map_characteristics  #  just yield from here without checking if its been done

            # loop through the subsampled maps, and optimized maps for this setting
            for enmap, mapp in enumerate(subsampled_plans):
                d = []
                for node_id in mapp:
                    # do the blockassignment and district_df method thing to get the actual map, district df, etc.
                    district_blocks = block_assignments[node_id]
                    d.append(
                        {
                            "state": state,
                            "state_num": state_num,
                            "census_block": [census_tract_string(x) for x in district_blocks],
                            "district": str(node_id),
                            "N_WINNERS_PER_DISTRICT": int(
                                district_df.loc[node_id].n_seats
                            ),  # number of candidates for the district are in district_df
                        }
                    )
                # map_label = "{}".format(num_districts)
                map = pd.DataFrame(d)
                maphash = get_map_hash_string(map)  # get the maphash based on the map
                optimization_characteristics = {
                    "optimization": "subsampled",
                    "number": enmap,
                }  # also get a setting for the map that is based on state, district_num, optimized string, and map number (ordering in the file)
                # the setting thing can be a dictionary that I just save itself as a column in outputs, and then eval it when reading from it
                map_characteristics = {
                    "state": state,
                    "state_num": state_num,
                    "N_districts": num_districts,
                    "optimization_characteristics": optimization_characteristics,
                }
                dists = [x for y in list(map.census_block) for x in y]
                assert (
                    len(dists) == len(set(dists))
                    and set(dists) == all_blocks_in_state
                    and sum(list(map.N_WINNERS_PER_DISTRICT)) == num_seats_in_state
                )  # no block repeated, all blocks are covered, right number of winners
                yield maphash, map, map_characteristics  #  just yield from here without checking if its been done
                # print(enmap, maps_per_district_num_setting, maps_per_district_num_setting)
                if maps_per_district_num_setting is not None and enmap > maps_per_district_num_setting:
                    break
            del subsampled_plans

        # do the same thing for the optimized maps
        optimizedplanfile = "{}/{}_opt_results.p".format(
            "{}multimember_generation/optimization_results/".format(district_directory), statedistrict
        )
        if os.path.exists(optimizedplanfile):
            # print(optimizedplanfile)
            print("optimized", end=" ")
            optimized_plans_overallfile = pickle.load(open(optimizedplanfile, "rb"))
            # loop through the subsampled maps, and optimized maps for this setting

            for voting_method_optimized_for in optimized_plans_overallfile["fair"]:
                for enmap, mappnum in enumerate(optimized_plans_overallfile["fair"][voting_method_optimized_for]):
                    d = []
                    # print(optimized_plans_overallfile["fair"][voting_method_optimized_for][mappnum].keys())
                    for nodeindex in optimized_plans_overallfile["fair"][voting_method_optimized_for][mappnum]["solution_ixs"]:
                        # print(nodeindex)
                        district_df_row = district_df.iloc[nodeindex]  # iloc here bc indices
                        node_id = district_df_row.node_id
                        # print(node_id)
                        district_blocks = block_assignments[node_id]
                        d.append(
                            {
                                "state": state,
                                "state_num": state_num,
                                "census_block": [census_tract_string(x) for x in district_blocks],
                                "district": str(node_id),
                                "N_WINNERS_PER_DISTRICT": int(
                                    district_df_row.n_seats
                                ),  # number of candidates for the district are in district_df
                            }
                        )
                    # map_label = "{}".format(num_districts)
                    map = pd.DataFrame(d)
                    maphash = get_map_hash_string(map)  # get the maphash based on the map
                    optimization_characteristics = {
                        "optimization": "fair",
                        "optimization_voting_method_for": voting_method_optimized_for,
                        "number": mappnum,
                    }
                    map_characteristics = {
                        "state": state,
                        "state_num": state_num,
                        "N_districts": num_districts,
                        "optimization_characteristics": optimization_characteristics,
                    }
                    dists = [x for y in list(map.census_block) for x in y]
                    assert (
                        len(dists) == len(set(dists))
                        and set(dists) == all_blocks_in_state
                        and sum(list(map.N_WINNERS_PER_DISTRICT)) == num_seats_in_state
                    )  # no block repeated, all blocks are covered, right number of winners
                    yield maphash, map, map_characteristics  #  just yield from here without checking if its been done
                    # print(enmap, maps_per_district_num_setting, maps_per_district_num_setting)
                    if maps_per_district_num_setting is not None and enmap > maps_per_district_num_setting:
                        break
            for voting_method_optimized_for in optimized_plans_overallfile["unfair"]:
                # print(optimized_plans_overallfile["unfair"][voting_method_optimized_for].keys())
                for party_optimized_for in ["r_opt_solutions", "d_opt_solutions"]:
                    for enmap, mapp in enumerate(optimized_plans_overallfile["unfair"][voting_method_optimized_for][party_optimized_for]):
                        d = []
                        for node_id in mapp:
                            district_df_row = district_df.loc[node_id]
                            district_blocks = block_assignments[node_id]  # iloc here bc indices
                            d.append(
                                {
                                    "state": state,
                                    "state_num": state_num,
                                    "census_block": [census_tract_string(x) for x in district_blocks],
                                    "district": str(node_id),
                                    "N_WINNERS_PER_DISTRICT": int(
                                        district_df_row.n_seats
                                    ),  # number of candidates for the district are in district_df
                                }
                            )
                        # map_label = "{}".format(num_districts)
                        map = pd.DataFrame(d)
                        maphash = get_map_hash_string(map)  # get the maphash based on the map
                        optimization_characteristics = {
                            "optimization": "unfair",
                            "optimization_voting_method_for": voting_method_optimized_for,
                            "party_optimized_for": party_optimized_for,
                            "number": enmap,
                        }
                        map_characteristics = {
                            "state": state,
                            "state_num": state_num,
                            "N_districts": num_districts,
                            "optimization_characteristics": optimization_characteristics,
                        }
                        dists = [x for y in list(map.census_block) for x in y]
                        assert (
                            len(dists) == len(set(dists))
                            and set(dists) == all_blocks_in_state
                            and sum(list(map.N_WINNERS_PER_DISTRICT)) == num_seats_in_state
                        )  # no block repeated, all blocks are covered, right number of winners
                        yield maphash, map, map_characteristics  #  just yield from here without checking if its been done
                        # print(enmap, maps_per_district_num_setting, maps_per_district_num_setting)
                        if maps_per_district_num_setting is not None and enmap >= maps_per_district_num_setting - 1:
                            break

            del optimized_plans_overallfile


def create_single_district_in_state_files(example_map_for_state, block_assignments, district_df):
    # get all census blocks for the state -- by just opening up the "2" file, grabbing the first map, and getting all the block associated.
    all_blocks = []
    num_seats = 0
    for node_id in example_map_for_state:
        # do the blockassignment and district_df method thing to get the actual map, district df, etc.
        district_blocks = block_assignments[node_id]
        all_blocks.extend([census_tract_string(x) for x in district_blocks])
        num_seats += district_df.loc[node_id].n_seats
    assert len(set(all_blocks)) == len(all_blocks)  # should not repeat any block
    return set(all_blocks), num_seats


def from_sample_OH_maps(filename="data/multi_member_district_sample_plans.p", state="OH", maps_per_district_num=None):
    districts = pickle.load(open(filename, "rb"))

    all_census_blocks = [x for y in districts[2][0] for x in y]
    # print(all_census_blocks[0:10])
    state_num = "{:011d}".format(all_census_blocks[0])[0:2]
    # print(len(all_census_blocks))
    districts[1] = [[all_census_blocks]]

    for num_districts in sorted(districts):
        num_winners = int(16 / num_districts)
        if maps_per_district_num is None:
            maps_per_district_numloc = len(districts[num_districts])
        else:
            maps_per_district_numloc = maps_per_district_num
        for enmap, map in enumerate(districts[num_districts][0:maps_per_district_numloc]):
            d = []
            for en, district in enumerate(map):
                d.append(
                    {
                        "state": state,
                        "state_num": state_num,
                        "census_block": [census_tract_string(x) for x in district],
                        "district": str(en),
                        "N_WINNERS_PER_DISTRICT": num_winners,
                    }
                )
            # map_label = "{}".format(num_districts)
            map = pd.DataFrame(d)
            maphash = get_map_hash_string(map)
            map_characteristics = {"state": state, "state_num": state_num, "N_districts": num_districts}
            yield maphash, map, map_characteristics
            # if enmap > 10:
            # break


map_generators = {x.__name__: x for x in [from_sample_OH_maps, from_optimization_and_sampling_runs_per_state]}
