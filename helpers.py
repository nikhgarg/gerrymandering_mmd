from functools import lru_cache
from scipy.spatial.distance import euclidean
import hashlib

import pandas as pd
import glob
import re
import os
import ast


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
    
    
    import pandas as pd
import ast



#2025/09/06: Functions to combine per state output files into one

def transform_data(df):
    # {'optimization': 'unfair', 'optimization_voting_method_for': 'thiele_pav', 'party_optimized_for': 'd_opt_solutions', 'number': 0}
    
    
    # Parse optimization from optimization_characteristics
    # df['optimization'] = df['optimization_characteristics'].apply(
        # extract_optimization
    # )
    
    if 'optimization_characteristics' in df.columns:
        df['optimization_characteristics'] = df['optimization_characteristics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
        df['optimization'] = df.optimization_characteristics.apply(lambda x: x.get('optimization', None) if isinstance(x, dict) else None)    
        df['optimization_voting_method_for'] = df.optimization_characteristics.apply(lambda x: x.get('optimization_voting_method_for', None) if isinstance(x, dict) else None)
        df['party_optimized_for'] = df.optimization_characteristics.apply(lambda x: x.get('party_optimized_for', None) if isinstance(x, dict) else None)
        df['optimization_number'] = df.optimization_characteristics.apply(lambda x: x.get('number', None) if isinstance(x, dict) else None)
        
    if 'noise_parameters' in df.columns:
        df['noise_parameters'] = df['noise_parameters'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
        df['noise_parameters_loc'] = df.noise_parameters.apply(lambda x: x.get('loc', None) if isinstance(x, dict) else None)
        df['noise_parameters_scale'] = df.noise_parameters.apply(lambda x: x.get('scale', None) if isinstance(x, dict) else None)
    
    # Define column order for output
    columns = [
        # added 1/3/2026 so I can get which maps were used for the outputs
        "overall_hash", "map_hash", "settings_hash",

        'state', 'state_num', 'N_districts', 'N_VOTERS', 'total_winners', 'optimization', 'optimization_voting_method_for',
        'party_optimized_for', 'optimization_number',
        'ranking_method', 'DISTANCE_FUNCTION', 'noise_distribution',  'noise_parameters_loc', 'noise_parameters_scale',
        'fraction_voters_Republican', 'fraction_winners_Republican', 'medians',
        'cohesion_income_Republican', 'cohesion_income_Democrat', 'cohesion_income',
        'cohesion_partisan_score_Republican', 'cohesion_partisan_score_Democrat', 'cohesion_partisan_score',
        'cohesion_education_Republican', 'cohesion_education_Democrat', 'cohesion_education',
        'cohesion_racial_Republican', 'cohesion_racial_Democrat', 'cohesion_racial',
        'cohesion_geographic_Republican', 'cohesion_geographic_Democrat', 'cohesion_geographic',
        'winner_location_variance', 'winner_location_variance_Democrat', 'winner_location_variance_Republican'
    ]
    
    return df[columns]

def process_csv_files_to_combined_into_organized_df(folder_path, regex_pattern, output_filename):
    """
    Process CSV files matching regex pattern, transform and combine them
    """
    
    # Get all CSV files in folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter files that match the regex pattern
    pattern = re.compile(regex_pattern)
    matching_files = [f for f in csv_files if pattern.match(os.path.basename(f))]
    
    if not matching_files:
        print(f"No files found matching pattern: {regex_pattern}")
        return
    
    print(f"Found {len(matching_files)} files matching pattern:")
    for f in matching_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and transform each CSV
    transformed_dfs = []
    
    for file_path in matching_files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)
            transformed_df = transform_data(df)
            transformed_dfs.append(transformed_df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not transformed_dfs:
        print("No files were successfully processed")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(transformed_dfs, ignore_index=True)
    
    # Save combined dataframe
    output_path = os.path.join(folder_path, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined {len(transformed_dfs)} files into {output_filename}")
    print(f"Final dataset shape: {combined_df.shape}")
    print(f"Saved to: {output_path}")
    
    return combined_df