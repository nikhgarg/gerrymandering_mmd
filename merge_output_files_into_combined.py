# Merge output files into a combined dataframe
# written 2025/09/06 as part of revision for paper

from helpers import process_csv_files_to_combined_into_organized_df


# output_template = 'outputs_partisan_stv_20250906_tryrunning'

# output_template = 'outputs_partisan_stv_20250913_fewercandidates_gumbelnoise'
# output_templates = ['outputs_partisan_stv_20250913_fewercandidates_normalnoise', 'outputs_partisan_stv_20250913_fewercandidates_gumbelnoise']

# output_templates = ['outputs_20250917_fewercandidates_clean_more_gumbelnoise', 'outputs_20250917_fewercandidates_clean_more_normalnoise']


# for 1/3/2026 run so that it also saves map hashes
output_templates = [f'outputs_20250921_fewercandidates_morevoters_{noise_type}noise'.format(noise_type) for noise_type in ['normal','gumbel']]

# Combine all the per state output files into one organized dataframe and save as csv
# Code written 2025/09/06 as part of revision for paper

for output_template in output_templates:
    process_csv_files_to_combined_into_organized_df(
        "cached_values/outputs/",
        r"^{}_.*\.csv$".format(output_template),
        "organized/combined_{}_stv.csv".format(output_template)
    )