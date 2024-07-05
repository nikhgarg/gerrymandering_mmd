from visualization.helpers import *
from visualization.fancy_plots import *


def plot_all_for_one_state(df, state):
    boxplot_per_district_for_single_state_per_method(df, state, target_col="fraction_winners_Democrat", do_extremes_and_prop_line=True)

    print("polarization")
    boxplot_per_district_for_single_state_per_method(df, state, target_col="polarization")

    for cohesion in ["geographic", "partisan_score", "racial", "education", "income"]:
        print(cohesion)
        boxplot_per_district_for_single_state_per_method(df, state, target_col="cohesion_{}".format(cohesion))
        for party in ["Democrat", "Republican"]:
            print(cohesion, party)
            boxplot_per_district_for_single_state_per_method(df, state, target_col="cohesion_{}_{}".format(cohesion, party))

