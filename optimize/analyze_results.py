import os
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from optimize_pipeline import calculate_statewide_average_voteshare_with_third_party


state_constants = {
    'AL': {'seats': 7.0, 'vote_share': 0.6215384217483797},
    'AZ': {'seats': 9.0, 'vote_share': 0.5364263103733232},
    'AR': {'seats': 4.0, 'vote_share': 0.6350502951885596},
    'CA': {'seats': 53.0, 'vote_share': 0.36578066640411316},
    'CO': {'seats': 7.0, 'vote_share': 0.4616697689981887},
    'CT': {'seats': 5.0, 'vote_share': 0.43131609530673226},
    'FL': {'seats': 27.0, 'vote_share': 0.5084152944383389},
    'GA': {'seats': 14.0, 'vote_share': 0.5330968221576192},
    'HI': {'seats': 2.0, 'vote_share': 0.29844262480182354},
    'ID': {'seats': 2.0, 'vote_share': 0.6563684808818163},
    'IL': {'seats': 18.0, 'vote_share': 0.4032430762708615},
    'IN': {'seats': 9.0, 'vote_share': 0.5474116519867084},
    'IA': {'seats': 4.0, 'vote_share': 0.523383722294008},
    'KS': {'seats': 4.0, 'vote_share': 0.5864816538959026},
    'KY': {'seats': 6.0, 'vote_share': 0.6079743160596287},
    'LA': {'seats': 6.0, 'vote_share': 0.5983888671453019},
    'ME': {'seats': 2.0, 'vote_share': 0.4470798411663454},
    'MD': {'seats': 8.0, 'vote_share': 0.4269951897578011},
    'MA': {'seats': 9.0, 'vote_share': 0.3914116163285649},
    'MI': {'seats': 14.0, 'vote_share': 0.46476791596032424},
    'MN': {'seats': 8.0, 'vote_share': 0.4345866950659021},
    'MS': {'seats': 4.0, 'vote_share': 0.5727615669220875},
    'MO': {'seats': 8.0, 'vote_share': 0.5465957163292497},
    'NE': {'seats': 3.0, 'vote_share': 0.6019848867487783},
    'NV': {'seats': 4.0, 'vote_share': 0.4737504546780456},
    'NH': {'seats': 2.0, 'vote_share': 0.4948957974136881},
    'NJ': {'seats': 12.0, 'vote_share': 0.42567604232241085},
    'NM': {'seats': 3.0, 'vote_share': 0.4110475369013984},
    'NY': {'seats': 27.0, 'vote_share': 0.3655206996034086},
    'NC': {'seats': 13.0, 'vote_share': 0.5133038460762559},
    'OH': {'seats': 16.0, 'vote_share': 0.5297035023501533},
    'OK': {'seats': 5.0, 'vote_share': 0.6594055295797322},
    'OR': {'seats': 5.0, 'vote_share': 0.43135756204151393},
    'PA': {'seats': 18.0, 'vote_share': 0.4802225119359719},
    'RI': {'seats': 2.0, 'vote_share': 0.38736112150622964},
    'SC': {'seats': 7.0, 'vote_share': 0.5668331601897997},
    'TN': {'seats': 9.0, 'vote_share': 0.6102607750599166},
    'TX': {'seats': 36.0, 'vote_share': 0.5855894267837248},
    'UT': {'seats': 4.0, 'vote_share': 0.6955438144954558},
    'VA': {'seats': 11.0, 'vote_share': 0.46011305472360925},
    'WA': {'seats': 10.0, 'vote_share': 0.40777087091586245},
    'WV': {'seats': 3.0, 'vote_share': 0.6412395040289991},
    'WI': {'seats': 8.0, 'vote_share': 0.4926652977917956}
 }

def load_trial(state, k, experiment_dir,  districts_dir, scores_dir, opt_results_dir, load_block_assignments):
    if int(k) < 0: # HR4000
        k = math.ceil(state_constants[state]['seats'] / 5)
    trial = f'{state}_{k}'
    block_assignment_path = os.path.join(experiment_dir, 'block_assignments', f'{trial}_block_assignment.p')
    district_dfs_path = os.path.join(experiment_dir, districts_dir, f'{trial}_district_df.csv')
    district_scores_path = os.path.join(experiment_dir, scores_dir, f'{trial}_score_df.csv')
    optimization_results_path = os.path.join(experiment_dir, opt_results_dir, f'{trial}_opt_results.p')
    subsampled_plans_path = os.path.join(experiment_dir, 'subsampled_plans', f'{trial}_plans.p')

    return {
        'block_assignment': pickle.load(open(block_assignment_path, 'rb')) if load_block_assignments else None,
        'district_df': pd.read_csv(district_dfs_path, index_col='node_id'),
        'district_scores': pd.read_csv(district_scores_path, index_col='node_id'),
        'optimization_results': pickle.load(open(optimization_results_path, 'rb')),
        'subsampled_plans': pickle.load(open(subsampled_plans_path, 'rb')),
    }


def load_state(state, experiment_dir='default', districts_dir="district_dfs",
               scores_dir="district_scores", opt_results_dir="optimization_results",
               load_block_assignments=False):
    if experiment_dir == 'default':
        here = os.path.dirname(os.path.abspath(__file__))
        # TODO (hwr26): changed the default location for third party analysis
        experiment_dir = os.path.join(here, 'optimization_results', 'third_party')
    trials = [file[:-len('_opt_results.p')]
              for file in os.listdir(os.path.join(experiment_dir, opt_results_dir))
              if file[-2:] == '.p' and file[:2] == state]
    state_results = {}
    for trial in trials:
        k = trial.split('_')[1]
        state_results[trial] = load_trial(state, k, experiment_dir, districts_dir, scores_dir, opt_results_dir, load_block_assignments)
    return state_results


def grid_plot_all_voting_rules_extreme():
    rows = 9
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(18, 22))
    for ix, state in enumerate(list(state_constants.keys())):
        results = load_state(state, load_block_assignments=False)
        seats = state_constants[state]['seats']
        extremes = {}
        for trial in results:
            k = int(trial.split('_')[1])
            for voting_rule in results[trial]['optimization_results']['unfair']:
                extremes[k, voting_rule] = {
                    'D': seats - max(results[trial]['optimization_results']['unfair'][voting_rule]['d_opt_vals']),
                    'R': max(results[trial]['optimization_results']['unfair'][voting_rule]['r_opt_vals'])
                }

        ax = axs[ix // cols, ix % cols]
        ax.annotate(state, (0.02, .85), xycoords='axes fraction', size=14)

        df = pd.DataFrame(extremes).T.sort_index().unstack() / seats
        df['D']['independent'].plot(color='blue', linestyle='-', label='D_advantage_thiele_independent', ax=ax)
        df['D']['thiele_pav'].plot(color='blue', linestyle=':', label='D_advantage_thiele_pav', ax=ax)
        df['D']['thiele_squared'].plot(color='blue', linestyle='--', label='D_advantage_thiele_squared', ax=ax)

        ax.axhline(y=state_constants[state]['vote_share'], color='black', lw=1, label='proportional')

        df['R']['independent'].plot(color='red', linestyle='-', label='R_advantage_thiele_independent', ax=ax)
        df['R']['thiele_pav'].plot(color='red', linestyle=':', label='R_advantage_thiele_pav', ax=ax)
        df['R']['thiele_squared'].plot(color='red', linestyle='--', label='R_advantage_thiele_squared', ax=ax)

    axs[-1, -1].remove()
    axs[-1, -2].remove()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=2, bbox_to_anchor=(0.825, 0.06))


def state_seat_share_distributions(rule):
    distributions = {}
    for state in state_constants:
        n_seats = state_constants[state]['seats']
        state_distribution = {}
        results = load_state(state)
        for trial_key, trial_results in results.items():
            score_series = trial_results['district_scores'][rule]
            score_series = score_series.reindex(range(0, max(score_series.index)+1)).fillna(0).values
            distribution = list(score_series[np.array(trial_results['subsampled_plans'])].sum(axis=1))
            # Add dem and rep tails
            distribution.append(n_seats - max(pd.Series(trial_results['optimization_results']['unfair'][rule]['d_opt_vals'])))
            distribution.append(max(pd.Series(trial_results['optimization_results']['unfair'][rule]['r_opt_vals'])))
            summary_statistics = pd.Series(distribution).describe().iloc[3:] # get min, 25%,..., max
            state_distribution[int(trial_key.split('_')[1])] = summary_statistics / n_seats

        state_distribution_df = pd.DataFrame(state_distribution).sort_index(axis=1)
        distributions[state] = state_distribution_df
    return distributions


def competitiveness_distributions():
    competitiveness = {}
    for state in state_constants:
        state_competitiveness = {}
        for trial in results:
            state, k = trial.split('_')
            district_df = results[trial]['district_df']
            n_seats = district_df['n_seats'].values
            mean_vote_share = district_df['mean'].values
            pav_winner_interval = 1 / (n_seats + 1)
            n_rep_winners = np.floor(mean_vote_share / pav_winner_interval)
            # The losing share is the minimum number of vote share needed to win an additional seat
            losing_share = np.minimum(pav_winner_interval * (n_rep_winners+1) - mean_vote_share,
                                    mean_vote_share - pav_winner_interval * n_rep_winners)

            # Reindex by leaf node id
            losing_share_series = pd.Series(losing_share, index=district_df.index)
            losing_share_series = losing_share_series.reindex(range(0, max(losing_share_series.index)+1)).values


            # wasted_vote_by_district_by_plan is (plan x K) matrix of losing shares
            # TODO: summary statistic of wasted_votes in a plan
            wasted_vote_by_district_by_plan = losing_share_series[np.array(results[trial]['subsampled_plans'])]
            state_competitiveness[int(k)] = wasted_vote_by_district_by_plan
        competitiveness[state] = state_competitiveness
    return competitiveness


def grid_plot_seat_range_box_plot(distributions):
    rows = 9
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(18, 22))
    for ix, state in enumerate(list(distributions.keys())):
        ax = axs[ix // cols, ix % cols]
        ax.annotate(state, (0.02, .85), xycoords='axes fraction', size=14)
        ax.axhline(y=state_constants[state]['vote_share'], color='black', lw=1, label='proportional', ls=':')
        state_distribution_df = distributions[state]
        state_distribution_df.boxplot(whis=(0, 100), ax=ax)
    axs[-1, -1].remove()
    axs[-1, -2].remove()


def plot_all_state_distribution(distributions):
    sample = np.arange(0, 1.001, 0.001)
    full_distribution = np.zeros((len(sample), 5))
    total_seats = 0
    for state in distributions:
        seat_fraction = distributions[state].columns / state_constants[state]['seats']
        state_stats = distributions[state].values.T
        state_distr = state_stats[np.argmin(np.abs(np.subtract.outer(sample, seat_fraction)), axis=1)]
        full_distribution += state_distr * state_constants[state]['seats']
        total_seats += state_constants[state]['seats']

    ax = (pd.DataFrame(full_distribution, columns=distributions['AL'].index, index=sample) / total_seats).plot()
    ax.set_ylabel('R seat share')
    ax.set_xlabel('districts / seats')


def unfairness_by_rule():
    result_by_state_by_rule_k = {}
    for ix, state in enumerate(list(state_constants.keys())):
        results = load_state(state, load_block_assignments=False)
        seats = state_constants[state]['seats']
        extremes = {}
        for trial in results:
            k = int(trial.split('_')[1])
            for voting_rule in results[trial]['optimization_results']['unfair']:
                extremes[k, voting_rule] = {
                    'D': max(results[trial]['optimization_results']['unfair'][voting_rule]['d_opt_vals']),
                    'R': max(results[trial]['optimization_results']['unfair'][voting_rule]['r_opt_vals'])
                }
        result_by_state_by_rule_k[state] = extremes
    return result_by_state_by_rule_k


def result_by_state_by_rule_k(states, districts_dir, scores_dir, opt_results_dir):
    result_by_state_by_rule_k = {}
    for state in states:
        results = load_state(state, districts_dir=districts_dir,
                             scores_dir=scores_dir, opt_results_dir=opt_results_dir,
                             load_block_assignments=False)
        seats = state_constants[state]['seats']
        extremes = {}
        for trial in results:
            k = int(trial.split('_')[1])
            for voting_rule in results[trial]['optimization_results']['unfair']:
                extremes[k, voting_rule] = {
                    'D': max(results[trial]['optimization_results']['unfair'][voting_rule]['d_opt_vals']),
                    'R': max(results[trial]['optimization_results']['unfair'][voting_rule]['r_opt_vals']),
                    'T': max(results[trial]['optimization_results']['unfair'][voting_rule]['t_opt_vals'])
                }
            for voting_rule in results[trial]['optimization_results']['fair']:
                extremes[k, voting_rule]['fair'] = \
                    min([v['objective_value'] for v in results[trial]['optimization_results']['fair'][voting_rule].values()])
        result_by_state_by_rule_k[state] = extremes
    return result_by_state_by_rule_k


def partisan_gerrymander_advantage(states, result_by_state_by_rule_k, voting_rule, ddf_save_path):
    # matrix for party (state x seat_fraction)
    # states = [state for state in state_constants if state_constants[state]['seats'] > 1]
    state_sizes = np.array([state_constants[state]['seats'] for state in states])
    sample = np.arange(0, 1.001, 0.001)
    d_advantage_by_state = np.zeros((len(states), len(sample)))
    r_advantage_by_state = np.zeros((len(states), len(sample)))
    t_advantage_by_state = np.zeros((len(states), len(sample)))
    fair_by_state = np.zeros((len(states), len(sample)))
    for ix, state in enumerate(states):
        state_opt = sorted([(k, v) for (k, rule), v in result_by_state_by_rule_k[state].items()
                            if rule == voting_rule],
                           key=lambda x: x[0])
        n_state_seats = state_constants[state]['seats']
        proportional = calculate_statewide_average_voteshare_with_third_party(state, ddf_save_path)
        seat_fraction = np.array([k for k, _ in state_opt]) / n_state_seats
        r_opt = (np.array([v['R'] for _, v in state_opt]) / n_state_seats) - proportional[0]
        d_opt = (np.array([v['D'] for _, v in state_opt]) / n_state_seats) - proportional[1]
        t_opt = (np.array([v['T'] for _, v in state_opt]) / n_state_seats) - proportional[2]
        fair = (np.array([v['fair'] for _, v in state_opt]) / n_state_seats) / 3
        d_advantage_by_state[ix, :] = d_opt[np.argmin(np.abs(np.subtract.outer(sample, seat_fraction)), axis=1)]
        r_advantage_by_state[ix, :] = r_opt[np.argmin(np.abs(np.subtract.outer(sample, seat_fraction)), axis=1)]
        t_advantage_by_state[ix, :] = t_opt[np.argmin(np.abs(np.subtract.outer(sample, seat_fraction)), axis=1)]
        fair_by_state[ix, :] = fair[np.argmin(np.abs(np.subtract.outer(sample, seat_fraction)), axis=1)]
    average_d_advantage = np.average(d_advantage_by_state, weights=state_sizes, axis=0)
    average_r_advantage = np.average(r_advantage_by_state, weights=state_sizes, axis=0)
    average_t_advantage = np.average(t_advantage_by_state, weights=state_sizes, axis=0)
    average_fairness = np.average(fair_by_state, weights=state_sizes, axis=0)
    return average_d_advantage, average_r_advantage, average_t_advantage, average_fairness


def plot_partisan_advantage_by_rule(states, result_by_state_by_rule_k, ddf_save_path):
    voting_rules = ['thiele_pav', 'thiele_squared', 'thiele_independent']
    rule_advantage = {}
    for rule in voting_rules:
        average_d_advantage, average_r_advantage, average_t_advantage, average_fairness = partisan_gerrymander_advantage(states, result_by_state_by_rule_k, rule, ddf_save_path)
        rule_advantage[rule] = {
            'D': average_d_advantage,
            'R': average_r_advantage,
            'T': average_t_advantage,
            'fair': average_fairness
        }

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    domain = sample = np.arange(0, 1.001, 0.001)
    for ix, rule in enumerate(voting_rules):
        ax = axs[ix]
        r_advantage = rule_advantage[rule]['R']
        d_advantage = rule_advantage[rule]['D']
        t_advantage = rule_advantage[rule]['T']
        fair = rule_advantage[rule]['fair']
        ax.plot(domain, d_advantage, color='blue', lw=2, label='D Max Gerrymander')
        ax.plot(domain, r_advantage, color='red', lw=2, label='R Max Gerrymander')
        ax.plot(domain, t_advantage, color='green', lw=2, label='T Max Gerrymander')
        ax.plot(domain, fair, color='black', ls=':', lw=1, label='Fairness')
        ax.hlines(0, 0, 1, color='black')
        ax.set_ylabel('proportionality gap')
        ax.set_xlabel('districts / seats')
        ax.legend()
        ax.set_title(rule)


def get_hr4000_distributions(experiment_dir, min_k=10):
    # Load HR4000 results into a dataframe of seat distribution by state and rule
    # experiment_dir = os.path.join('optimization_results', 'hr4000_multimember_generation')
    distributions = {}
    for state in state_constants:
        if state_constants[state]['seats'] < min_k:
            continue
        trial_results = load_trial(state, -1, experiment_dir, False)
        n_seats = state_constants[state]['seats']
        for rule in trial_results['district_scores'].columns:
            score_series = trial_results['district_scores'][rule]
            score_series = score_series.reindex(range(0, max(score_series.index)+1)).fillna(0).values
            distribution = [score_series[plan].sum() for plan in trial_results['subsampled_plans']]
            # Add dem and rep tails
            distribution.append(n_seats - max(pd.Series(trial_results['optimization_results']['unfair'][rule]['d_opt_vals'])))
            distribution.append(max(pd.Series(trial_results['optimization_results']['unfair'][rule]['r_opt_vals'])))
            summary_statistics = pd.Series(distribution).describe().iloc[3:] # get min, 25%,..., max
            distributions[(state, rule)] = summary_statistics / n_seats
    distr_df = pd.DataFrame(distributions)
    partisan_ranking = [
        state for state, _ in
        sorted([(state, state_constants[state]['vote_share'])
            for state in list(set(distr_df.columns.get_level_values(0)))],
        key=lambda x: x[1])
    ]
    distr_df =  distr_df[partisan_ranking].unstack().reset_index()
    distr_df.columns = ['state', 'rule', 'percentile', 'seats']
    distr_df.rule = distr_df.rule.replace({
        'thiele_pav': 'STV and PAV',
        'thiele_independent': 'Winner take all',
        'thiele_squared': 'Thiele squared'
    })
    return distr_df


def create_hr4000_mmd_baseline_distr_df(baseline_distributions, hr_4000_distribution_df):
    # Combine HR4000 results with all 3 member and all 5 member (or as close as possible)
    # baseline_distributions = state_seat_share_distributions('thiele_pav')
    # hr_4000_distribution_df = get_hr4000_distributions(experiment_dir)

    def compute_hr4000_bounds(group):
        group.loc[(group.rule == 'HR4000') & (group.percentile == 'min'), 'seats'] = group.seats.min()
        group.loc[(group.rule == 'HR4000') & (group.percentile == 'max'), 'seats'] = group.seats.max()
        return group

    def baseline_sizes(k):
        return (math.floor(k / 3), math.ceil(k / 5))

    hr4000_pav = hr_4000_distribution_df.query("rule == 'STV and PAV'")
    dfs = [hr4000_pav]
    states = list(hr4000_pav.state.unique())
    for state in states:
        ub, lb = baseline_sizes(state_constants[state]['seats'])
        while ub not in baseline_distributions[state]:
            ub -= 1
        while lb not in baseline_distributions[state]:
            lb += 1
        ub_df = pd.DataFrame({'state': state, 'rule': 'lb', 'seats': baseline_distributions[state][lb]}).reset_index()
        lb_df = pd.DataFrame({'state': state, 'rule': 'ub', 'seats': baseline_distributions[state][ub]}).reset_index()
        dfs.append(lb_df.rename(columns={'index': 'percentile'})[['state', 'rule', 'percentile', 'seats']])
        dfs.append(ub_df.rename(columns={'index': 'percentile'})[['state', 'rule', 'percentile', 'seats']])

    baseline_df = pd.concat(dfs)
    baseline_df = baseline_df.replace({
        'rule': {'STV and PAV': 'HR4000', 'ub': "Three-member districts", 'lb': 'Five-member districts'}
    })
    return baseline_df.groupby('state', sort=False).apply(compute_hr4000_bounds)

def create_hr4000_smd_baseline_distr_df(baseline_distributions, hr_4000_distribution_df):
    # Combine HR4000 results with SMDs
    # baseline_distributions = state_seat_share_distributions('thiele_pav')
    # hr_4000_distribution_df = get_hr4000_distributions(experiment_dir)

    hr4000_pav = hr_4000_distribution_df.query("rule == 'STV and PAV'")
    dfs = [hr4000_pav]
    states = list(hr4000_pav.state.unique())
    for state in states:
        n_seats = state_constants[state]['seats']
        smd_df = pd.DataFrame({
            'state': state,
            'rule': 'smd',
            'seats': baseline_distributions[state][n_seats]
        }).reset_index()
        dfs.append(smd_df.rename(columns={'index': 'percentile'})[['state', 'rule', 'percentile', 'seats']])

    baseline_df = pd.concat(dfs)
    baseline_df = baseline_df.replace({
        'rule': {'STV and PAV': 'HR4000', 'smd': 'Single member districts'}
    })
    return baseline_df

def plot_hr4000(distr_df, savepath):
    # distr_df is the result of either
    #   get_hr4000_distributions() for HR4000 boxplot by voting rule
    #   create_hr4000_baseline_distr_df() for boxplot by choice of number and size of districts
    fig, ax = plt.subplots(figsize=(15, 4))
    pal = sns.color_palette([
        sns.color_palette("RdYlBu", 10)[0],
        sns.color_palette("PRGn", 10)[0 - 3],
        sns.color_palette("RdYlBu", 10)[-1],
    ])
    sns.boxplot(x=distr_df['state'], y=distr_df['seats'], hue=distr_df['rule'],
                ax=ax, whis=(0, 100), fliersize=0, palette=pal,
                saturation=0.75, linewidth=1.1)
    sns.despine()

    # Add markers for proportionality
    states = []
    [states.append(x) for x in distr_df.state.values if x not in states]
    n_states = len(states)
    for ix, state in enumerate(states):
        xmin = ix / n_states + .005
        xmax = xmin + (1 / n_states) - .01
        vs = state_constants[state]['vote_share']
        ax.axhline(vs, xmin=xmin, xmax=xmax, lw=1.25, ls=':', color='black')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], label='Proportional', color='black', ls=':'))
    ax.legend(title=None, handles=handles, fontsize=14, frameon=False)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel('Republican seat share', fontsize=12)
    fig.savefig(savepath, format='pdf', bbox_inches='tight')
