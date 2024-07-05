import os
import copy
import time
import json
import pickle
import networkx as nx
import argparse
from collections import OrderedDict
from load import load_opt_data
from center_selection import *
from partition import *
from multimember_tree import MultiMemberSHPNode

class DefaultCostFunction:
    def __init__(self, lengths):
        self.lengths = lengths

    def get_costs(self, area_df, centers):
        population = area_df.population.values
        index = list(area_df.index)
        costs = self.lengths[np.ix_(centers, index)] * population
        costs **= (1 + random.random())
        return {center: {index[bix]: cost for bix, cost in enumerate(costs[cix])}
                for cix, center in enumerate(centers)}


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


class ColumnGenerator:
    """
    Generates columns with the Stochastic Hierarchical Paritioning algorithm.
    Maintains samples tree and logging data.
    """
    def __init__(self, config):
        """
        Initialized with configuration dict
        Args:
            config: (dict) the following are the required keys
                state: (str) 2 letter abbreviation
                n_districts: (int)
                population_tolerance: (float) ideal population +/- factor epsilon
                max_sample_tries: (int) number of attempts at each node
                n_samples: (int) the fan-out split width
                n_root_samples: (int) the split width of the root node w
                max_n_splits: (int) max split size z_max
                min_n_splits: (int) min split size z_min
                max_split_population_difference: (float) maximum
                    capacity difference between 2 sibling nodes
                event_logging: (bool) log events for visualization
                verbose: (bool) print runtime information
                selection_method: (str) seed selection method to use
                perturbation_scale: (float) pareto distribution parameter
                n_random_seeds: (int) number of fixed seeds in seed selection
                capacities: (str) style of capacity matching/computing
                capacity_weights: (str) 'voronoi' or 'fractional'
                IP_gap_tol: (float) partition IP gap tolerance
                IP_timeout: (float) maximum seconds to spend solving IP

        """
        state_abbrev = config['state']
        state_df, G, lengths, edge_dists = load_opt_data(state_abbrev)
        lengths /= 1000

        self.state_abbrev = state_abbrev

        ideal_pop = state_df.population.values.sum() / config['n_seats']
        max_pop_variation = ideal_pop * config['population_tolerance']

        config['max_pop_variation'] = max_pop_variation
        config['ideal_pop'] = ideal_pop

        self.config = config
        self.G = G
        self.state_df = state_df
        self.lengths = lengths
        self.edge_dists = edge_dists

        self.sample_queue = []
        self.internal_nodes = {}
        self.leaf_nodes = {}
        self.max_id = 0
        self.root = None

        self.failed_regions = []
        self.failed_root_samples = 0
        self.n_infeasible_partitions = 0
        self.n_successful_partitions = 0

        self.cost_fn = DefaultCostFunction(lengths)

    def _assign_id(self):
        self.max_id += 1
        return self.max_id

    def retry_sample(self, problem_node, sample_internal_nodes, sample_leaf_nodes):
        def get_descendents(node_id):
            if self.config['verbose']:
                print('executing get_descendents()')
            direct_descendents = [child for partition in
                                  sample_internal_nodes[node_id].children_ids
                                  for child in partition]
            indirect_descendants = [get_descendents(child) for child in direct_descendents
                                    if (child in sample_internal_nodes)]
            return flatten(direct_descendents + indirect_descendants)

        if problem_node.parent_id == 0:
            raise RuntimeError('Root partition region not subdivisible')
        parent = sample_internal_nodes[problem_node.parent_id]

        parent.infeasible_children += 1
        if parent.infeasible_children > self.config['parent_resample_trials']:
            # Failure couldn't be corrected
            if self.config['verbose']:
                print(problem_node.area)
            raise RuntimeError('Unable to sample tree')

        branch_ix, branch_ids = parent.get_branch(problem_node.id)
        nodes_to_delete = set()
        for node_id in branch_ids:
            nodes_to_delete.add(node_id)
            if node_id in sample_internal_nodes:
                for child_id in get_descendents(node_id):
                    nodes_to_delete.add(child_id)

        for node_id in nodes_to_delete:
            if node_id in sample_leaf_nodes:
                del sample_leaf_nodes[node_id]
            elif node_id in sample_internal_nodes:
                del sample_internal_nodes[node_id]

        parent.delete_branch(branch_ix)
        self.sample_queue = [parent] + [n for n in self.sample_queue if n.id not in nodes_to_delete]
        if self.config['verbose']:
            print(f'Pruned branch from {parent.id} with {len(nodes_to_delete)}'
                  f' deleted descendents.')

    def generate(self):
        """
        Main method for running the generation process.

        Returns: None

        """
        completed_root_samples = 0
        n_root_samples = self.config['n_root_samples']

        root = MultiMemberSHPNode(self.config['n_seats'],
                                    self.config['n_districts'],
                                    list(self.state_df.index),
                                    0, is_root=True)
        self.root = root
        self.internal_nodes[root.id] = root

        while completed_root_samples < n_root_samples:
            # For each root partition, we attempt to populate the sample tree
            # If failure in particular root, prune all work from that root
            # partition. If successful, commit subtree to whole tree.
            self.sample_queue = [root]
            sample_leaf_nodes = {}
            sample_internal_nodes = {}
            try:
                if self.config['verbose']:
                    print('Root sample number', completed_root_samples)
                while len(self.sample_queue) > 0:
                    node = self.sample_queue.pop()
                    child_samples = self.sample_node(node)
                    if len(child_samples) == 0:  # Failure detected
                        self.failed_regions.append(node.area)
                        # Try to correct failure
                        self.retry_sample(node, sample_internal_nodes, sample_leaf_nodes)
                        continue
                    for child in child_samples:
                        if child.n_districts == 1:
                            sample_leaf_nodes[child.id] = child
                        else:
                            self.sample_queue.append(child)
                    if not node.is_root:
                        sample_internal_nodes[node.id] = node
                self.internal_nodes.update(sample_internal_nodes)
                self.leaf_nodes.update(sample_leaf_nodes)
                completed_root_samples += 1
            except RuntimeError:  # Stop trying on root partition
                print('Root sample failed')
                self.root.children_ids = self.root.children_ids[:-1]
                self.root.partition_times = self.root.partition_times[:-1]
                self.failed_root_samples += 1

    def sample_node(self, node):
        """
        Generate children partitions of a region contained by [node].

        Args:
            node: (SHPnode) Node to be samples

        Returns: A flattened list of child regions from multiple partitions.

        """
        area_df = self.state_df.loc[node.area]
        samples = []
        n_trials = 0
        n_samples = 1 if node.is_root else self.config['n_samples']
        if not isinstance(n_samples, int):
            n_samples = int((n_samples // 1) + (random.random() < n_samples % 1))
        while len(samples) < n_samples and n_trials < self.config['max_sample_tries']:
            partition_start_t = time.time()
            child_nodes = self.make_partition(area_df, node)
            partition_end_t = time.time()
            if child_nodes:
                self.n_successful_partitions += 1
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
                node.partition_times.append(partition_end_t - partition_start_t)
            else:
                self.n_infeasible_partitions += 1
                node.n_infeasible_samples += 1
            n_trials += 1

        return [node for sample in samples for node in sample]

    def make_partition(self, area_df, node):
        """
        Using a random seed, attempt one split from a sample tree node.
        Args:
            area_df: (DataFrame) Subset of rows of state_df for the node region
            node: (SHPnode) the node to sample from

        Returns: (list) of shape nodes for each sub-region in the partition.

        """
        children_sizes_and_seats = node.sample_multimember_splits_and_child_sizes(self.config)

        # dict : {center_ix : (child size, child seats)}
        children_centers = OrderedDict(self.select_centers(area_df, children_sizes_and_seats))

        if not node.is_root:
            G = nx.subgraph(self.G, node.area)
            edge_dists = {center: nx.shortest_path_length(G, source=center)
                          for center in children_centers}
        else:
            G = self.G
            edge_dists = {center: self.edge_dists[center] for center in children_centers}

        pop_bounds = self.make_pop_bounds(children_centers)
        costs = self.cost_fn.get_costs(area_df, list(children_centers.keys()))
        connectivity_sets = edge_distance_connectivity_sets(edge_dists, G)

        partition_IP, xs = make_partition_IP(costs,
                                             connectivity_sets,
                                             area_df.population.to_dict(),
                                             pop_bounds)

        partition_IP.Params.MIPGap = self.config['IP_gap_tol']
        partition_IP.update()
        partition_IP.optimize()
        try:
            districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                           for i in children_centers}
            feasible = all([nx.is_connected(nx.subgraph(self.G, distr)) for
                            distr in districting.values()])
            if not feasible:
                print('WARNING: PARTITION NOT CONNECTED')
        except AttributeError:
            feasible = False

        if self.config['verbose']:
            if feasible:
                print('successful sample')
            else:
                print('infeasible')

        if feasible:
            return [MultiMemberSHPNode(pop_bounds[center]['n_seats'],
                    pop_bounds[center]['n_districts'],
                    area, self._assign_id(), node.id)
                    for center, area in districting.items()]
        else:
            node.n_infeasible_samples += 1
            return []

    def select_centers(self, area_df, children_sizes):
        """
        Routes arguments to the right seed selection function.
        Args:
            area_df: (DataFrame) Subset of rows of state_df of the node region
            children_sizes: (int list) Capacity of the child regions
        Returns: (dict) {center index: # districts assigned to that center}

        """
        centers = iterative_random(area_df, len(children_sizes), self.lengths)

        center_capacities = get_capacities(centers, children_sizes,
                                           area_df, self.config)

        return center_capacities

    def make_pop_bounds(self, children_centers):
        """
        Finds the upper and lower population bounds of a dict of center sizes
        Args:
            children_centers: (dict) {center index: # districts}

        Returns: (dict) center index keys and upper/lower population bounds
            and # districts as values in nested dict

        """
        pop_deviation = self.config['max_pop_variation']
        pop_bounds = {}
        # Make the bounds for an area considering # area districts and tree level
        for center, (n_districts, n_seats) in children_centers.items():
            levels_to_leaf = max(math.ceil(math.log2(n_districts)), 1)
            distr_pop = self.config['ideal_pop'] * n_seats

            ub = distr_pop + pop_deviation / levels_to_leaf
            lb = distr_pop - pop_deviation / levels_to_leaf

            pop_bounds[center] = {
                'ub': ub,
                'lb': lb,
                'n_seats': n_seats,
                'n_districts': n_districts
            }

        return pop_bounds


def run_generation(base_config, states, experiment_dir, skip_existing=True):
    SAVE_DIR = "optimization_results"

    save_path = os.path.join(SAVE_DIR, experiment_dir)
    os.makedirs(save_path, exist_ok=True)
    existing_trials = set(os.listdir(save_path))

    for state in states:
        n_seats = seat_map[state]
        if base_config['use_HR_4000_rules']:
            if n_seats < 6:
                continue
            valid_district_numbers = [math.ceil(n_seats / 5)]
        else:
            valid_district_numbers = list(range(2, 10)) + list(range(10, 21, 2)) + list(range(23, 53, 3))
            valid_district_numbers = list(filter(lambda x: x < seat_map[state], valid_district_numbers))
            valid_district_numbers.append(seat_map[state])
        for n_districts in valid_district_numbers:
            trial = f'{state}_{n_districts}.p'
            if skip_existing and trial in existing_trials:
                print(f'Skipping trial {trial}, already ran')
                continue
            print(f'Starting {state} with {n_districts} districts')
            trial_config = copy.deepcopy(base_config)
            trial_config['state'] = state
            trial_config['n_districts'] = n_districts
            trial_config['n_seats'] = n_seats
            trial_config['n_root_samples'] = int(round((1000 / n_districts) ** 1.2))
            trial_config['n_samples'] = int(round((300 / n_districts) ** 0.5))

            cg = ColumnGenerator(trial_config)
            start_t = time.time()
            cg.generate()
            runtime = time.time() - start_t
            print(f'{state} with {n_districts} districts finished in {runtime / 60} mins')

            pickle.dump({
                'generation_time': runtime,
                'leaf_nodes': cg.leaf_nodes,
                'internal_nodes': cg.internal_nodes,
            }, open(os.path.join(save_path, trial), 'wb'))


def create_trials(n_threads, seat_map):
    state_weights = {}
    for state in seat_map:
        valid_district_numbers = list(range(2, 10)) + list(range(10, 21, 2)) + list(range(23, 53, 3))
        valid_district_numbers = list(filter(lambda x: x < seat_map[state], valid_district_numbers))
        valid_district_numbers.append(seat_map[state])
        state_weights[state] = seat_map[state] * len(valid_district_numbers)
    trial_lists = [[] for _ in range(n_threads)]
    trial_weights = np.zeros(n_threads)
    for state, weight in sorted(state_weights.items(), key=lambda x: x[1], reverse=True):
        trial_ix = np.argmin(trial_weights)
        trial_weights[trial_ix] += weight
        trial_lists[trial_ix].append(state)
    return [l[::-1] for l in trial_lists]


if __name__ == '__main__':
    seat_map = {
        'AL': 7,
        'AZ': 9,
        'AR': 4,
        'CA': 53,
        'CO': 7,
        'CT': 5,
        'FL': 27,
        'GA': 14,
        'HI': 2,
        'ID': 2,
        'IL': 18,
        'IN': 9,
        'IA': 4,
        'KS': 4,
        'KY': 6,
        'LA': 6,
        'ME': 2,
        'MD': 8,
        'MA': 9,
        'MI': 14,
        'MN': 8,
        'MS': 4,
        'MO': 8,
        'NE': 3,
        'NV': 4,
        'NH': 2,
        'NJ': 12,
        'NM': 3,
        'NY': 27,
        'NC': 13,
        'OH': 16,
        'OK': 5,
        'OR': 5,
        'PA': 18,
        'RI': 2,
        'SC': 7,
        'TN': 9,
        'TX': 36,
        'UT': 4,
        'VA': 11,
        'WA': 10,
        'WV': 3,
        'WI': 8
    }
    tree_config = {
	'capacity_weights': 'voronoi',
        'max_sample_tries': 25,
        'parent_resample_trials': 3,
        'n_samples': 3,
        'n_root_samples': 10,
        'max_n_splits': 4,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'verbose': False,
    }
    gurobi_config = {
        'IP_gap_tol': 1e-4,
        'IP_timeout': 10,
    }
    multimember_pdp_config = {
        'population_tolerance': .01,
        'use_HR_4000_rules': False
    }
    base_config = {**tree_config,
                   **gurobi_config,
                   **multimember_pdp_config}
    EXPERIMENT_DIR = "multimember_generation"
    HR4000_EXPERIMENT_DIR = "hr4000_multimember_generation"

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default=0, type=int)
    parser.add_argument('-n', '--n_trials', default=4, type=int)
    parser.add_argument('-hr', '--use_hr_4000_rules', default=False, const=True, type=bool, nargs='?')
    args = parser.parse_args()

    trials = create_trials(args.n_trials, seat_map)
    trial_ix = args.index

    if args.use_hr_4000_rules:
        base_config['use_HR_4000_rules'] = True
        run_generation(base_config, trials[trial_ix], HR4000_EXPERIMENT_DIR)
    else:
        run_generation(base_config, trials[trial_ix], EXPERIMENT_DIR)
