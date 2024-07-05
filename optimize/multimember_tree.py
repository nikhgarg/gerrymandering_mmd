import math
import random
import numpy as np
from functools import lru_cache


class MultiMemberSHPNode:
    def __init__(self, n_seats, n_districts, area, id, parent_id=None, is_root=False):
        """
        SHPNodes store information needed to reconstruct the tree and gathers
        metadata from the generation process.

        Args:
            n_seats: (int) the capacity of the node
            area: (list) of block indices associated with the region
            id: (int) unique integer to identify this node
            parent_id: (int) id of parent node
            is_root: (bool) if this node is the root of the sample tree
        """
        self.is_root = is_root
        self.parent_id = parent_id
        # the number of representatives to be elected from this region.
        self.n_seats = n_seats  
        # the number of districts contained in the region.
        self.n_districts = n_districts 

        self.area_hash = hash(frozenset(area))
        self.id = id

        self.area = area
        self.children_ids = []
        self.partition_times = []

        self.n_infeasible_samples = 0
        self.infeasible_children = 0

    def get_branch(self, child_id):
        for branch_ix, branch in enumerate(self.children_ids):
            if child_id in branch:
                return branch_ix, branch
        raise ValueError(f'Node {child_id} does not exist within node {self.id}')

    def delete_branch(self, branch_ix):
        del self.children_ids[branch_ix]
        del self.partition_times[branch_ix]

    def sample_multimember_splits_and_child_sizes(self, config):
        """
        Samples split size and capacity for children with multimember districts

        Args:
            config: (dict) ColumnGenerator configuration

        Returns: (int list) of child node capacities.

        """
        if config['use_HR_4000_rules']:
            districts_to_allocate = random.choice(get_hr4000_combinations(self.n_seats))
            n_distrs = len(districts_to_allocate)
        else:
            s = int(self.n_seats // self.n_districts) # min seats / district
            b = int(self.n_seats % self.n_districts) # num s + 1 districts
            a = int(self.n_districts - b) # num s districts
            n_distrs = a + b
            districts_to_allocate = [s] * a + [(s+1)] * b

        random.shuffle(districts_to_allocate)

        split_size = random.randint(min(config['min_n_splits'], n_distrs),
                        min(config['max_n_splits'], n_distrs))

        ub = max(math.ceil(config['max_split_population_difference']
                        * n_distrs / split_size), 2)
        lb = max(math.floor((1 / config['max_split_population_difference'])
                            * n_distrs / split_size), 1)

        child_n_seats = np.array(districts_to_allocate[:split_size])
        child_n_distrs = np.ones(split_size)

        for distr_size in districts_to_allocate[split_size:]:
            ix = random.randint(0, split_size - 1)
            if child_n_distrs[ix] < ub:
                child_n_distrs[ix] += 1
                child_n_seats[ix] += distr_size

        return [(d, s) for d, s in zip(child_n_distrs, child_n_seats)]

    def __repr__(self):
        """Utility function for printing a SHPNode."""
        print_str = "Node %d \n" % self.id
        internals = self.__dict__
        for k, v in internals.items():
            if k == 'area':
                continue
            print_str += k + ': ' + v.__repr__() + '\n'
        return print_str


@lru_cache(maxsize=100)
def get_hr4000_combinations(k):
    combinations = [[[]]]
    for i in range(1, k+1):
        combos_with_duplicates = []
        for s in [3, 4, 5]:
            if i - s < 0:
                break
            for combo in combinations[i - s]:
                combos_with_duplicates.append(sorted(combo + [s]))
        combos = list(map(list, set(map(tuple, combos_with_duplicates))))
        combinations.append(combos)
    return combinations[k]