import os
import pickle
import random
import copy
from tree_dp import *


def get_node_info(leaf_nodes, internal_nodes):
    """
    Preprocessing function to annotate SHP tree nodes with their parent id
     and solution counts.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).

    Returns: ((dict) solution_count, (dict) parent-nodes)
        solution_count - {(SHPNode.id) id: (int) number of feasible partitions of region}
        parent_nodes - {(SHPNode.id): (SHPNode.id) of parent}

    """
    solution_count = {}
    parent_nodes = {}
    nodes = {**leaf_nodes, **internal_nodes}
    root = internal_nodes[0]

    def recursive_compute(current_node, all_nodes):
        if not current_node.children_ids:
            return 1

        total_districtings = 0
        for sample in current_node.children_ids:
            sample_districtings = 1
            for child_id in sample:
                child_node = nodes[child_id]
                parent_nodes[child_node.id] = current_node.id
                sample_districtings *= recursive_compute(child_node, all_nodes)

            total_districtings += sample_districtings
        solution_count[current_node.id] = total_districtings
        return total_districtings

    recursive_compute(root, nodes)

    return solution_count, parent_nodes


def prune_sample_space(internal_nodes,
                       solution_count,
                       parent_nodes,
                       target_size=1000):
    """
    Prune the sample tree to make enumeration tractable.

    WARNING: this function has side effects. It mutates the internal_nodes
    member variables in place. Care should be taken to use a copy.

    Args:
        internal_nodes: (SHPNode list) with node capacity >1 (has child nodes).
        solution_count: (dict) {(SHPNode.id) node: (int) size of feasible partition set}
        parent_nodes: (dict) {(SHPNode.id) node: (SHPNode.id) parent node id}
        target_size: (int) the ideal size to prune the feasible space

    Returns: (SHPNode list) internal nodes with pruned partition samples

    """
    def recompute_node_size(node):
        new_node_size = 0
        for sample in node.children_ids:
            sample_districtings = 1
            for child_id in sample:
                sample_districtings *= solution_count.get(child_id, 1)
            new_node_size += sample_districtings
        return new_node_size

    root = internal_nodes[0]
    assert root.is_root
    assert target_size > 0

    nodes_by_size = {}
    for node in internal_nodes.values():
        try:
            nodes_by_size[int(node.n_districts)].append(node)
        except KeyError:
            nodes_by_size[int(node.n_districts)] = [node]

    current_node_prune_size = 2
    for size, node_list in nodes_by_size.items():
        random.shuffle(node_list)
    while solution_count[root.id] > target_size:
        n_skinny_nodes = 0
        node_list = nodes_by_size.get(current_node_prune_size, [])
        for node in node_list:
            if solution_count[root.id] <= target_size:
                break
            if len(node.children_ids) > 1:
                node.children_ids = node.children_ids[:-1]
                solution_count[node.id] = recompute_node_size(node)
                parent_id = parent_nodes.get(node.id, None)
                while parent_id is not None:
                    parent_node = internal_nodes[parent_id]
                    solution_count[parent_id] = recompute_node_size(parent_node)
                    parent_id = parent_nodes.get(parent_id, None)
            else:
                n_skinny_nodes += 1
        if n_skinny_nodes == len(nodes_by_size.get(current_node_prune_size, [])):
            current_node_prune_size += 1
    return internal_nodes


if __name__ == '__main__':
    EXPERIMENT_DIR = os.path.join('optimization_results', 'multimember_generation')
    SUBSAMPLE_SAVE_PATH = os.path.join(EXPERIMENT_DIR, 'subsample')
    os.makedirs(SUBSAMPLE_SAVE_PATH, exist_ok=True)
    TARGET_SIZE = 1000

    for file in sorted(os.listdir(EXPERIMENT_DIR)):
        tree = pickle.load(open(os.path.join(EXPERIMENT_DIR, file), 'rb'))
        solution_count, parent_nodes = get_node_info(tree['leaf_nodes'], tree['internal_nodes'])
        pruned_interal_nodes = prune_sample_space(tree['internal_nodes'],
                                                    solution_count,
                                                    parent_nodes,
                                                    target_size=TARGET_SIZE)

        partitions = enumerate_partitions(tree['leaf_nodes'], pruned_internal_nodes)

        save_file = os.path.join(SUBSAMPLE_SAVE_PATH, f'{file[:-2]}_subsample_{TARGET_SIZE}.p')
        pickle.dump(partitions, open(save_file, 'wb'))
        print(f'Finished subsampling {file}')
        
