import itertools


def enumerate_partitions(leaf_nodes, internal_nodes):
    """
    Enumerate all feasible plans stored in the sample tree.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).

    Returns: A list of lists, each inner list is a plan comprised of leaf node ids.

    """
    def feasible_partitions(node, node_dict):
        if not node.children_ids:
            return [[node.id]]

        partitions = []
        for disjoint_sibling_set in node.children_ids:
            sibling_partitions = []
            for child in disjoint_sibling_set:
                sibling_partitions.append(feasible_partitions(node_dict[child],
                                                              node_dict))
            combinations = [list(itertools.chain.from_iterable(combo))
                            for combo in itertools.product(*sibling_partitions)]
            partitions.append(combinations)

        return list(itertools.chain.from_iterable(partitions))

    root = internal_nodes[0] 

    node_dict = {**internal_nodes, **leaf_nodes}
    return feasible_partitions(root, node_dict)


def query_tree(leaf_nodes, internal_nodes, query_vals):
    """
    Dynamic programming method to find plan which maximizes linear district metric.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        query_vals: (list) of metric values per node.

    Returns: (float, list) tuple of optimal objective value and optimal plan.

    """
    nodes = {**leaf_nodes,  **internal_nodes}
    id_to_ix = {nid: ix for ix, nid in enumerate(sorted(leaf_nodes))}
    root = internal_nodes[0]

    def recursive_query(current_node, all_nodes):
        if not current_node.children_ids:
            return query_vals[id_to_ix[current_node.id]], [current_node.id]

        node_opts = []
        for sample in current_node.children_ids:  # Node partition
            sample_value = 0
            sample_opt_nodes = []
            for child_id in sample:  # partition slice
                child_node = nodes[child_id]
                child_value, child_opt = recursive_query(child_node, all_nodes)
                sample_value += child_value
                sample_opt_nodes += child_opt

            node_opts.append((sample_value, sample_opt_nodes))

        return max(node_opts, key=lambda x: x[0])

    return recursive_query(root, nodes)


def query_per_root_partition(leaf_nodes, internal_nodes, query_vals):
    """
    Dynamic programming method to find plan per root partition which
    maximizes linear district metric.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        query_vals: (list) of metric values per node.

    Returns: (float list, list list) tuple of optimal objective values and optimal plans 
        per root partition.

    """
    nodes = {**leaf_nodes,  **internal_nodes}
    id_to_ix = {nid: ix for ix, nid in enumerate(sorted(leaf_nodes))}
    root = internal_nodes[0]

    def recursive_query(current_node, all_nodes):
        if not current_node.children_ids:
            return query_vals[id_to_ix[current_node.id]], [current_node.id]

        node_opts = []
        partitions = current_node.children_ids
        if current_node.is_root:
            partitions = [partitions[ROOT_PARTITION_IX]]
        for sample in partitions:  # Node partition
            sample_value = 0
            sample_opt_nodes = []
            for child_id in sample:  # partition slice
                child_node = nodes[child_id]
                child_value, child_opt = recursive_query(child_node, all_nodes)
                sample_value += child_value
                sample_opt_nodes += child_opt

            node_opts.append((sample_value, sample_opt_nodes))

        return max(node_opts, key=lambda x: x[0])

    partition_optimal_plans = []
    partition_optimal_values = []
    for ROOT_PARTITION_IX in range(0, len(root.children_ids)):
        val, plan = recursive_query(root, nodes)
        partition_optimal_values.append(val)
        partition_optimal_plans.append(plan)
    return partition_optimal_values, partition_optimal_plans
