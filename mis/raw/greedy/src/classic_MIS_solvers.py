import random
import cplex
import networkx as nx

from typing import List, Dict, Tuple


def solve_weighted_mis(graph: nx.Graph) -> List[int]:
    """
    Solve the MIS problem and return a single optimal solution.

    :param graph: A NetworkX graph where each node has a "weight" attribute.
    :return: A list of nodes representing the optimal solution.
    """
    if not graph.nodes:
        return []

    # Initialize weight for unweighted graph
    for node in graph.nodes():
        if "weight" not in graph.nodes[node]:
            raise ValueError(f"Node {node} is missing the 'weight' attribute.")

    c = cplex.Cplex()

    # Relabel the graph with consecutive integers
    solved_graph = nx.convert_node_labels_to_integers(graph)
    weights = list(nx.get_node_attributes(graph, "weight").values())

    # Setting variables as binary
    c.variables.add(
        names=[str(node) for node in solved_graph.nodes()],
        types=[c.variables.type.binary] * len(solved_graph.nodes())
    )

    # Independence constraints
    c.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(ind=[str(u), str(v)], val=[1.0, 1.0]) 
            for u, v in solved_graph.edges()
        ],
        senses=["L"] * solved_graph.number_of_edges(),
        rhs=[1.0] * solved_graph.number_of_edges(),
    )

    # Objective function definition
    c.objective.set_linear([(str(node), weights[node]) for node in solved_graph.nodes()])
    c.objective.set_sense(c.objective.sense.maximize)

    # Solve MIP without logs
    c.set_log_stream(None)
    c.set_results_stream(None)
    c.set_warning_stream(None)
    c.solve()

    # Extract solution
    solution_values = c.solution.get_values()
    selected_nodes = [
        node for node, value in enumerate(solution_values) if value >= 0.9
    ]

    # Convert back to original node labels
    conversion_table = list(graph.nodes())
    return [conversion_table[node] for node in selected_nodes]


def weighted_greedy_independent_set(graph: nx.Graph) -> List[int]:
    """
    Approximate greedy algorithm for the Weighted Independent Set problem on a NetworkX graph.

    :param graph: A NetworkX graph where each node has a "weight" attribute.
    :return: A list of integers representing the approximate maximum weight independent set.
    """
    if not all("weight" in graph.nodes[u] for u in graph.nodes):
        raise ValueError("All nodes must have a 'weight' attribute.")

    # Sort nodes by weight/degree descending
    nodes = sorted(
        graph.nodes,
        key=lambda u: graph.nodes[u]["weight"] / (1 + graph.degree[u]),
        reverse=True,
    )

    independent_set = set()
    visited = set()

    for u in nodes:
        if u not in visited:
            # Add node to the independent set
            independent_set.add(u)
            # Mark the node and its neighbors as visited
            visited.add(u)
            visited.update(graph.neighbors(u))

    return list(independent_set)


def weighted_generate_diff_greedy_mis(graph: nx.Graph, priority: dict = None) -> List[int]:
    """
    Generate a maximal independent set using a greedy approach.

    :param graph: The input graph.
    :param priority: Priority values for nodes. Defaults to all zeros.
    :return: A list of integers representing the independent set.
    """
    if priority is None:
        priority = {node: 0 for node in graph.nodes}
    # Sort nodes by priority and weight/degree ratio
    sorted_nodes = sorted(
        graph.nodes,
        key=lambda x: (
            (priority[x] + graph.degree[x]) / graph.nodes[x]["weight"],
            random.random() + 0.5,
        ),
    )
    mis = set()
    for node in sorted_nodes:
        if all(neigh not in mis for neigh in graph.neighbors(node)):
            mis.add(node)
    return list(mis)


def weighted_generate_different_mis(graph: nx.Graph, lattice_id_coord_dic: Dict[int, Tuple[float, float]], k: int) -> List[List[int]]:
    """
    Generate `k` maximal independent sets with maximal differences.

    :param graph: The input graph.
    :param k: The number of independent sets to generate.
    :return: A list of lists, where each sublist contains the nodes of an independent set.
    """
    priority = {node: 0 for node in graph.nodes}
    mis_sets: List[List[int]] = []

    for _ in range(k):
        # Generate a new MIS using current priority
        mis = weighted_generate_diff_greedy_mis(graph, priority)
        mis_sets.append(mis)

        # Update priorities to discourage reuse of current MIS nodes
        for node in graph.nodes:
            if node in mis:
                priority[node] += graph.size() / 100

    return sorted(mis_sets, key=len, reverse=True)