import random
import networkx as nx
import typing

from mis.pipeline.postprocessor import BasePostprocessor
from mis.shared.types import MISSolution, Weighting
from mis.shared.graphs import BaseWeightPicker

if typing.TYPE_CHECKING:
    from mis import SolverConfig


class Maximization(BasePostprocessor):
    """
    A postprocessor dedicated to improving MIS results provided by a quantum algorithm.

    This postprocessor expects that a result could be vulnerable to bitflips, so it
    will attempt to fix any result provided by the quantum algorithm, to make it
    independent (if it's not independent) and maximal (if it's not maximal).
    """

    def __init__(
        self,
        config: "SolverConfig",
        frequency_threshold: float = 1e-7,
        augment_rounds: int = 10,
        seed: int = 0,
    ):
        """
        frequency_threshold: Minimal frequency to check. Discard any solution which show
            up with a frequency <= frequency_threshold. Set 0 to never discard any solution.
        augment_rounds: The number of attempts to augment an independent set to
            add possibly missing nodes.
        seed: A random seed.
        """
        self.frequency_threshold = frequency_threshold
        self.augment_rounds = augment_rounds
        self.seed = seed
        self.picker = BaseWeightPicker.for_weighting(config.weighting)
        self.weighting = config.weighting

    def postprocess(self, solution: MISSolution) -> MISSolution | None:
        """
        The main entry point: attempt to improve a solution.
        """
        if solution.frequency < self.frequency_threshold:
            return None
        if not self.is_independent_solution(solution):
            solution = self.reduce_to_independence(solution)
        # From this point, we're sure that `solution` is independent.
        if nx.is_dominating_set(solution.instance.graph, solution.nodes):
            # Maximum independent set, we can't do better.
            return solution
        return self.augment_to_maximal(solution)

    def is_independent_solution(self, solution: MISSolution) -> bool:
        """
        Check whether a solution is independent.
        """
        return self._is_independent_list(graph=solution.instance.graph, nodes=solution.node_indices)

    def _is_independent_list(self, graph: nx.Graph, nodes: list[int]) -> bool:
        """
        Check whether a list of nodes within a graph is independent.

        A list is dependent if there exist two nodes in `nodes` that are
        connected by edges in `graph`, independent otherwise.

        Attributes:
            graph: The graph to which the nodes belong.
            nodes: A list of nodes in graph.
        """
        for i, u in enumerate(nodes):
            assert graph.has_node(u), f"Node {u} does not belong to the graph"
            for v in nodes[i:]:
                if graph.has_edge(u, v):
                    return False
        return True

    def augment_to_maximal(self, solution: MISSolution) -> MISSolution:
        """Augment a given set up to a maximal IS using a greedy algorithm running k times.

        See https://doi.org/10.48550/arXiv.2202.09372 section 2.3 of supplementary material for reference.
        """
        unpicked_nodes: set[int] = set(solution.instance.graph.nodes) - set(solution.node_indices)

        # The best solution so far.
        best_pick = solution.node_indices
        best_weight = self.picker.subgraph_weight(solution.instance.graph, best_pick)

        rng = random.Random(self.seed)
        for _ in range(self.augment_rounds):
            # Pick an arbitrary and random order.
            order = list(unpicked_nodes)
            rng.shuffle(order)

            # Attempt to grow the list of nodes in this order.
            picked = list(solution.node_indices)
            weight = self.picker.subgraph_weight(solution.instance.graph, picked)
            for node in order:
                maybe_picked = list(picked)  # Copy the list.
                maybe_picked.append(node)
                if self._is_independent_list(graph=solution.instance.graph, nodes=maybe_picked):
                    # Commit our pick.
                    picked = maybe_picked
                    weight += self.picker.node_weight(solution.instance.graph, node)

            # Once we have picked as many nodes as possible, time to check whether
            # we have improved on the best solution.
            match self.weighting:
                case Weighting.WEIGHTED:
                    if weight > best_weight:
                        best_pick = picked
                case Weighting.UNWEIGHTED:
                    if len(picked) > len(best_pick):
                        best_pick = picked
        return MISSolution(
            instance=solution.instance,
            frequency=solution.frequency,
            nodes=best_pick,
        )

    def reduce_to_independence(self, solution: MISSolution) -> MISSolution:
        """Reduce the given candidate solution to an independent state of graph g.

        We progressively remove the nodes with highest number of neighbours.

        See https://doi.org/10.48550/arXiv.2202.09372 section 2.3 of supplementary material for reference.
        """

        # Simplify the graph by removing the nodes that we have already rejected.
        simplified_graph = solution.instance.graph.copy()
        rejected_nodes = set(simplified_graph.nodes) - set(solution.node_indices)
        simplified_graph.remove_nodes_from(rejected_nodes)

        while True:
            # Pick the remaining node causing the largest number of conflicts.
            ranked_nodes = [
                (node, len(list(simplified_graph.neighbors(node))))
                for node in simplified_graph.nodes
            ]
            highest_node = max(ranked_nodes, key=lambda tup: tup[1])

            # Remove this node from the graph. Eventually, `simplified_graph` will become an independent
            # set (at worst, a singleton).
            simplified_graph.remove_node(highest_node[0])
            retained_nodes = set(simplified_graph.nodes)

            candidate = list(retained_nodes)
            if self._is_independent_list(graph=solution.instance.graph, nodes=candidate):
                # As the empty set is independent and `candidates` keeps decreasing,
                # this will eventually be `True`.
                return MISSolution(
                    instance=solution.instance, nodes=candidate, frequency=solution.frequency
                )
