import abc
from dataclasses import dataclass
from enum import Enum
import typing
from typing import Any

import networkx as nx
from networkx.classes.reportviews import DegreeView
from mis.pipeline.preprocessor import BasePreprocessor
from mis.shared.graphs import is_independent, BaseWeightPicker, closed_neighborhood
from mis.shared.types import Objective

if typing.TYPE_CHECKING:
    from mis.pipeline.config import SolverConfig

class _TwinCategory(str, Enum):
    InSolution = "IN-SOLUTION"
    """
    The twin nodes are necessarily part of the solution.
    """
    Independent = "INDEPENDENT"
    CannotRemove = "CANNOT-REMOVE"
    """
    The rule does not work for these twin nodes.
    """

@dataclass
class _Twin:
    node: int
    category: _TwinCategory
    neighbours: list[int]

class Kernelization(BasePreprocessor):
    def __init__(self, config: "SolverConfig", graph: nx.Graph) -> None:
        if config.objective == Objective.MAXIMIZE_SIZE:
            self._kernelizer = UnweightedKernelization(config, graph)
        elif config.objective == Objective.MAXIMIZE_WEIGHT:
            self._kernelizer = WeightedKernelization(config, graph)
        else:
            raise ValueError(f"invalid objective {config.objective}")

    def preprocess(self) -> nx.Graph:
        return self._kernelizer.preprocess()

    def rebuild(self, partial_solution: set[int]) -> set[int]:
        return self._kernelizer.rebuild(partial_solution)


class BaseKernelization(BasePreprocessor, abc.ABC):
    """
    Shared base class for kernelization.
    """

    def __init__(self, config: "SolverConfig", graph: nx.Graph) -> None:
        self.cost_picker = BaseWeightPicker.for_objective(config.objective)

        # The latest version of the graph.
        # We rewrite it progressively to decrease the number of
        # nodes and edges.
        self.kernel: nx.Graph = graph.copy()
        self.initial_number_of_nodes = self.kernel.number_of_nodes()
        self.rule_application_sequence: list[BaseRebuilder] = []
        self.config = config
        self.is_weighted_mode = config.objective == Objective.MAXIMIZE_WEIGHT

        # An index used to generate new node numbers.
        self._new_node_gen_counter: int = 1
        if self.initial_number_of_nodes > 0:
            self._new_node_gen_counter = max(self.kernel.nodes()) + 1

        # Get rid of any node with a self-loop (a node that is its own
        # neighbour), as it cannot be part of a solution and we rely upon
        # their absence in rule applications.
        for node in list(self.kernel.nodes()):
            if self.kernel.has_edge(node, node):
                self.kernel.remove_node(node)


    def rebuild(self, partial_solution: set[int]) -> set[int]:
        """
        Rebuild a MIS solution to the original graph from
        a partial MIS solution on the reduced graph obtained
        by kernelization.
        """
        partial_solution = set(partial_solution)
        for rule_app in reversed(self.rule_application_sequence):
            rule_app.rebuild(partial_solution)
        return partial_solution

    def is_independent(self, nodes: list[int]) -> bool:
        """
        Determine if a set of nodes represents an independent set
        within a given graph.

        Returns:
            True if the nodes in `nodes` represent an independent
                set within `graph`.
            False otherwise, i.e. if there's at least one connection
                between two nodes of `nodes`
        """
        return is_independent(self.kernel, nodes)

    def is_subclique(self, nodes: list[int]) -> bool:
        """
        Determine whether a list of nodes represents a clique
        within the graph, i.e. whether every pair of nodes is connected.
        """
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if not self.kernel.has_edge(u, v):
                    return False
        return True

    def node_weight(self, node: int) -> float:
        return self.cost_picker.node_weight(self.kernel, node)

    @abc.abstractmethod
    def is_maximum(self, node: int, neighbours: list[int]) -> bool:
        """
        Determine whether any neighbour of a node has a weight strictly
        greater than that node. 
        """
        ...

    def is_isolated_and_maximum(self, node: int) -> bool:
        """
        Determine whether a node is isolated and maximum, i.e.
        1. this node + its neighbours represent a clique; AND
        2. no node in the neighborhood has a weight strictly greater than `node`.
        """
        neighborhood = closed_neighborhood(self.kernel, node)
        if not self.is_subclique(nodes=neighborhood):
            return False
        return self.is_maximum(node, neighborhood)

    @abc.abstractmethod
    def add_node(self, weight: float) -> int:
        """
        Add a new node with a unique index and weight.
        """
        ...

    def preprocess(self) -> nx.Graph:
        """
        Apply all rules, exhaustively, until the graph cannot be reduced
        further, storing the rules for rebuilding after the fact.
        """
        # Invariant: from this point, `self.kernel` does not contain any
        # self-loop.
        while (kernel_size_start := self.kernel.number_of_nodes()) > 0:
            self.search_rule_neighborhood_removal()
            self.search_rule_isolated_node_removal() # TODO: In the original, the weighted kernelizer has essentially two copies of this rule, once with weight and once without. Double-check.
            self.search_rule_twin_reduction()
            self.search_rule_node_fold()
            self.search_rule_unconfined_and_diamond()

            kernel_size_end: int = self.kernel.number_of_nodes()
            assert kernel_size_end <= kernel_size_start # Just in case.
            if kernel_size_start == kernel_size_end:
                # We didn't find any rule to apply, time to stop.
                break
        return self.kernel

    # -----------------neighborhood_removal---------------------------
    @abc.abstractmethod
    def search_rule_neighborhood_removal(self) -> None:
        ...

    # -----------------isolated_node_removal---------------------------
    @abc.abstractmethod
    def get_nodes_with_strictly_higher_weight(self, node: int, neighborhood: list[int]) -> list[int]:
        ...

    def apply_rule_isolated_node_removal(self, isolated: int) -> None:
        neighborhood = list(self.kernel.neighbors(isolated))
        higher = self.get_nodes_with_strictly_higher_weight(isolated, neighborhood)
        rule_app = RebuilderIsolatedNodeRemoval(isolated, higher)
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_nodes_from(neighborhood)
        self.kernel.remove_node(isolated)


    def search_rule_isolated_node_removal(self) -> None:
        """
        Remove any isolated node (see `is_isolated` for a definition).
        """
        for node in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(node):
                # This might be possible if our iterator has not
                # been invalidated but our operation caused the node to
                # disappear from `self.kernel`.
                continue

            if self.is_isolated_and_maximum(node):
                self.apply_rule_isolated_node_removal(node)


    # -----------------node_fold---------------------------

    def fold_three(self, v: int, u: int, x: int, v_prime: int) -> None:
        """
        Fold three nodes V, U and X into a new single node V'.
        """
        neighbors_v_prime = set(self.kernel.neighbors(u)) | set(self.kernel.neighbors(x))
        for node in neighbors_v_prime:
            self.kernel.add_edge(v_prime, node)
        self.kernel.remove_nodes_from([v, u, x])

    def apply_rule_node_fold(self, v: Any, w_v: float, u: Any, w_u: float, x: Any, w_x: float) -> None:
        v_prime = self.add_node(w_u + w_x - w_v)
        rule_app = RebuilderNodeFolding(v, u, x, v_prime) # FIXME: Adapt?
        self.rule_application_sequence.append(rule_app)
        self.fold_three(v, u, x, v_prime)

    def search_rule_node_fold(self) -> None:
        """
        If a node V has exactly two neighbours U and X and there is no edge
        between U and X, fold U, V and X and into a single node.
        """
        if self.kernel.number_of_nodes() == 0:
            return
        assert isinstance(self.kernel.degree, DegreeView)
        for v in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(v):
                # This might be possible if our iterator has not
                # been invalidated but our operation caused `v` to
                # disappear from `self.kernel`.
                continue
            if self.kernel.degree(v) != 2:
                continue
            [u, x] = self.kernel.neighbors(v)
            if self.kernel.has_edge(u, x):
                continue
            w_u = self.node_weight(u)
            w_v = self.node_weight(v)
            w_x = self.node_weight(x)
            if w_v >= w_u + w_x:
                # Always false in unweighted mode.
                # Cannot fold.
                continue
            if w_v < w_u:
                # Always false in unweighted mode.
                # Cannot fold.
                continue
            if w_v < w_x:
                # Always false in unweighted mode.
                # Cannot fold.
                continue
            self.apply_rule_node_fold(
                v=v, w_v=w_v,
                u=u, w_u=w_u,
                x=x, w_x=w_x)

    # -----------------twin reduction---------------------------

    @abc.abstractmethod
    def twin_category(self, u: int, v: int, neighbours: list[int]) -> _TwinCategory:
        """
        Arguments:
            - u, v: two distinct nodes with the same set of neighbours
            - neighbours: the neighbours of u
        """
        ...

    def find_twin(self, v: int) -> _Twin | None:
        """
        Find a twin of a node, i.e. another node with the same
        neighbours.
        """
        neighbors_v: set[int] = set(self.kernel.neighbors(v)) # FIXME: We could factorize this.

        for u in self.kernel.nodes():
            if u == v:
                continue
            neighbors_u: set[int] = set(self.kernel.neighbors(u))
            if neighbors_u == neighbors_v:
                # Note: Since there are no self-loops, we can deduce
                # that U and V are also not neighbours.
                category= self.twin_category(u, v, list(neighbors_u))
                if category == _TwinCategory.CannotRemove:
                    continue
                return _Twin(
                    node=int(u),
                    neighbours=list(neighbors_u),
                    category=category)
        return None

    def fold_twin(self, u: int, v: int, v_prime: int, u_neighbours: list[int]) -> None:
        neighborhood_u_neighbours: list[int] = list(
            set().union(*[set(self.kernel.neighbors(node)) for node in u_neighbours])
        )
        for neigh in neighborhood_u_neighbours:
            self.kernel.add_edge(v_prime, neigh)
        self.kernel.remove_nodes_from([u, v])
        self.kernel.remove_nodes_from(u_neighbours)

    def apply_rule_twins_in_solution(self, v: int, u: int, neighbors_u: list[int]) -> None:
        rule_app = RebuilderTwinAlwaysInSolution(v, u)
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_nodes_from(neighbors_u)
        self.kernel.remove_nodes_from([u, v])

    def apply_rule_twin_independent(self, u: int, v: int, neighbours: list[int]) -> None:
        """
        Arguments:
            - u, v: two distinct nodes with the same set of neighbours
            - neighbours: the neighbours of u (which are also the neighbours of v)
        """
        w_u = self.node_weight(self.kernel.nodes[u])
        w_v = self.node_weight(self.kernel.nodes[v])
        w_u_neighbours_sum = self.cost_picker.subgraph_weight(self.kernel, neighbours)
        v_prime = self.add_node(w_u_neighbours_sum - (w_u + w_v))
        rule_app_B = RebuilderTwinIndependent(v, u, neighbours, v_prime)
        self.rule_application_sequence.append(rule_app_B)
        self.fold_twin(v_prime, u, v, neighbours)

    def search_rule_twin_reduction(self) -> None:
        """
        If a node V and a node U have the exact same neighbours
        (which indicates that they're not nightbours themselves),
        we can merge U, V and their neighborhoods.
        """
        if self.kernel.number_of_nodes() == 0:
            return
        assert isinstance(self.kernel.degree, DegreeView)
        for v in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(v):
                continue
            twin: _Twin | None = self.find_twin(v)
            if twin is None:
                continue
            u = twin.node
            category = self.twin_category(u, v, twin.neighbours)
            if category == _TwinCategory.Independent:
                # FIXME: The definition is very different between MIS and wMIS.
                self.apply_rule_twin_independent(u, v, twin.neighbours)
            elif category == _TwinCategory.InSolution:
                self.apply_rule_twins_in_solution(u, v, twin.neighbours)
            else:
                # We cannot remove this twin.
                pass

    # -----------------unconfined reduction---------------------------

    @abc.abstractmethod
    def search_rule_unconfined_and_diamond(self) -> None:
        ...


class UnweightedKernelization(BaseKernelization):
    def add_node(self, weight: float) -> int:
        assert weight == 1.0
        node = self._new_node_gen_counter
        self._new_node_gen_counter += 1
        self.kernel.add_node(node)
        return node

    def is_maximum(self, node: int, neighbours: list[int]) -> bool:
        """
        Since all nodes have the same weight, no node has a strictly higher weight.        
        """
        return True

    # -----------------isolated node removal--------------------
    def get_nodes_with_strictly_higher_weight(self, node: int, neighborhood: list[int]) -> list[int]:
        """
        Since all nodes have the same weight, no node has a strictly higher weight.
        """
        return []

    # -----------------neighborhood removal---------------------

    def search_rule_neighborhood_removal(self) -> None:
        # This rule is a noop in unweighted mode.
        return

    # -----------------twin reduction---------------------------

    def twin_category(self, u: int, v: int, neighbours: list[int]) -> _TwinCategory:
        if len(neighbours) != 3:
            # The heuristic only works with exactly 3 neighbours.
            return _TwinCategory.CannotRemove
        if self.is_independent(neighbours):
            # Either all the neighbours are part of the solution or U and V
            # are part of the solution.
            return _TwinCategory.Independent
        # Since we have exactly 3 neighbours and there is at least one dependency:
        # 1. at most 2 neighbours are part of the solution;
        # 2. at least 1 neighbour W is not part of the solution.
        #
        # Since 2., there is a solution that includes U and V.
        #
        # Since 1., a solution that includes U and V is at least as
        # good as a solution that includes some of the neighbours.
        #
        # Therefore, we can always adopt U and V as a solution.
        return _TwinCategory.InSolution

    # -----------------unconfined reduction---------------------------
    def aux_search_confinement(
        self, neighbors_S: set[int], S: set[int]
    ) -> tuple[int, int, set[int]]:
        min: int = -1
        min_value: int = self.initial_number_of_nodes + 2
        min_set_diff: set[int] = set()
        for u in neighbors_S:
            neighbors_u: set[int] = set(self.kernel.neighbors(u))
            inter: set[int] = neighbors_u & S
            if len(inter) == 1:
                if len(neighbors_u - neighbors_S - S) < min_value:
                    min = u
                    min_set_diff = neighbors_u - neighbors_S - S
                    min_value = len(min_set_diff)
        return min, min_value, min_set_diff

    def apply_rule_unconfined(self, v: int) -> None:
        rule_app = RebuilderUnconfined()
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_node(v)

    def unconfined_loop(self, v: int, S: set[int], neighbors_S: set[int]) -> bool:
        min: int = 0
        min_value: int = 0
        min_set_diff: set[int] = set()
        min, min_value, min_set_diff = self.aux_search_confinement(neighbors_S, S)
        next_loop: bool = False
        # If there is no such node, then v is confined.
        if min == -1:
            pass
            """
            if self.find_diamond_reduction(neighbors_S, S):
                self.apply_rule_diamond(v)
            """
        # If N(u)\N[S] = ∅, then v is unconfined.
        if min_value == 0:
            self.apply_rule_unconfined(v)
        # If N (u)\ N [S] is a single node w,
        # then add w to S and repeat the algorithm.
        elif min_value == 1:
            w = list(min_set_diff)[0]
            S.add(w)
            neighbors_S |= set(self.kernel.neighbors(w))
            neighbors_S -= {w}
            next_loop = True
        # Otherwise, v is confined.
        else:
            pass
        return next_loop

    def search_rule_unconfined_and_diamond(self) -> None:
        if self.kernel.number_of_nodes() == 0:
            return
        for v in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(v):
                continue
            # First, initialize S = {v}.
            S: set[int] = {v}
            neighbors_S: set[int] = set(self.kernel.neighbors(v))
            while self.unconfined_loop(v, S, neighbors_S):
                # Then find u∈N(S) such that |N(u) ∩ S| = 1
                # and |N(u)\N[S]| is minimized
                pass


class WeightedKernelization(BaseKernelization):
    """
    Apply well-known transformations to the graph to reduce its size without
    compromising the result.

    This algorithm is adapted from e.g.:
    https://schulzchristian.github.io/thesis/masterarbeit_demian_hespe.pdf

    Unless you are experimenting with your own preprocessors, you should
    probably use Kernelization in your pipeline.
    """

    def add_node(self, weight: float) -> int:
        node = self._new_node_gen_counter
        self._new_node_gen_counter += 1
        self.kernel.add_node(node)
        self.cost_picker.set_node_weight(self.kernel, node, weight)
        return node

    def is_maximum(self, node: int, neighbours: list[int]) -> bool:
        max: float = self.node_weight(self.kernel.nodes[node])
        for v in neighbours:
            if v != node and self.node_weight(self.kernel.nodes[v]) > max:
                return False
        return True

    # -----------------isolated node removal--------------------
    def get_nodes_with_strictly_higher_weight(self, node: int, neighborhood: list[int]) -> list[int]:
        pivot = self.node_weight(self.kernel.nodes[node])
        result = []
        for n in neighborhood:
            if self.node_weight(self.kernel.nodes[n]) > pivot:
                result.append(pivot)
        return result


    # -----------------unconfined reduction---------------------------

    def search_rule_unconfined_and_diamond(self) -> None:
        # This rule doesn't apply in weighted mode.
        return None

    # -----------------neighborhood_removal---------------------------
    def neighborhood_weight(self, node: int) -> float:
        return self.cost_picker.subgraph_weight(self.kernel, list(self.kernel.neighbors(node)))

    def apply_rule_neighborhood_removal(self, node: int):
        rule_app = RebuilderNeighborhoodRemoval(node)
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_nodes_from(self.kernel.neighbors(node))
        self.kernel.remove_node(node)

    def search_rule_neighborhood_removal(self) -> None:
        """
        Weighted: If a node has a greater weight than all its neighbours together,
        remove the node (it will be part of the WMIS) and all its neighbours (they
        won't).
        Unweighted: Noop.
        """

        for node in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(node):
                # This might be possible if our iterator has not
                # been invalidated but our operation caused the node to
                # disappear from `self.kernel`.
                continue
            node_weight: float = self.kernel.nodes[node]["weight"]
            neighborhood_weight_sum = self.neighborhood_weight(node)
            if node_weight >= neighborhood_weight_sum:
                self.apply_rule_neighborhood_removal(node)

    def get_lower_higher_weights(self, isolated: int, neighborhood: list[int]) -> tuple[list[int], list[int]]:
        isolated_weight: float = self.node_weight(self.kernel.nodes[isolated])
        lower: list[int] = []
        higher: list[int] = []
        for node in neighborhood:
            if node == isolated:
                continue
            if self.node_weight(self.kernel.nodes[node]) <= isolated_weight:
                lower.append(node)
            else:
                higher.append(node)
        return lower, higher






    # -----------------twin reduction---------------------------


    def twin_category(self, u: int, v: int, neighbours: list[int]) -> _TwinCategory:
        w_u: float = self.node_weight(self.kernel.nodes[u])
        w_v: float = self.node_weight(self.kernel.nodes[v])
        w_neighbours: list[float] = [self.node_weight(self.kernel.nodes[node]) for node in neighbours]
        w_neighbours_sum: float = sum(w_neighbours)
        if w_u + w_v >= w_neighbours_sum:
            # U and V are always part of the solution and the neighbours are never part of the solution.
            return _TwinCategory.InSolution
        if self.is_independent(neighbours) and w_u + w_v > w_neighbours_sum - min(w_neighbours):
            # Either a subset of the neighbours are part of the solution or U and V are part of the solution.
            return _TwinCategory.Independent
        else:
            # We don't have a nice rule to handle this case.
            return _TwinCategory.CannotRemove



class BaseRebuilder(abc.ABC):
    """
    The pre-processing operations attempt to remove edges
    and/or vertices from the original graph. Therefore,
    when we build a MIS for these reduced graphs (the
    "partial solution"), we may end up with a solution
    that does not work for the original graph.

    Each rebuilder corresponds to one of the operations
    that previously reduced the size of the graph, and is
    charged with adapting the MIS solution to the greater graph.
    """

    @abc.abstractmethod
    def rebuild(self, partial_solution: set[int]) -> None: ...

    """
    Convert a solution `partial_solution` that is valid on a reduced
    graph to a solution that is valid on the graph prior to this
    reduction step.
    """


class RebuilderIsolatedNodeRemoval(BaseRebuilder):
    def __init__(self, isolated: int, higher: list[int]):
        self.isolated = isolated
        self.higher = higher

    def rebuild(self, partial_solution: set[int]) -> None:
        if len(partial_solution & set(self.higher)) == 0:
            partial_solution.add(self.isolated)


class RebuilderNodeFolding(BaseRebuilder):
    def __init__(self, v: int, u: int, x: int, v_prime: int):
        self.v = v
        self.u = u
        self.x = x
        self.v_prime = v_prime

    def rebuild(self, partial_solution: set[int]) -> None:
        if self.v_prime in partial_solution:
            partial_solution.add(self.u)
            partial_solution.add(self.x)
            partial_solution.remove(self.v_prime)
        else:
            partial_solution.add(self.v)


class RebuilderUnconfined(BaseRebuilder):
    def rebuild(self, partial_solution: set[int]) -> None:
        pass


class RebuilderTwinIndependent(BaseRebuilder):
    def __init__(self, v: int, u: int, neighbours: list[int], v_prime: int):
        """
        Invariants:
         - V has exactly the same neighbours as U;
         - there is no self-loop around U or V (hence U and V are not
            neighbours);
         - there is no edge between any of the neighbours;
         - V' is the node obtained by merging U, V and the neighbours.
        """
        self.v: int = v
        self.u: int = u
        self.neighbours = neighbours
        self.v_prime: int = v_prime

    def rebuild(self, partial_solution: set[int]) -> None:
        if self.v_prime in partial_solution:
            # Since V' is part of the solution, none of its
            # neighbours is part of the solution. Consequently,
            # either U and V can be added to grow the solution
            # or neighbours can be added to grow the solution,
            # without affecting the rest of the system.
            partial_solution.update(self.neighbours)
            partial_solution.remove(self.v_prime)
        else:
            # The only neighbours of U and V are represented
            # by V'. Since V' is not part of the solution,
            # and since U and V are not neighbours, we can
            # always add U and V.
            partial_solution.add(self.u)
            partial_solution.add(self.v)


class RebuilderTwinAlwaysInSolution(BaseRebuilder):
    def __init__(self, v: int, u: int):
        """
        Invariants:
         - U has exactly 3 neighbours;
         - V has exactly the same neighbours as U;
         - there is no self-loop around U;
         - there is at least one connection between two neighbours of U.
        """
        self.v: int = v
        self.u: int = u

    def rebuild(self, partial_solution: set[int]) -> None:
        # Because of the invariants, U and V are always part of the solution.
        partial_solution.add(self.u)
        partial_solution.add(self.v)

class RebuilderNeighborhoodRemoval(BaseRebuilder):
    def __init__(self, dominant_vertex: int):
        self.dominant_vertex = dominant_vertex
    
    def rebuild(self, partial_solution: set[int]) -> None:
        partial_solution.add(self.dominant_vertex)
