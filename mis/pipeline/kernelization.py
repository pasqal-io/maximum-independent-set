import abc
from dataclasses import dataclass
from enum import Enum
import logging
import typing
from typing import Any, Iterable

import networkx as nx
from networkx.classes.reportviews import DegreeView
from mis.pipeline.preprocessor import BasePreprocessor
from mis.shared.graphs import is_independent, BaseWeightPicker, closed_neighborhood
from mis.shared.types import Weighting

if typing.TYPE_CHECKING:
    from mis.pipeline.config import SolverConfig

logger = logging.getLogger(__name__)


class _TwinCategory(str, Enum):
    """
    A category of twins, used in twin reduction.
    """

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
class _ConfinementAux:
    node: int
    set_diff: set[int]


@dataclass
class _Twin:
    node: int
    category: _TwinCategory
    neighbors: list[int]


class Kernelization(BasePreprocessor):
    def __init__(self, config: "SolverConfig", graph: nx.Graph) -> None:
        match config.weighting:
            case Weighting.UNWEIGHTED:
                self._kernelizer: UnweightedKernelization | WeightedKernelization = (
                    UnweightedKernelization(config, graph)
                )
            case Weighting.WEIGHTED:
                self._kernelizer = WeightedKernelization(config, graph)
            case _:
                raise NotImplementedError

    def preprocess(self) -> nx.Graph:
        """
        Run preprocessing steps on the graph.

        This will reduce the size of the graph. Do not forget to call `rebuild()` to
        convert solutions on the reduced graph into solutions on the original graph!
        """
        return self._kernelizer.preprocess()

    def rebuild(self, partial_solution: set[int]) -> set[int]:
        """
        Expand from MIS solutions on a reduced graph obtained by `preprocess()` into
        solutions on the original graph.

        Arguments:
            partial_solution A solution on the reduced graph. Note that we do not
            check that this solution is correct.
        Returns:
            A solution on the original graph.
        """
        return self._kernelizer.rebuild(partial_solution)

    def is_independent(self, nodes: list[int]) -> bool:
        """
        Determine if a set of nodes represents an independent set within a given graph.

        Arguments:
            A list of nodes within the latest iteration of the graph.
        """
        return self._kernelizer.is_independent(nodes)


class BaseKernelization(BasePreprocessor, abc.ABC):
    """
    Shared base class for kernelization.
    """

    def __init__(self, config: "SolverConfig", graph: nx.Graph) -> None:
        self.cost_picker = BaseWeightPicker.for_weighting(config.weighting)

        # The latest version of the graph.
        # We rewrite it progressively to decrease the number of
        # nodes and edges.
        self.kernel: nx.Graph = graph.copy()
        self.initial_number_of_nodes = self.kernel.number_of_nodes()
        self.rule_application_sequence: list[BaseRebuilder] = []
        self.config = config

        # An index used to generate new node numbers.
        self._new_node_gen_counter: int = 1
        if self.initial_number_of_nodes > 0:
            self._new_node_gen_counter = max(self.kernel.nodes()) + 1

        # Get rid of any node with a self-loop (a node that is its own
        # neighbor), as it cannot be part of a solution and we rely upon
        # their absence in rule applications.
        for node in list(self.kernel.nodes()):
            if self.kernel.has_edge(node, node):
                self.kernel.remove_node(node)

    """
    Apply all the rules, in every possible order, until the graph cannot
    be reduced further.

    This method is left abstract as the list of rules may differ for
    various kinds of graphs (e.g. unweighted vs. weighted).
    """

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
        """
        Return the weight of a single node.
        """
        return self.cost_picker.node_weight(self.kernel, node)

    def subgraph_weight(self, nodes: Iterable[int]) -> float:
        """
        Return the total weight of a subgraph.
        """
        return self.cost_picker.subgraph_weight(self.kernel, nodes)

    @abc.abstractmethod
    def is_maximum(self, node: int, neighbors: list[int]) -> bool:
        """
        Determine whether any neighbor of a node has a weight strictly
        greater than that node.
        """
        ...

    def is_isolated_and_maximum(self, node: int) -> bool:
        """
        Determine whether a node is isolated and maximum, i.e.
        1. this node + its neighbors represent a clique; AND
        2. no node in the neighborhood has a weight strictly greater than `node`.
        """
        neighborhood = closed_neighborhood(self.kernel, node)
        if not self.is_subclique(nodes=neighborhood):
            return False
        return self.is_maximum(node, neighborhood)

    @abc.abstractmethod
    def add_node(self, weight: float) -> int:
        """
        Add a new node with a unique index.
        """
        ...

    def preprocess(self) -> nx.Graph:
        """
        Apply all rules, exhaustively, until the graph cannot be reduced
        further, storing the rules for rebuilding after the fact.
        """
        # Invariant: from this point, `self.kernel` does not contain any
        # self-loop.
        self.initial_cleanup()
        while (kernel_size_start := self.kernel.number_of_nodes()) > 0:
            logger.info("preprocessing - current kernel size is %s", kernel_size_start)
            self.search_rule_neighborhood_removal()
            self.search_rule_isolated_node_removal()
            self.search_rule_twin_reduction()
            self.search_rule_node_fold()
            self.search_rule_unconfined_and_diamond()

            kernel_size_end: int = self.kernel.number_of_nodes()
            assert kernel_size_end <= kernel_size_start  # Just in case.
            if kernel_size_start == kernel_size_end:
                # We didn't find any rule to apply, time to stop.
                logger.info("preprocessing - ran out of rules")
                break
        return self.kernel

    def add_rebuilder(self, rebuilder: "BaseRebuilder") -> None:
        """
        Store a rebuilder step to be called during rebuild().
        """
        logger.debug("adding rebuilder %s", rebuilder)
        self.rule_application_sequence.append(rebuilder)

    # -----------------cleanup----------------------------------------
    @abc.abstractmethod
    def initial_cleanup(self) -> None:
        """
        One-time cleanup of nodes that are trivially useless, e.g. negative weights.
        """
        ...

    # -----------------neighborhood_removal---------------------------
    @abc.abstractmethod
    def search_rule_neighborhood_removal(self) -> None:
        """
        Weighted: If a node has a greater weight than all its neighbors together,
        remove the node (it will be part of the WMIS) and all its neighbors (they
        won't).
        Unweighted: Noop.
        """
        ...

    # -----------------isolated_node_removal---------------------------
    @abc.abstractmethod
    def get_nodes_with_strictly_higher_weight(
        self, node: int, neighborhood: Iterable[int]
    ) -> list[int]:
        """
        Return the nodes with a weight strictly higher than a give node.

        Arguments:
            node: The main node.
            neighborhood: The list of nodes in which to search for a
                weight strictly higher than `node`.

        Returns:
            A list (possibly empty) of nodes from `neighborhood`. All
            these nodes are guaranteed to have a weight strictly higher
            than that of `node`.

            In unweighted mode, this list is always empty.
        """
        ...

    def apply_rule_isolated_node_removal(self, isolated: int) -> None:
        """
        Remove an isolated node / store the rebuild operation.

        Arguments:
            isolated An isolated node. We do not re-check that it is isolated.
        """
        neighborhood = list(self.kernel.neighbors(isolated))
        higher = self.get_nodes_with_strictly_higher_weight(isolated, neighborhood)
        rule_app = RebuilderIsolatedNodeRemoval(isolated, higher)
        self.add_rebuilder(rule_app)
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
                logger.debug("search_rule_isolated_node_removal: removing node %s", node)
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

    def apply_rule_node_fold(
        self, v: Any, w_v: float, u: Any, w_u: float, x: Any, w_x: float
    ) -> None:
        """
        Fold three nodes V, U and X into a new single node / store the rebuild operation.

        Arguments:
            v, u, x: Three nodes. U and X MUST both be neightours of V. There MUST NOT
                be any edge between U and X.
            w_v, w_u, w_x: The weight of nodes v, u, x. We MUST have w_u + w_x > w_v
                (always true in unweighted mode). We MUST have w_x <= w_v and w_u <= w_v
                (always true in unweighted mode).
        """
        v_prime = self.add_node(w_u + w_x - w_v)
        rule_app = RebuilderNodeFolding(v, u, x, v_prime)
        self.add_rebuilder(rule_app)
        self.fold_three(v, u, x, v_prime)

    def search_rule_node_fold(self) -> None:
        """
        If a node V has exactly two neighbors U and X and there is no edge
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
            self.apply_rule_node_fold(v=v, w_v=w_v, u=u, w_u=w_u, x=x, w_x=w_x)

    # -----------------twin reduction---------------------------

    @abc.abstractmethod
    def twin_category(self, u: int, v: int, neighbors: list[int]) -> _TwinCategory:
        """
        Determine which operations we can perform on two twin nodes.

        Arguments:
            - u, v: two distinct nodes with the same set of neighbors
            - neighbors: the neighbors of u (or equivalently v)
        """
        ...

    def find_removable_twin(self, v: int) -> _Twin | None:
        """
        Find a twin of a node, i.e. another node with the same
        neighbors, and check what we can do with this node.

        Arguments:
            v: a node
        Returns:
            A `_Twin` if any twin of v was found and it can be
            removed, `None` otherwise.
        """
        neighbors_v: set[int] = set(self.kernel.neighbors(v))
        # We recompute `neighbors_v` at each call to `find_removable_twin` because we
        # may have changed the graph between these calls.

        for u in self.kernel.nodes():
            if u == v:
                continue
            neighbors_u: set[int] = set(self.kernel.neighbors(u))
            if neighbors_u == neighbors_v:
                # Note: Since there are no self-loops, we can deduce
                # that U and V are also not neighbors.
                list_neighbors_u = list(neighbors_u)
                category = self.twin_category(u, v, list_neighbors_u)
                if category == _TwinCategory.CannotRemove:
                    continue
                return _Twin(node=int(u), neighbors=list_neighbors_u, category=category)
        return None

    def fold_twin(self, u: int, v: int, v_prime: int, u_neighbors: list[int]) -> None:
        """
        Fold two twins U and V into a single node V'.

        Arguments:
            u, v: The nodes to fold.
            v_prime: The new node, already created.
            u_neighbors: The neighbors of U (or equivalently of V).
        """
        neighborhood_u_neighbors: list[int] = list(
            set().union(*[set(self.kernel.neighbors(node)) for node in u_neighbors])
        )
        for neigh in neighborhood_u_neighbors:
            self.kernel.add_edge(v_prime, neigh)
        self.kernel.remove_nodes_from([u, v])
        self.kernel.remove_nodes_from(u_neighbors)

    def apply_rule_twins_in_solution(self, v: int, u: int, neighbors_u: list[int]) -> None:
        """
        Remove two twin nodes and their neighbors / store the rebuild operation.

        We use this rule when we know that the twins will always be part of the solution.

        Arguments:
            u, v: The twin nodes.
            neighbors_u: The neighbors of U (or equivalently of V).
        """
        rule_app = RebuilderTwinAlwaysInSolution(v, u)
        self.add_rebuilder(rule_app)
        self.kernel.remove_nodes_from(neighbors_u)
        self.kernel.remove_nodes_from([u, v])

    def apply_rule_twin_independent(self, u: int, v: int, neighbors: list[int]) -> None:
        """
        Remove two twin nodes and their neighbors / store the rebuild operation.

        We use this rule when U and V are independent, i.e. either all the neighbors
        are part of the solution or both U and V are part of the solution, but we don't
        yet know which.

        Arguments:
            u, v: The twin nodes.
            neighbors_u: The neighbors of U (or equivalently of V).
        """
        w_u = self.node_weight(u)
        w_v = self.node_weight(v)
        w_neighbors = self.subgraph_weight(neighbors)
        new_weight = w_neighbors - (w_u + w_v)
        assert (
            new_weight > 0
        )  # In weighted mode, we have checked that `w_u + w_v < w_neighbors_sum`. In non-weighted, it's exactly 1.0
        v_prime = self.add_node(new_weight)
        rule_app_B = RebuilderTwinIndependent(v, u, neighbors, v_prime)
        self.add_rebuilder(rule_app_B)
        self.fold_twin(u=u, v=v, v_prime=v_prime, u_neighbors=neighbors)

    def search_rule_twin_reduction(self) -> None:
        """
        If a node V and a node U have the exact same neighbors
        (which indicates that they're not nightbours themselves),
        we may be able to merge U, V and their neighborhoods into
        a single node.

        Note: as of this writing, in unweighted mode, the heuristic
        works when there are exactly 3 neighbors.
        """
        if self.kernel.number_of_nodes() == 0:
            return
        for v in list(self.kernel.nodes()):
            # Since we're modifying `self.kernel` while iterating, we're
            # calling `list()` to make sure that we still have some kind
            # of valid iterator.
            if not self.kernel.has_node(v):
                continue
            twin: _Twin | None = self.find_removable_twin(v)
            if twin is None:
                continue
            u = twin.node
            category = self.twin_category(u, v, twin.neighbors)
            if category == _TwinCategory.Independent:
                self.apply_rule_twin_independent(u, v, twin.neighbors)
            elif category == _TwinCategory.InSolution:
                self.apply_rule_twins_in_solution(u, v, twin.neighbors)
            else:
                # We cannot remove this twin.
                pass

    # -----------------unconfined reduction---------------------------

    @abc.abstractmethod
    def search_rule_unconfined_and_diamond(self) -> None:
        """
        Look for unconfined nodes, i.e. a category of nodes
        for which we can prove easily that they cannot be part
        of a solution.
        """
        ...


class UnweightedKernelization(BaseKernelization):
    """
    Apply well-known transformations to the graph to reduce its size without
    compromising the result.

    This algorithm is adapted from e.g.:
    https://schulzchristian.github.io/thesis/masterarbeit_demian_hespe.pdf

    Unless you are experimenting with your own preprocessors, you should
    probably use `Kernelization` in your pipeline.
    """

    def add_node(self, weight: float) -> int:
        """
        Add a node with a weight of exactly 1.0.

        Arguments:
            weight: MUST be 1.0 in unweighted mode.

        Returns:
            The index of the new node.
        """
        assert weight == 1.0
        # There are only additions and subtractions of small integers, so we don't expect rounding errors.
        node = self._new_node_gen_counter
        self._new_node_gen_counter += 1
        self.kernel.add_node(node)
        return node

    def is_maximum(self, node: int, neighbors: list[int]) -> bool:
        """
        Since all nodes have the same weight, no node has a strictly higher weight.
        """
        return True

    # -----------------cleanup----------------------------------------
    def initial_cleanup(self) -> None:
        """
        One-time cleanup of nodes that are trivially useless, e.g. negative weights.
        """
        # In unweighted, nothing to do.
        return

    # -----------------isolated node removal--------------------
    def get_nodes_with_strictly_higher_weight(
        self, node: int, neighborhood: Iterable[int]
    ) -> list[int]:
        """
        Since all nodes have the same weight, no node has a strictly higher weight.
        """
        return []

    # -----------------neighborhood removal---------------------

    def search_rule_neighborhood_removal(self) -> None:
        # This rule is a noop in unweighted mode.
        return

    # -----------------twin reduction---------------------------

    def twin_category(self, u: int, v: int, neighbors: list[int]) -> _TwinCategory:
        """
        Determine which operations we can perform on two twin nodes.

        Arguments:
            - u, v: two distinct nodes with the same set of neighbors
            - neighbors: the neighbors of u (or equivalently v)

        Returns:
            - CannotRemove if the number of neighbors is not exactly 3.
            - Independent if the neighbors are independent from each other.
            - InSolution otherwise.
        """
        if len(neighbors) != 3:
            # The heuristic only works with exactly 3 neighbors.
            return _TwinCategory.CannotRemove
        if self.is_independent(neighbors):
            # Either all the neighbors are part of the solution or U and V
            # are part of the solution.
            return _TwinCategory.Independent
        # Since we have exactly 3 neighbors and there is at least one dependency:
        # 1. at most 2 neighbors are part of the solution;
        # 2. at least 1 neighbor W is not part of the solution.
        #
        # Since 2., there is a solution that includes U and V.
        #
        # Since 1., a solution that includes U and V is at least as
        # good as a solution that includes some of the neighbors.
        #
        # Therefore, we can always adopt U and V as a solution.
        return _TwinCategory.InSolution

    # -----------------unconfined reduction---------------------------
    def aux_search_confinement(self, neighbors_S: set[int], S: set[int]) -> _ConfinementAux | None:
        """
        Attempt to find a neighbor U of S such that:
        - there is exactly one node in S that is a neighbor of U;
        - the number of neighbors of U that are not neighbors of S is minimized.
        """

        # Best node found so far.
        best_node: int | None = None

        # The number of neighbors we're attempting to minimize.
        best_set_diff_len: int | None = None

        # The best `neighbors(U) \ (neighbors(S) union {S})` we've found so far.
        best_set_diff: set[int] = set()

        for u in neighbors_S:
            neighbors_u: set[int] = set(self.kernel.neighbors(u))
            inter: set[int] = neighbors_u & S
            if len(inter) != 1:
                continue
            candidate_set_diff = neighbors_u - neighbors_S - S
            candidate_set_diff_len = len(neighbors_u - neighbors_S - S)
            if best_set_diff_len is None or candidate_set_diff_len < best_set_diff_len:
                best_node = u
                best_set_diff = candidate_set_diff
                best_set_diff_len = candidate_set_diff_len
        if best_node is None:
            return None
        assert best_set_diff_len is not None
        return _ConfinementAux(node=best_node, set_diff=best_set_diff)

    def apply_rule_unconfined(self, v: int) -> None:
        rule_app = RebuilderUnconfined()
        self.add_rebuilder(rule_app)
        self.kernel.remove_node(v)

    def unconfined_loop(self, v: int, S: set[int], neighbors_S: set[int]) -> bool:
        """
        Starting from a node V, attempt to determine whether it is unconfined.

        Arguments:
            v The node we're examining.
            S (inout) A set we're building to help determine whether v is unconfined.
                Should initially be {v}, grown during each iteration.
            neighbors_S (inout) The set of neighbors of all elements of S

        Returns False if the work is over (we couldn't find any matching value).
        """
        confinement = self.aux_search_confinement(neighbors_S, S)
        # If there is no such node, then v is confined and we can't do anything with it.
        if confinement is None:
            return False
        diff_size = len(confinement.set_diff)
        # If N(u)\N[S] = ∅, then v is unconfined, we can simply remove it.
        if diff_size == 0:
            self.apply_rule_unconfined(v)
            return False
        # If N (u)\ N [S] is a single node w,
        # then add w to S and repeat the algorithm.
        elif diff_size == 1:
            w = next(iter(confinement.set_diff))  # We've just checked that this set has size 1.
            S.add(w)
            neighbors_S |= set(self.kernel.neighbors(w))
            neighbors_S -= {w}
            return True
        # Otherwise, v is confined.
        return False

    def search_rule_unconfined_and_diamond(self) -> None:
        """
        Look for unconfined nodes, i.e. nodes for which we can prove easily
        that they cannot be part of a solution.
        """
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
            # Then find u∈N(S) such that |N(u) ∩ S| = 1
            # and |N(u)\N[S]| is minimized
            while self.unconfined_loop(v, S, neighbors_S):
                # All the work is done in `unconfined_loop`.
                pass


class WeightedKernelization(BaseKernelization):

    def add_node(self, weight: float) -> int:
        """
        Add a new node with a unique index.

        Arguments:
            weight: A strictly positive weight.
        Returns:
            The index of the new node.
        """
        # Our invariant is that we never add nodes with a weight <= 0.
        assert weight > 0
        node = self._new_node_gen_counter
        self._new_node_gen_counter += 1
        self.kernel.add_node(node)
        self.cost_picker.set_node_weight(self.kernel, node, weight)
        return node

    def is_maximum(self, node: int, neighbors: list[int]) -> bool:
        """
        Determine whether any neighbor of a node has a weight strictly
        greater than that node.
        """
        max: float = self.node_weight(node)
        for v in neighbors:
            if v != node and self.node_weight(v) > max:
                return False
        return True

    # -----------------cleanup----------------------------------------
    def initial_cleanup(self) -> None:
        """
        One-time cleanup of nodes that are trivially useless, e.g. negative weights.
        """
        # Negative weight nodes can never be part of a solution, so we
        # simply remove them.
        #
        # After this, our invariant is that any node we create has a
        # weight > 0.
        for node in list(self.kernel.nodes):
            if self.node_weight(node) <= 0:
                self.kernel.remove_node(node)

    # -----------------isolated node removal--------------------
    def get_nodes_with_strictly_higher_weight(
        self, node: int, neighborhood: Iterable[int]
    ) -> list[int]:
        """
        Return the nodes with a weight strictly higher than a give node.

        Arguments:
            node: The main node.
            neighborhood: The list of nodes in which to search for a
                weight strictly higher than `node`.

        Returns:
            A list (possibly empty) of nodes from `neighborhood`. All
            these nodes are guaranteed to have a weight strictly higher
            than that of `node`.
        """
        pivot = self.node_weight(node)
        result: list[int] = []
        for n in neighborhood:
            if self.node_weight(n) > pivot:
                result.append(n)
        return result

    # -----------------unconfined reduction---------------------------

    def search_rule_unconfined_and_diamond(self) -> None:
        """
        Look for unconfined nodes, i.e. a category of nodes
        for which we can prove easily that they cannot be part
        of a solution.

        As of this writing, we don't know how to detect unconfined
        nodes in weighted mode, so this step is a NOOP.
        """
        return None

    # -----------------neighborhood_removal---------------------------
    def neighborhood_weight(self, node: int) -> float:
        """
        Return the total weight of the neighbors of a node.
        """
        return self.subgraph_weight(self.kernel.neighbors(node))

    def apply_rule_neighborhood_removal(self, node: int) -> None:
        """
        Remove a node and its neighbors / store rebuild role.
        """
        rule_app = RebuilderNeighborhoodRemoval(node)
        self.add_rebuilder(rule_app)
        # Note: We need to copy the iterator into a list as `remove_nodes_from` will
        # invalidate the `neighbors` iterator.
        self.kernel.remove_nodes_from(list(self.kernel.neighbors(node)))
        self.kernel.remove_node(node)

    def search_rule_neighborhood_removal(self) -> None:
        """
        If a node has a greater weight than all its neighbors together,
        remove the node and all its neighbors.

        During rebuild, the node will always be part of the WMIS, the
        neighbors never will.
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
            node_weight: float = self.node_weight(node)
            neighborhood_weight_sum = self.neighborhood_weight(node)
            if node_weight >= neighborhood_weight_sum:
                self.apply_rule_neighborhood_removal(node)

    # -----------------twin reduction---------------------------

    def twin_category(self, u: int, v: int, neighbors: list[int]) -> _TwinCategory:
        """
        Determine which operations we can perform on two twin nodes.

        Arguments:
            - u, v: two distinct nodes with the same set of neighbors
            - neighbors: the neighbors of u (or equivalently v)

        Returns:
            If the total weight of U + V is at least as large of the total weight of the
                neighbors, we can always find a solution in which U and V both are, so
                `InSolution`.
            If the neighbors are independent and, naming X the neighbor with the smallest
                weight, if the weight of U + V + X is at least as large of the total weight
                of the neighbors, then there is always a solution in which either both U and
                V are, or all the neighbors are, so `Independent`.
            Otherwise, we don't have any specific preprocessing for these twins, so `CannotRemove`.
        """
        w_u: float = self.node_weight(u)
        w_v: float = self.node_weight(v)
        w_neighbors: list[float] = [self.node_weight(node) for node in neighbors]
        w_neighbors_sum: float = sum(w_neighbors)
        if w_u + w_v >= w_neighbors_sum:
            # U and V are always part of the solution and the neighbors are never part of the solution.
            return _TwinCategory.InSolution
        if self.is_independent(neighbors) and (w_u + w_v >= w_neighbors_sum - min(w_neighbors)):
            # Either a subset of the neighbors is part of the solution or U and V are part of the solution.
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
        """
        By definition, unconfined nodes are never part of the solution,
        so rebuilding is a noop.
        """
        pass


class RebuilderTwinIndependent(BaseRebuilder):
    def __init__(self, v: int, u: int, neighbors: list[int], v_prime: int):
        """
        Invariants:
         - V has exactly the same neighbors as U;
         - there is no self-loop around U or V (hence U and V are not
            neighbors);
         - there is no edge between any of the neighbors;
         - V' is the node obtained by merging U, V and the neighbors.
        """
        self.v: int = v
        self.u: int = u
        self.neighbors = neighbors
        self.v_prime: int = v_prime

    def rebuild(self, partial_solution: set[int]) -> None:
        if self.v_prime in partial_solution:
            # Since V' is part of the solution, none of its
            # neighbors is part of the solution. Consequently,
            # either U and V can be added to grow the solution
            # or neighbors can be added to grow the solution,
            # without affecting the rest of the system.
            partial_solution.update(self.neighbors)
            partial_solution.remove(self.v_prime)
        else:
            # The only neighbors of U and V are represented
            # by V'. Since V' is not part of the solution,
            # and since U and V are not neighbors, we can
            # always add U and V.
            partial_solution.add(self.u)
            partial_solution.add(self.v)


class RebuilderTwinAlwaysInSolution(BaseRebuilder):
    def __init__(self, v: int, u: int):
        """
        Invariants:
         - V has exactly the same neighbors as U;
         - there is no self-loop around U;
         - there is at least one connection between two neighbors of U.
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
