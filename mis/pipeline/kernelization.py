import abc

import networkx as nx
from networkx.classes.reportviews import DegreeView
from mis.pipeline.preprocessor import BasePreprocessor
from mis.shared.graphs import is_independent


class BaseKernelization(BasePreprocessor, abc.ABC):
    """
    Shared base class for kernelization.
    """

    def __init__(self, graph: nx.Graph) -> None:
        # The latest version of the graph.
        # We rewrite it progressively to decrease the number of
        # nodes and edges.
        self.kernel: nx.Graph = graph.copy()
        self.initial_number_of_nodes = self.kernel.number_of_nodes()
        self.rule_application_sequence: list[BaseRebuilder] = []

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

    @abc.abstractmethod
    def preprocess(self) -> nx.Graph:
        # Invariant: from this point, `self.kernel` does not contain any
        # self-loop.
        ...

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

    def is_isolated(self, node: int) -> bool:
        """
        Determine whether a node is isolated, i.e. this node + its neighbours
        represent a clique.
        """
        closed_neighborhood: list[int] = list(self.kernel.neighbors(node))
        closed_neighborhood.append(node)
        if self.is_subclique(nodes=closed_neighborhood):
            return True
        return False

    def _add_node(self) -> int:
        """
        Add a new node with a unique index.
        """
        node = self._new_node_gen_counter
        self._new_node_gen_counter += 1
        self.kernel.add_node(node)
        return node


class Kernelization(BaseKernelization):
    """
    Apply well-known transformations to the graph to reduce its size without
    compromising the result.

    This algorithm is adapted from e.g.:
    https://schulzchristian.github.io/thesis/masterarbeit_demian_hespe.pdf

    Unless you are experimenting with your own preprocessors, you should
    probably use Kernelization in your pipeline.
    """

    def preprocess(self) -> nx.Graph:
        """
        Apply all rules, exhaustively, until the graph cannot be reduced
        further, storing the rules for rebuilding after the fact.
        """
        while (kernel_size_start := self.kernel.number_of_nodes()) > 0:
            self.search_rule_isolated_node_removal()
            self.search_rule_twin_reduction()
            self.search_rule_node_fold()
            self.search_rule_unconfined_and_diamond()
            kernel_size_end: int = self.kernel.number_of_nodes()
            if kernel_size_start - kernel_size_end == 0:
                # We didn't find any rule to apply, time to stop.
                break
        return self.kernel

    # -----------------isolated_node_removal---------------------------
    def apply_rule_isolated_node_removal(self, isolated: int) -> None:
        rule_app = RebuilderIsolatedNodeRemoval(isolated)
        self.rule_application_sequence.append(rule_app)
        neighborhood = list(self.kernel.neighbors(isolated))
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
            if self.is_isolated(node):
                self.apply_rule_isolated_node_removal(node)

    # -----------------unweighted_node_folding---------------------------

    def _fold_three(self, v: int, u: int, x: int, v_prime: int) -> None:
        """
        Fold three nodes V, U and X into a new single node V'.
        """
        neighbors_v_prime = set(self.kernel.neighbors(u)) | set(self.kernel.neighbors(x))
        for node in neighbors_v_prime:
            self.kernel.add_edge(v_prime, node)
        self.kernel.remove_nodes_from([v, u, x])

    def apply_rule_node_fold(self, v: int, u: int, x: int) -> None:
        v_prime = self._add_node()
        rule_app = RebuilderNodeFolding(v, u, x, v_prime)
        self.rule_application_sequence.append(rule_app)
        self._fold_three(v, u, x, v_prime)

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
            if self.kernel.has_node(v):
                if self.kernel.degree(v) == 2:
                    [u, x] = self.kernel.neighbors(v)
                    if not self.kernel.has_edge(u, x):
                        self.apply_rule_node_fold(v, u, x)

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
            go_to_next_loop: bool = True
            while go_to_next_loop:
                # Then find u∈N(S) such that |N(u) ∩ S| = 1
                # and |N(u)\N[S]| is minimized
                go_to_next_loop = self.unconfined_loop(v, S, neighbors_S)

    # -----------------twin reduction---------------------------
    def fold_twin(self, u: int, v: int, v_prime: int, neighbors_u: list[int]) -> None:
        w_0: int = neighbors_u[0]
        w_1: int = neighbors_u[1]
        w_2: int = neighbors_u[2]
        neighbors_w_0 = set(self.kernel.neighbors(w_0))
        neighbors_w_1 = set(self.kernel.neighbors(w_1))
        neighbors_w_2 = set(self.kernel.neighbors(w_2))
        neighbors_v_prime = neighbors_w_0 | neighbors_w_1 | neighbors_w_2
        for node in neighbors_v_prime:
            self.kernel.add_edge(node, v_prime)
        self.kernel.remove_nodes_from([u, v, w_0, w_1, w_2])

    def find_twin(self, v: int) -> int | None:
        """
        Find a twin of a node, i.e. another node with the same
        neighbours.
        """
        neighbors_v: set[int] = set(self.kernel.neighbors(v))
        for u in list(self.kernel.nodes()):
            # Note: It might be sufficient to walk through
            # the neighbours of neighbours of neighbours of v.
            # Unclear whether this would be faster.
            if u == v:
                continue
            if not self.kernel.has_node(u):
                # FIXME: Can this happen?
                continue
            if not self.kernel.has_node(v):
                # FIXME: Can this happen?
                continue
            neighbors_u: set[int] = set(self.kernel.neighbors(u))
            if neighbors_u == neighbors_v:
                # Note: Since there are no self-loops, we can deduce
                # that U and V are also not neighbours.
                return int(u)
        return None

    def apply_rule_twin_independent(self, v: int, u: int, neighbors_u: list[int]) -> None:
        v_prime = self._add_node()
        rule_app = RebuilderTwinIndependent(
            v, u, neighbors_u[0], neighbors_u[1], neighbors_u[2], v_prime
        )
        self.rule_application_sequence.append(rule_app)
        self.fold_twin(u, v, v_prime, neighbors_u)

    def apply_rule_twin_has_dependency(self, v: int, u: int, neighbors_u: list[int]) -> None:
        rule_app = RebuilderTwinHasDependency(v, u)
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_nodes_from(neighbors_u)
        self.kernel.remove_nodes_from([u, v])

    def search_rule_twin_reduction(self) -> None:
        """
        If a node has exactly 3 neighbours and a twin (another
        node with the exact same neighbours), we can merge the
        5 nodes.
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
            if self.kernel.degree(v) != 3:
                continue
            u: int | None = self.find_twin(v)
            if u is None:
                continue
            neighbors_u: list[int] = list(self.kernel.neighbors(u))
            if self.is_independent(neighbors_u):
                self.apply_rule_twin_independent(v, u, neighbors_u)
            else:
                self.apply_rule_twin_has_dependency(v, u, neighbors_u)


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
    def __init__(self, isolated: int):
        self.isolated = isolated

    def rebuild(self, partial_solution: set[int]) -> None:
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
    def __init__(self, v: int, u: int, w_0: int, w_1: int, w_2: int, v_prime: int):
        """
        Invariants:
         - U has exactly 3 neighbours W0, W1, W2;
         - V has exactly the same neighbours as U;
         - there is no self-loop around U or V (hence U and V are not
            neighbours);
         - there is no edge between W1, W2, W3;
         - V' is the node obtained by merging U, V, W1, W2, W3.
        """
        self.v: int = v
        self.u: int = u
        self.w_0: int = w_0
        self.w_1: int = w_1
        self.w_2: int = w_2
        self.v_prime: int = v_prime

    def rebuild(self, partial_solution: set[int]) -> None:
        if self.v_prime in partial_solution:
            # Since V' is part of the solution, none of its
            # neighbours is part of the solution. Consequently,
            # either U and V can be added to grow the solution
            # or W0, W1, W2 can be added to grow the solution,
            # without affecting the rest of the system.
            partial_solution.add(self.w_0)
            partial_solution.add(self.w_1)
            partial_solution.add(self.w_2)
            partial_solution.remove(self.v_prime)
        else:
            # The only neighbours of U and V are represented
            # by V'. Since V' is not part of the solution,
            # and since U and V are not neighbours, we can
            # always add U and V.
            partial_solution.add(self.u)
            partial_solution.add(self.v)


class RebuilderTwinHasDependency(BaseRebuilder):
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
