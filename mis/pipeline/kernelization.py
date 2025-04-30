import abc

import networkx as nx
from networkx.classes.reportviews import DegreeView
from preprocessor import BasePreprocessor


class Kernelization(BasePreprocessor, abc.ABC):
    def __init__(self, graph: nx.Graph):
        self.graph: nx.Graph = graph.copy()
        self.kernel: nx.Graph = graph.copy()
        self.rule_application_sequence: list[BaseRebuilder] = []

        self.new_node_current_index = 1
        if self.kernel.number_of_nodes() > 0:
            self.new_node_current_index = max([
                node for node in self.graph.nodes()]
            ) + 1

    @abc.abstractmethod
    def exhaustive_rules_applications(self) -> nx.Graph:
        ...

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

    def is_independent(self, graph: nx.Graph, nodes: set[int]) -> bool:
        """
        Determine if a set of nodes represents an independent set
        within a given graph.

        Returns:
            True if the nodes in `nodes` represent an independent
                set within `graph`.
            False otherwise, e.g. if there's at least one connection
                between two nodes of `nodes`
        """
        for u in nodes:
            for v in nodes:
                if graph.has_edge(u, v):
                    return False
        return True

    def is_subclique(self, graph: nx.Graph, nodelist: list[int]) -> bool:
        H: nx.Graph = graph.subgraph(nodelist)
        n: int = len(nodelist)
        return H.size() == n * (n - 1) / 2

    def is_isolated(self, vertex: int) -> bool:
        closed_neighborhood: list[int] = list(self.kernel.neighbors(vertex))
        closed_neighborhood.append(vertex)
        if self.is_subclique(
            graph=self.kernel,
            nodelist=closed_neighborhood
        ):
            return True
        return False


class UnweightedKernelization(Kernelization):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph=graph)
        # unweighted graph, Node names have to be positive integers

    def exhaustive_rules_applications(self) -> nx.Graph:
        while len(list(self.kernel.nodes())) > 0:
            kernel_size_start: int = len(list(self.kernel.nodes()))
            self.search_rule_isolated_vertex_removal()
            self.search_rule_twin_reduction()
            self.search_rule_vertex_fold()
            self.search_rule_unconfined_and_diamond()
            kernel_size_end: int = len(list(self.kernel.nodes()))
            if kernel_size_start - kernel_size_end == 0:
                break
        return self.kernel

    # -----------------isolated_vertex_removal---------------------------
    def apply_rule_isolated_vertex_removal(self, isolated: int) -> None:
        rule_app = RebuilderIsolatedVertexRemoval(isolated)
        self.rule_application_sequence.append(rule_app)
        neighborhood = list(self.kernel.neighbors(isolated))
        self.kernel.remove_nodes_from(neighborhood)
        self.kernel.remove_node(isolated)

    def search_rule_isolated_vertex_removal(self) -> None:
        for node in list(self.kernel.nodes()):
            if not self.kernel.has_node(node):
                continue
            if self.is_isolated(node):
                self.apply_rule_isolated_vertex_removal(node)

    # -----------------unweighted_vertex_folding---------------------------

    def folding(self, v: int, u: int, x: int, v_prime: int) -> None:
        N_u: list[int] = list(self.kernel.neighbors(u))
        N_x: list[int] = list(self.kernel.neighbors(x))
        N_v_prime: list[int] = list(set(N_u) | set(N_x))
        for node in N_v_prime:
            self.kernel.add_edge(v_prime, node)
        self.kernel.remove_nodes_from([v, u, x])

    def apply_rule_vertex_fold(self, v: int, u: int, x: int):
        v_prime: int = self.new_node_current_index
        self.kernel.add_node(v_prime)
        self.new_node_current_index += 1
        rule_app = RebuilderVertexFolding(v, u, x, v_prime)
        self.rule_application_sequence.append(rule_app)
        self.folding(v, u, x, v_prime)

    def search_rule_vertex_fold(self) -> None:
        if self.kernel.number_of_nodes() == 0:
            return
        assert isinstance(self.kernel.degree, DegreeView)
        for v in list(self.kernel):
            if not self.kernel.has_node(v):
                continue
            if self.kernel.has_node(v):
                if self.kernel.degree(v) == 2:
                    neighbors: list[int] = list(self.kernel.neighbors(v))
                    u: int = neighbors[0]
                    x: int = neighbors[1]
                    if not self.kernel.has_edge(u, x):
                        self.apply_rule_vertex_fold(v, u, x)

    # -----------------unconfined reduction---------------------------
    def search_confinement(self,
                           N_S: set[int],
                           S: set[int]) -> tuple[int, int, set[int]]:
        min: int = -1
        min_value: int = self.graph.number_of_nodes() + 2
        min_set_diff: set[int] = set()
        for u in N_S:
            N_u: set[int] = set(self.kernel.neighbors(u))
            inter: set[int] = N_u & S
            if len(inter) == 1:
                if len(N_u - N_S - S) < min_value:
                    min = u
                    min_set_diff = N_u - N_S - S
                    min_value = len(min_set_diff)
        return min, min_value, min_set_diff

    def apply_rule_unconfined(self, v: int) -> None:
        rule_app = RebuilderUnconfined()
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_node(v)

    def unconfined_loop(self, v: int, S: set[int], N_S: set[int]) -> bool:
        min: int = 0
        min_value: int = 0
        min_set_diff: set[int] = set()
        min, min_value, min_set_diff = self.search_confinement(N_S, S)
        next_loop: bool = False
        # If there is no such vertex, then v is confined.
        if min == -1:
            pass
            '''
            if self.find_diamond_reduction(N_S, S):
                self.apply_rule_diamond(v)
            '''
        # If N(u)\N[S] = ∅, then v is unconfined.
        if min_value == 0:
            self.apply_rule_unconfined(v)
        # If N (u)\ N [S] is a single vertex w,
        # then add w to S and repeat the algorithm.
        elif min_value == 1:
            w = list(min_set_diff)[0]
            S.add(w)
            N_S |= set(self.kernel.neighbors(w))
            N_S -= {w}
            next_loop = True
        # Otherwise, v is confined.
        else:
            pass
        return next_loop

    def search_rule_unconfined_and_diamond(self) -> None:
        if self.kernel.number_of_nodes() == 0:
            return
        for v in list(self.kernel.nodes()):
            if not self.kernel.has_node(v):
                continue
            # First, initialize S = {v}.
            S: set[int] = {v}
            N_S: set[int] = set(self.kernel.neighbors(v))
            go_to_next_loop: bool = True
            while go_to_next_loop:
                # Then find u∈N(S) such that |N(u) ∩ S| = 1
                # and |N(u)\N[S]| is minimized
                go_to_next_loop = self.unconfined_loop(v, S, N_S)

    # -----------------twin reduction---------------------------
    def fold_twin(self, u: int, v: int, v_prime: int, N_u: set[int]) -> None:
        list_N_u = list(N_u)
        w_0: int = list_N_u[0]
        w_1: int = list_N_u[1]
        w_2: int = list_N_u[2]
        N_w_0 = set(self.kernel.neighbors(w_0))
        N_w_1 = set(self.kernel.neighbors(w_1))
        N_w_2 = set(self.kernel.neighbors(w_2))
        N_v_prime = N_w_0 | N_w_1 | N_w_2
        for node in N_v_prime:
            self.kernel.add_edge(node, v_prime)
        self.kernel.remove_nodes_from([u, v, w_0, w_1, w_2])

    def find_twin(self, v: int) -> int:
        for u in list(self.kernel.nodes()):
            if not self.kernel.has_node(u):
                continue
            if not self.kernel.has_node(v):
                continue
            if u != v:
                N_v: set[int] = set(self.kernel.neighbors(v))
                N_u: set[int] = set(self.kernel.neighbors(u))
                if N_u == N_v:
                    return u
        return -1

    def apply_rule_twin_independent(self, v: int, u: int,
                                    N_u: set[int]) -> None:
        v_prime: int = self.new_node_current_index
        self.kernel.add_node(v_prime)
        self.new_node_current_index += 1
        rule_app = RebuilderTwinIndependent(v,
                                            u,
                                            list(N_u)[0],
                                            list(N_u)[1],
                                            list(N_u)[2],
                                            v_prime)
        self.rule_application_sequence.append(rule_app)
        self.fold_twin(u, v, v_prime, N_u)

    def apply_rule_twin_has_dependency(self, v: int, u: int,
                                       N_u: set[int]) -> None:
        rule_app = RebuilderTwinHasDependency(v, u)
        self.rule_application_sequence.append(rule_app)
        self.kernel.remove_nodes_from(list(N_u))
        self.kernel.remove_nodes_from([u, v])

    def search_rule_twin_reduction(self) -> None:
        if self.kernel.number_of_nodes() == 0:
            return
        assert isinstance(self.kernel.degree, DegreeView)
        for v in list(self.kernel.nodes()):
            if not self.kernel.has_node(v):
                continue
            if self.kernel.degree(v) == 3:
                u: int = self.find_twin(v)
                if u != -1:
                    N_u: set[int] = set(self.kernel.neighbors(u))
                    if self.is_independent(self.kernel, N_u):
                        self.apply_rule_twin_independent(v, u, N_u)
                    else:
                        self.apply_rule_twin_has_dependency(v, u, N_u)


class BaseRebuilder(abc.ABC):
    """
    The pre-processing operations attempt to remove edges
    and/or vertices from the original graph. Therefore,
    when we build a MIS for these reduced graphs (the
    "partial solution"), we may end up with a solution
    that does not work for the original graph.

    Each rebuilder corresponds to one of the operations
    that reduced the size of the graph, and is charged
    with adapting the MIS solution to the greater graph.
    """

    @abc.abstractmethod
    def rebuild(self, partial_solution: set[int]) -> None:
        ...


class RebuilderIsolatedVertexRemoval(BaseRebuilder):
    def __init__(self, isolated: int):
        self.isolated = isolated

    def rebuild(self, partial_solution: set[int]) -> None:
        partial_solution.add(self.isolated)


class RebuilderVertexFolding(BaseRebuilder):
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
    def __init__(self):
        pass

    def rebuild(self, partial_solution: set[int]) -> None:
        pass


class RebuilderTwinIndependent(BaseRebuilder):
    def __init__(self, v: int, u: int, w_0: int,
                 w_1: int, w_2: int, v_prime: int):
        self.v: int = v
        self.u: int = u
        self.w_0: int = w_0
        self.w_1: int = w_1
        self.w_2: int = w_2
        self.v_prime: int = v_prime

    def rebuild(self, partial_solution: set[int]) -> None:
        if self.v_prime in partial_solution:
            partial_solution.add(self.w_0)
            partial_solution.add(self.w_1)
            partial_solution.add(self.w_2)
            partial_solution.remove(self.v_prime)
        else:
            partial_solution.add(self.u)
            partial_solution.add(self.v)


class RebuilderTwinHasDependency(BaseRebuilder):
    def __init__(self, v: int, u: int):
        self.v: int = v
        self.u: int = u

    def rebuild(self, partial_solution: set[int]) -> None:
        partial_solution.add(self.u)
        partial_solution.add(self.v)
