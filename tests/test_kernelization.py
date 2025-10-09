from __future__ import annotations
from enum import Enum
import pytest
import random

from mis import MISInstance, MISSolver, BackendConfig
from mis.pipeline.config import SolverConfig
import mis.pipeline.kernelization as ker
from mis.shared.types import Weighting
import networkx as nx


class GraphVariant(str, Enum):
    RAW = "RAW"
    SOME_WEIGHTS = "SOME WEIGHTS"
    ALL_WEIGHTS = "ALL WEIGHTS"


def add_weights_everywhere(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    for i, node in enumerate(graph.nodes):
        graph.nodes[node]["weight"] = float(i + 1)
    return graph


def add_weights_somewhere(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    for i, node in enumerate(graph.nodes):
        if i % 2 == 0:
            graph.nodes[node]["weight"] = float(i + 1)
    return graph


def graph_variants(graph: nx.Graph) -> list[tuple[GraphVariant, nx.Graph]]:
    return [
        (GraphVariant.RAW, graph),
        (GraphVariant.ALL_WEIGHTS, add_weights_everywhere(graph)),
        (GraphVariant.SOME_WEIGHTS, add_weights_somewhere(graph)),
    ]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_is_subclique(weighting: Weighting) -> None:
    """
    Test clique detection with Kernlizer.is_subclique.
    """
    SIZE = 10
    graph = nx.Graph()

    # Prepare a clique
    clique_set: set[int] = set()
    while len(clique_set) < SIZE:
        clique_set.add(random.randint(0, 100000000))
    for node in clique_set:
        graph.add_node(node)
    clique = list(clique_set)
    for i, node in enumerate(clique):
        for node_2 in clique[i + 1 :]:
            graph.add_edge(node, node_2)

    # Prepare a few points that are not in the clique
    out_of_clique = None
    while out_of_clique is None:
        node = random.randint(0, 1000000000)
        if node in clique_set:
            continue
        out_of_clique = node
        graph.add_node(node)

    test_instance = ker.Kernelization(SolverConfig(weighting=weighting), graph)._kernelizer
    assert test_instance.is_subclique(clique)
    assert not test_instance.is_subclique([out_of_clique, *clique_set])


graph_isolated = nx.Graph()
graph_isolated.add_nodes_from(range(6))
graph_isolated.add_edges_from([(0, 1), (1, 2), (2, 5), (5, 1), (2, 3), (3, 4), (4, 5)])

graph_folding = nx.Graph()
graph_folding.add_nodes_from(range(9))
graph_folding.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8)])

P_3 = nx.Graph()
P_3.add_nodes_from(range(3))
P_3.add_edges_from([(0, 1), (1, 2)])

CLAW = nx.Graph()
CLAW.add_nodes_from(range(4))
CLAW.add_edges_from([(0, 1), (1, 2), (1, 3)])

K23_CLAW_bis = nx.Graph()
K23_CLAW_bis.add_nodes_from(range(7))
K23_CLAW_bis.add_edges_from(
    [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (6, 3), (6, 4), (5, 6)]
)

K23_CLAW_twin_linked = nx.Graph()
K23_CLAW_twin_linked.add_nodes_from(range(7))
K23_CLAW_twin_linked.add_edges_from(
    [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (6, 3), (6, 4), (5, 6), (0, 1)]
)

K23_CLAW_N_linked = nx.Graph()
K23_CLAW_N_linked.add_nodes_from(range(7))
K23_CLAW_N_linked.add_edges_from(
    [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (6, 3), (6, 4), (5, 6), (2, 3)]
)

K4 = nx.Graph()
K4.add_nodes_from(range(3))
K4.add_edges_from([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])

K3_CLAW = nx.Graph()
K3_CLAW.add_nodes_from(range(6))
K3_CLAW.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (4, 3), (5, 3)])


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
@pytest.mark.parametrize("graph", [g for g in graph_variants(graph_isolated)])
def test_apply_rule_isolated_node_removal(
    weighting: Weighting, graph: tuple[GraphVariant, nx.Graph]
) -> None:
    """
    Test isolated node removal.

    This test only checks the removal/rebuild operations, without attempting to determine whether the node
    needs to be removed.
    """
    _, graph_isolated = graph
    test_instance = ker.Kernelization(SolverConfig(weighting=weighting), graph_isolated)._kernelizer
    test_instance.apply_rule_isolated_node_removal(0)

    # We should have removed {0}
    assert not test_instance.kernel.has_node(0)
    if weighting == Weighting.UNWEIGHTED or "weight" not in graph_isolated.nodes[1]:
        assert not test_instance.kernel.has_node(1)
    else:
        # In weighted mode, 1 has a higher weight than 0 so its weight
        # should have decreased.
        assert test_instance.kernel.has_node(1)
        assert (
            test_instance.node_weight(1)
            == graph_isolated.nodes[1]["weight"] - graph_isolated.nodes[0]["weight"]
        )

    assert all(test_instance.kernel.has_node(node) for node in [2, 3, 4, 5])
    test_mis = test_instance.rebuild(frozenset({2}))
    assert test_mis == [frozenset({0, 2})]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
@pytest.mark.parametrize("graph", [g for g in graph_variants(graph_isolated)])
def test_search_rule_isolated_node_removal(
    weighting: Weighting, graph: tuple[GraphVariant, nx.Graph]
) -> None:
    variant, graph_isolated = graph
    test_instance = ker.Kernelization(SolverConfig(weighting=weighting), graph_isolated)._kernelizer
    test_instance.search_rule_isolated_node_removal()

    # {0} is isolated and maximal, so removing {0} and its neighborhood.
    assert not any(test_instance.kernel.has_node(node) for node in [0, 1])
    assert all(test_instance.kernel.has_node(node) for node in [2, 3, 4, 5])
    test_mis = test_instance.rebuild(frozenset({2}))
    assert test_mis == [frozenset({0, 2})]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_folding_uw(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=graph_folding
    )._kernelizer
    test_instance.fold_three(0, 1, 2, 9)
    assert not all(test_instance.kernel.has_node(node) for node in [0, 1, 2])
    assert all(test_instance.kernel.has_node(node) for node in [9, 3, 4, 5, 6, 7, 8])


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_apply_rule_node_fold_uw(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=graph_folding
    )._kernelizer
    test_instance.apply_rule_node_fold(v=0, w_v=1.0, u=1, w_u=1.0, x=2, w_x=1.0)
    assert not all(test_instance.kernel.has_node(node) for node in [0, 1, 2])
    assert all(test_instance.kernel.has_node(node) for node in [9, 3, 4, 5, 6, 7, 8])
    mis_1 = test_instance.rebuild(frozenset({9}))
    assert mis_1 == [frozenset({1, 2})]
    mis_2 = test_instance.rebuild(frozenset({4}))
    assert mis_2 == [frozenset({0, 4})]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_rule_node_fold(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=graph_folding
    )._kernelizer
    test_instance.search_rule_node_fold()
    assert not all(test_instance.kernel.has_node(node) for node in [0, 1, 2])
    assert all(test_instance.kernel.has_node(node) for node in [9, 3, 4, 5, 6, 7, 8])
    mis_1 = test_instance.rebuild(frozenset({9}))
    assert mis_1 == [frozenset({1, 2})]
    mis_2 = test_instance.rebuild(frozenset({4}))
    assert mis_2 == [frozenset({0, 4})]


def test_aux_search_confinement() -> None:
    # This operation exists only on unweighted kernelization.
    weighting = Weighting.UNWEIGHTED
    test_instance = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=P_3)
    confinement_1 = test_instance.aux_search_confinement({1}, {0})
    assert confinement_1 is not None
    assert confinement_1.node == 1
    assert confinement_1.set_diff == set({2})

    confinement_2 = test_instance.aux_search_confinement({1}, {0, 2})
    assert confinement_2 is None

    confinement_3 = test_instance.aux_search_confinement({2}, {0, 1})
    assert confinement_3 is not None
    assert confinement_3.node == 2
    assert confinement_3.set_diff == set()

    test_instance_2 = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=CLAW)
    confinement_4 = test_instance_2.aux_search_confinement({1}, {0})
    assert confinement_4 is not None
    assert confinement_4.node == 1
    assert confinement_4.set_diff == set({2, 3})


def test_apply_rule_unconfined() -> None:
    # This operation exists only on unweighted kernelization.
    weighting = Weighting.UNWEIGHTED
    test_instance = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=P_3)
    test_instance.apply_rule_unconfined(2)
    assert frozenset(test_instance.kernel.nodes) == frozenset({0, 1})


def test_unconfined_loop() -> None:
    # This operation exists only on unweighted kernelization.
    weighting = Weighting.UNWEIGHTED
    test_instance = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=P_3)
    assert test_instance.unconfined_loop(0, {0}, {1})
    assert not test_instance.unconfined_loop(0, {0, 1}, {2})
    test_instance_2 = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=CLAW)
    assert not test_instance_2.unconfined_loop(0, {0}, {1})
    test_instance_3 = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=K4)
    assert not test_instance_3.unconfined_loop(0, {0}, {1, 2, 3})


def test_search_rule_unconfined_and_diamond() -> None:
    # This operation exists only on unweighted kernelization.
    weighting = Weighting.UNWEIGHTED
    test_instance = ker.UnweightedKernelization(SolverConfig(weighting=weighting), graph=K3_CLAW)
    test_instance.search_rule_unconfined_and_diamond()
    assert frozenset(test_instance.kernel) == frozenset({2, 4, 5})


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_fold_twin_uw(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_bis
    )._kernelizer
    test_instance.fold_twin(0, 1, 10, [2, 3, 4])
    assert not all(test_instance.kernel.has_node(node) for node in [0, 1, 2, 3, 4])
    assert all(test_instance.kernel.has_node(node) for node in [10, 5, 6])
    N_v_prime = frozenset(test_instance.kernel.neighbors(10))
    assert N_v_prime == frozenset({5, 6})


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_find_removable_twin(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_bis
    )._kernelizer
    twin_1 = test_instance.find_removable_twin(0)
    assert twin_1 is not None
    assert twin_1.category == "INDEPENDENT"
    test_instance2 = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_twin_linked
    )._kernelizer
    assert test_instance2.find_removable_twin(0) is None


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_apply_rule_twin_ind(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_bis
    )._kernelizer
    test_instance.apply_rule_twin_independent(0, 1, [2, 3, 4])
    assert frozenset(test_instance.kernel.nodes()) == frozenset({5, 6, 7})
    mis_1 = test_instance.rebuild(frozenset({7}))
    assert mis_1 == [frozenset({2, 3, 4})]
    mis_2 = test_instance.rebuild(frozenset())
    assert mis_2 == [frozenset({0, 1})]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_apply_rule_twin_not_ind(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_N_linked
    )._kernelizer
    test_instance.apply_rule_twins_in_solution(0, 1, [2, 3, 4])
    assert frozenset(test_instance.kernel.nodes()) == frozenset({5, 6})
    mis_1 = test_instance.rebuild(frozenset())
    assert mis_1 == [frozenset({0, 1})]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_search_rule_twin_reduction(weighting: Weighting) -> None:
    test_instance = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_bis
    )._kernelizer
    test_instance.search_rule_twin_reduction()
    assert frozenset(test_instance.kernel.nodes()) == frozenset({5, 6, 7})
    mis_1 = test_instance.rebuild(frozenset({7}))
    assert mis_1 == [frozenset({2, 3, 4})]
    mis_2 = test_instance.rebuild(frozenset())
    assert mis_2 == [frozenset({0, 1})]
    test_instance_2 = ker.Kernelization(
        SolverConfig(weighting=weighting), graph=K23_CLAW_N_linked
    )._kernelizer
    test_instance_2.apply_rule_twins_in_solution(0, 1, [2, 3, 4])
    assert frozenset(test_instance_2.kernel.nodes()) == frozenset({5, 6})
    mis_3 = test_instance_2.rebuild(frozenset())
    assert mis_3 == [frozenset({0, 1})]


def test_kernelization_does_not_eat_solutions(python_dependency_graph: MISInstance) -> None:
    """Test for issue 135: kernelization should not eat solutions."""
    instance = python_dependency_graph
    solver = MISSolver(
        instance,
        config=SolverConfig(
            preprocessor=lambda config, graph: ker.Kernelization(config, graph),
            backend=BackendConfig(),
            max_number_of_solutions=10,
        ),
    )
    solutions = solver.solve()

    assert len(solutions) == 2
    expected = [
        {"mygreatlib", "anotherlib", "Python 3.10"},
        {"mygreatlib", "anotherlib", "Python 3.13"},
    ]
    for nodes in expected:
        found = False
        for sol in solutions:
            if set(sol.nodes) == nodes:
                found = True
                break
        assert found, f"Found solution {nodes} among {[set(sol.nodes) for sol in solutions]}"
