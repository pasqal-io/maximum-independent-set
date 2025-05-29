from mis.data.graphs import load_dimacs
from mis.pipeline.kernelization import Kernelization


def test_hexagon():
    """
    Test loading the C6 from a DIMACS file.
    """
    dataset = load_dimacs("tests/data/dimacs/hexagon.txt")
    assert dataset.instance.graph.number_of_nodes() == 6
    assert dataset.instance.graph.number_of_edges() == 6
    assert len(dataset.solutions) > 0
    assert all(isinstance(solution, list) for solution in dataset.solutions)
    # C6 graph has a maximum independent set of size 3
    for solution in dataset.solutions:
        assert Kernelization(dataset.instance.graph).is_independent(solution)
        assert len(solution) <= 3


def slowtest_petersen():
    """
    Test loading the Petersen graph from a DIMACS file.
    """
    dataset = load_dimacs("tests/data/dimacs/petersen.txt")
    assert dataset.instance.graph.number_of_nodes() == 10
    assert dataset.instance.graph.number_of_edges() == 15
    assert len(dataset.solutions) > 0
    assert all(isinstance(solution, list) for solution in dataset.solutions)
    # Petersen graph has a maximum independent set of size 5
    for solution in dataset.solutions:
        assert Kernelization(dataset.instance.graph).is_independent(solution)
        assert len(solution) <= 5


def slowtest_a265032():
    """
    Test loading the Petersen graph from a DIMACS file.
    """
    dataset = load_dimacs("tests/data/dimacs/a265032_1dc.64.txt")
    assert dataset.instance.graph.number_of_nodes() == 64
    assert dataset.instance.graph.number_of_edges() == 543
    assert len(dataset.solutions) > 0
    assert all(isinstance(solution, list) for solution in dataset.solutions)
    # graph has a maximum independent set of size 10
    for solution in dataset.solutions:
        assert Kernelization(dataset.instance.graph).is_independent(solution)
        assert len(solution) <= 10
