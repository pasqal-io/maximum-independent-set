[![PyPI version](https://badge.fury.io/py/maximum-independent-set.svg)](https://pypi.org/project/maximum-independent-set/)
[![Tests](https://github.com/pasqal-io/maximum-independent-set/actions/workflows/test.yml/badge.svg)](https://github.com/pasqal-io/maximum-independent-set/actions/workflows/test.yml)
![Coverage](https://img.shields.io/codecov/c/github/pasqal-io/maximum-independent-set?style=flat-square)


# Maximum independent set

The [Maximum Independent Set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)) problem (MIS) is a standard and widespread graph problem in scheduling, network theory, error correction, and even in the quantum sector as part of more general optimization algorithms (e.g., QUBO formulations) or as a benchmark on quantum annealers or neutral atom devices.

There is currently no known polynomial-time algorithm for general graphs running on classical (non-quantum) devices, which means that, in practice, finding an exact solution for large graphs is generally not possible due to time and hardware limitations. For this reason, most applications of MIS must satisfy themselves with finding approximate solutions. As it turns out, in some cases, even finding approximate solutions is considered hard. For these reasons, there is high interest in solving MIS on quantum devices.

The maximum-independent-set library provides the means to achieve this: it compiles an MIS into a form suited for execution on existing analog quantum hardware, such as the commercial QPUs produced by Pasqal. It is designed for **scientists and engineers** working on optimization problems—**no quantum computing knowledge required** and **no quantum computer needed** for testing.

This library lets users treat the solver as a **black box**: feed in a graph of interest, get back an optimal (or near-optimal) independent set. For more advanced users, it offers tools to **fine-tune algorithmic strategies**, leverage **quantum hardware** via the Pasqal cloud, or even **experiment with custom quantum sequences** and processing pipelines.

Users setting their first steps into quantum computing will learn how to implement the core algorithm in a few simple steps and run it using the Pasqal Neutral Atom QPU. More experienced users will find this library to provide the right environment to explore new ideas - both in terms of methodologies and data domain - while always interacting with a simple and intuitive QPU interface.

## Installation

### Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```
  "maximum-independent-set"
```

to the list of `dependencies`.

### Using `pip` or `pipx`
To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
$ python -m venv venv

```

2. Enter the venv

```sh
$ . venv/bin/activate
```

3. Install the package

```sh
$ pip install maximum-independent-set
# or
$ pipx install maximum-independent-set
```

## QuickStart

```python
from mis import MISSolver, MISInstance, BackendConfig, SolverConfig
import networkx as nx

# Generate a simple graph (here, a triangle)
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2)])
instance = MISInstance(graph)

# Use a default quantum backend.
config = SolverConfig(backend=BackendConfig())
solver = MISSolver(instance, config)

# Solve the MIS problem.
results = solver.solve()

# Show the results.
print("MIS solutions:", results)
results[0].draw()
```

## Documentation

[Documentation](https://pasqal-io.github.io/maximum-independent-set/latest/)

[Tutorials](https://pasqal-io.github.io/maximum-independent-set/latest/tutorial%201a%20-%20Using%20a%20Quantum%20Device%20to%20solve%20MIS/).

[Full API documentation](https://pasqal-io.github.io/maximum-independent-set/latest/api/mis/).

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/maximum-independent-set) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
