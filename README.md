# Maximum independent set


The **Maximum Independent Set (MIS)** library provides a flexible, powerful, and user-friendly Python interface for solving [Maximum Independent Set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)) problem using Quantum technologies. It is designed for **scientists and engineers** working on optimization problems—**no quantum computing knowledge required** and **no quantum computer needed** for testing.

This library lets users treat the solver as a **black box**: feed in a graph, get back an optimal (or near-optimal) independent set. For more advanced users, it offers tools to **fine-tune algorithmic strategies**, leverage **quantum hardware** via the Pasqal cloud, or even **experiment with custom quantum sequences** and processing pipelines.

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
from mis import MISSolver, MISInstance, SolverConfig
from mis.pipeline.backends import QutipBackend
import networkx as nx

# Generate a simple graph (here, a triangle)
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2)])
instance = MISInstance(graph)

# Use a quantum solver.
config = SolverConfig(backend=QutipBackend())
solver = MISSolver(instance, config)

# Solve the MIS problem
results = solver.solve().result()

print("MIS solutions:", results)
```

## Documentation

[Using a Quantum Device to solve MIS](https://pasqal-io.github.io/maximum-independent-setl/blob/main/examples/tutorial%201a%20-%20Using%20a%20Quantum%20Device%20to%20solve%20MIS.ipynb)


See also the [full API documentation](https://pasqal-io.github.io/maximum-independent-set/latest/).

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/maximum-independent-set) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
