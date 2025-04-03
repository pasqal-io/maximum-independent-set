# maximum independent set


The **Maximum Independent Set (MIS)** library provides a flexible, powerful, and user-friendly Python interface for solving Maximum Independent Set and Weighted Maximum Independent Set problems. It is designed for **scientists and engineers** working on optimization problems—**no quantum computing knowledge required**.

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
from mis import MISSolver, MISInstance
import networkx as nx

# Generate a simple graph (triangle)
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2)])

# Create an instance for the solver
instance = MISInstance(graph)

# Solve the MIS problem
results = MISSolver.solve(instance)

print("MIS solution:", results.solution)
print("Solution cost:", results.cost)
```

## Documentation

We have a two parts tutorial:

1. [Using a Quantum Device to solve MIS](https://pasqal-io.github.io/maximum-independent-setl/blob/main/examples/tutorial%201a%20-%20Using%20a%20Quantum%20Device%20to%20solve%20MIS.ipynb)
2. [Example Use Case](https://pasqal-io.github.io/maximum-independent-setl/blob/main/examples/tutorial%201b%20-%20MIS%20Example%20Use%20Case.ipynb)
3. [Backend and Solver Configuration](https://pasqal-io.github.io/maximum-independent-setl/blob/main/examples/tutorial%202%20-%20Backend%20and%20Solver%20Configuration.ipynb)
4. [Sampling & Analysis](https://pasqal-io.github.io/maximum-independent-setl/blob/main/examples/tutorial%203%20-%20Sampling%20&%20Analysis.ipynb)


See also the [full API documentation](https://pasqal-io.github.io/maximum-independent-set/latest/).

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/maximum-independent-set) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
