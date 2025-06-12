# Usage

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

# Solve the MIS problem.
results = solver.solve().result()

# Show the results.
print("MIS solutions:", results)
results[0].draw()
```
