# Usage

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
results = solver.solve().result()

# Show the results.
print("MIS solutions:", results)
results[0].draw()
```
