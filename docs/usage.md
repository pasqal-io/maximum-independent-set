# Usage

```python
from mis import MISSolver, MISInstance
from mis.config import MISConfig
from mis.pipeline.backends import QutipBackend
import networkx as nx

# Generate a simple graph (here, a triangle)
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2)])
instance = MISInstance(graph)

# Use a quantum solver.
config = MISConfig(backend=QutipBackend())

# Solve the MIS problem
results = MISSolver.solve(instance, config).result()

print("MIS solution:", results.solution)
print("Solution cost:", results.cost)
```
