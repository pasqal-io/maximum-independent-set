# Usage

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
