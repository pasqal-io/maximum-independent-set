# Usage

```python
# Load a dataset
import torch_geometric.datasets as pyg_dataset
og_ptcfm = pyg_dataset.TUDataset(root="dataset", name="PTC_FM")

# Setup a quantum feature extractor for this dataset.
# In this example, we'll use QutipExtractor, to emulate a Quantum Device on our machine.
import mis.data.graphs as mis_graphs
import mis.data.extractors as mis_extractors
extractor = mis_extractors.QutipExtractor(compiler=mis_graphs.PTCFMCompiler())

# Add the graphs, compile them and look at the results.
extractor.add_graphs(graphs=og_ptcfm)
extractor.compile()
processed_dataset = extractor.run().processed_data

# Prepare a machine learning pipeline with Scikit Learn.
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = [data for data in processed_dataset]  # Features
y = [data.target for data in processed_dataset]  # Targets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)

# Train a kernel
from mis.kernel import QuantumEvolutionKernel as mis
kernel = mis(mu=0.5)
model = SVC(kernel=kernel, random_state=42)
model.fit(X_train, y_train)
```
