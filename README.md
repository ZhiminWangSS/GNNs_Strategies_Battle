# GNNs_Strategies_Battle

### code structure

- utils
  - model.py # GCN class
  - training.py
  - test.py
- dataset
  - OGBN-arxiv
  - OGBL-ddi
- result
  - GDP/
  - NFP/
  - SNP/
  - DNP/





### Stage 1 Infrastructure construction

- model: GCN
- dataset:
  - OGBN-arxiv [link](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
  - OGBL-ddi [link](https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi)
- distributed strategy:
  - GDP
  - NFP
  - SNP
  - DNP
- metrics: 
  - Communication Overhead
    - Average epoch time; GPU communication time; Communication Ratio
  - Computational Overhead
    - Maximum memory usage, GPU utilization ratio
  - Convergence
    - Number of epochs to convergence, Loss curve
  - Performance
    - Accuracy, ROC-AUC



### Stage 2 training and analysis

- distributed strategy:
  - GDP
  - NFP
  - SNP
  - DNP
- GPU nums:
  - 1
  - 2
  - 4

total 4 * 3 = 12 training  and be divided to 4 people