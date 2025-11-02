# GNN Parallel Strategies Battle

Compare different parallel training strategies (GDP, NFP, SNP, DNP) for Graph Neural Networks on OGBN-arxiv and OGBL-ddi datasets.

## Setup

**Prerequisites:**
- Python 3.11
- [uv](https://github.com/astral-sh/uv) - Install with: `pip install uv`


**Tested on Linux**
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html # change bases on your cuda verison
uv sync
```

## Run Example

```bash
python gdp_example.py
```

## Project Structure

```
├── datasets/         # Dataset loaders (OGBN-arxiv, OGBL-ddi)
├── models/           # GCN model
├── strategies/       # Parallel strategies for train and test (GDP, NFP, SNP, DNP)
├── metrics/          # Training metrics (communication, convergence)
```

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



### Stage 2 Training and analysis

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