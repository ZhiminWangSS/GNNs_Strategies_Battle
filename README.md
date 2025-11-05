# GNN Parallel Strategies Battle

Compare different parallel training strategies (GDP, NFP, SNP, DNP) for Graph Neural Networks on OGBN-arxiv and OGBL-ddi datasets.

## Setup

```bash
conda create -n dgl_torch python=3.12.1 -y
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install scikit-learn
```

## Graph Generator
- Supports three types of graphs: **ER**, **BA**, and **SBM**, with hyperparameters to control structural properties.  
- Supports two partitioning methods: **Metis** and **Random**. It is recommended to save the partitioned graphs in a `tmp` folder.
- Provides DataLoaders for specified subgraphs, ready for **node classification** and **link prediction** tasks. Recommended to initialize one DataLoader per process, and make sure `GPU num = subgraph num = process num`.

### Run an example
```bash
cd GNNs_Strategies_Battle
python datasets/data_generator.py
```

## Project Structure	

```
├── dataset_generator # generator.py # linjia
├── models/           # GCN model.py
├── training/       # sampling strategies for train and test 
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