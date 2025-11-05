# Partition Strategies Battle

Compare different graph partitioning for graphs with differnt properties in terms of computational overhead and performance.

## Setup

```bash
conda create -n dgl_torch python=3.12.1 -y
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install scikit-learn
```

## Graph Generator
- Supports three types of graphs: **ER**, **BA**, and **SBM**, with hyperparameters to control structural properties.  
- Supports two partitioning methods: **Metis**, **Random** and **Direct**. It is recommended to save the partitioned graphs in a `tmp` folder.
- Provides DataLoaders for specified subgraphs, ready for **node classification** and **link prediction** tasks. Recommended to initialize one DataLoader per process, and make sure `GPU num = subgraph num = process num`.

### Run an example
```bash
cd GNNs_Strategies_Battle
python datasets/data_generator.py
```

## Project Structure	

```
├── datasets          # data_generator.py # linjia
├── models/           # gcn.py
├── training/         # scripts for parallel training
├── metrics/          # metrics
```