# GNN Parallel Strategies Battle

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
- Supports two partitioning methods: **Metis** and **Random**. It is recommended to save the partitioned graphs in a `tmp` folder.
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

## Training Guide
- 代码逻辑：
  - 调用` data_generator.py`  的 `GraphGenerator` 类生成图数据集。
  - 初始化分布式训练环境
  - 分布式训练和指标记录

- 数据集生成：
  - 参数和保存路径可以在prepare_graph函数中调整，没有来及写好统一的接口，如果有空可以自己写一个传参接口


```

- node classifation
train_node.py
python train_node.py

- link prediction
train_link.py
python train_link.py
```
- 训练结果可以通过tensorboard查看，默认保存在`./training/runs`文件夹下，命名为`{task}_{time}`，通过以下指令启动查看。

```py
tensorboard --logdir ./runs
```