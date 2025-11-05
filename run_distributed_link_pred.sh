#!/bin/bash

# 单机多卡分布式训练启动脚本
# 使用 torchrun 启动多GPU训练

echo "🚀 启动单机多卡分布式边预测训练..."

# 检查是否有可用的GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未检测到 NVIDIA GPU，请检查CUDA环境"
    exit 1
fi

# 获取GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "📊 检测到 $NUM_GPUS 个GPU"

# 设置环境变量
export PYTHONPATH="/home/zmwang/storage/codes/GNNs_Strategies_Battle:$PYTHONPATH"

# 使用torchrun启动分布式训练
echo "🎯 启动分布式训练..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    training.py/train_link.py

echo "✅ 分布式训练完成！"