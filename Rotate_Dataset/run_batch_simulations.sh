#!/bin/bash

# 设置输出根目录
OUTPUT_SIMPLE_DIR="/home/zh/fueltank_datasets/datasets/datasets_simple_rotate"
OUTPUT_16_DIR="/home/zh/fueltank_datasets/datasets/datasets_16_rotate"
OUTPUT_69R_DIR="/home/zh/fueltank_datasets/datasets/datasets_69r_rotate"
OUTPUT_911_DIR="/home/zh/fueltank_datasets/datasets/datasets_911_rotate"
COMPRESSED_OUTPUT_911_DIR="/home/zh/fueltank_datasets/compressed_datasets/911_rotate"
COMPRESSED_OUTPUT_16_DIR="/home/zh/fueltank_datasets/compressed_datasets/16_rotate"
COMPRESSED_OUTPUT_69R_DIR="/home/zh/fueltank_datasets/compressed_datasets/69r_rotate"
COMPRESSED_OUTPUT_SIMPLE_DIR="/home/zh/fueltank_datasets/compressed_datasets/simple_rotate"

echo "开始运行批量仿真..."

# 1. 运行 simple 仿真
echo "--------------------------------------------------"
echo "正在运行 batch_simulation_simple_rotate.py ..."
python batch_simulation_simple_rotate.py --output "$OUTPUT_SIMPLE_DIR"
if [ $? -ne 0 ]; then
    echo "batch_simulation_simple_rotate.py 运行失败"
    exit 1
fi
# 2. 运行 16r 仿真
echo "--------------------------------------------------"
echo "正在运行 batch_simulation_16r_rotate.py ..."
python batch_simulation_16r_rotate.py --output "$OUTPUT_16_DIR"
if [ $? -ne 0 ]; then
    echo "batch_simulation_16r_rotate.py 运行失败"
    exit 1
fi

# 3. 运行 69r 仿真
echo "--------------------------------------------------"
echo "正在运行 batch_simulation_69r_rotate.py ..."
python batch_simulation_69r_rotate.py --output "$OUTPUT_69R_DIR"
if [ $? -ne 0 ]; then
    echo "batch_simulation_69r_rotate.py 运行失败"
    exit 1
fi

# 4. 运行 911 仿真
echo "--------------------------------------------------"
echo "正在运行 batch_simulation_911_rotate.py ..."
python batch_simulation_911_rotate.py --output "$OUTPUT_911_DIR"
if [ $? -ne 0 ]; then
    echo "batch_simulation_911_rotate.py 运行失败"
    exit 1
fi

echo "正在压缩 911 数据..."
python compress_dataset_rotationsequence_fueltank.py --dataset-dir "$OUTPUT_911_DIR" --output "$COMPRESSED_OUTPUT_911_DIR"
echo "正在压缩 16 数据..."
python compress_dataset_rotationsequence_fueltank.py --dataset-dir "$OUTPUT_16_DIR" --output "$COMPRESSED_OUTPUT_16_DIR"
echo "正在压缩 69r 数据..."
python compress_dataset_rotationsequence_fueltank.py --dataset-dir "$OUTPUT_69R_DIR" --output "$COMPRESSED_OUTPUT_69R_DIR"
echo "正在压缩 simple 数据..."
python compress_dataset_rotationsequence_fueltank.py --dataset-dir "$OUTPUT_SIMPLE_DIR" --output "$COMPRESSED_OUTPUT_SIMPLE_DIR"

echo "--------------------------------------------------"
echo "所有仿真任务已完成！"
