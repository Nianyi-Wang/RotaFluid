#!/usr/bin/env python3
import os
import sys
import json
import msgpack
import msgpack_numpy
import zstandard as zstd
from pathlib import Path
from glob import glob
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from create_physics_records_zh_rotationsequence import compress_single_scene  # 使用新的压缩函数
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def get_simulation_params(path):
    """从路径中提取仿真参数 (适配 batch_simulation_2 动态模型目录结构)"""
    parts = path.parts
    # time
    time_part = next(p for p in parts if p.startswith('time_'))
    time = time_part.split('_')[1].replace('s', '')
    # angles
    angle_part = next(p for p in parts if p.startswith('roll_'))
    roll_pitch = angle_part.split('_')
    roll = roll_pitch[1]
    pitch = roll_pitch[3]
    # fluid
    fluid_part = next(p for p in parts if p.startswith('fluid_'))
    fluid = fluid_part
    # sequence
    seq_part = next(p for p in parts if p.startswith('constant_'))
    seq_type = 'A' if seq_part == 'constant_acceleration' else 'C'
    # dynamic model
    dynamic_part = next((p for p in parts if p.startswith('dynamic_')), None)
    dynamic_model = dynamic_part[len('dynamic_'):] if dynamic_part else 'default'
    # sim id
    sim_part = next((p for p in parts if p.startswith('sim_')), None)
    sim_id = sim_part.split('_')[1] if sim_part else '0042'
    
    print(f"解析参数: dynamic={dynamic_model}, time={time}, roll={roll}, pitch={pitch}, fluid={fluid}, type={seq_type}, sim={sim_id}")
    return time, roll, pitch, fluid, seq_type, dynamic_model, sim_id

def process_single_dir(output_path, splits, sim_dir, num_threads=8):
    """处理单个目录的函数"""
    if not sim_dir.is_dir():
        return None, str(sim_dir)
        
    try:
        # 获取仿真参数 (新增 dynamic_model 和 sim_id)
        time, roll, pitch, fluid, seq_type, dynamic_model, sim_id = get_simulation_params(sim_dir)
        
        # 更新前缀，加入动态模型和sim_id避免冲突
        prefix_name = f"sim{sim_id}_D{dynamic_model}_T{time}_R{roll}_P{pitch}_{fluid}_{seq_type}"
        
        # sim_dir 已经是 sim_XXXX 目录，不需要再拼接
        if not sim_dir.exists():
            print(f"跳过不存在的目录: {sim_dir}")
            return None, str(sim_dir)
            
        # 使用线程池调用压缩函数
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future = executor.submit(
                compress_single_scene,
                input_dir=str(sim_dir),
                output_dir=str(output_path),
                prefix_name=prefix_name,
                splits=splits
            )
            future.result()  # 等待执行完成
            
        print(f"已处理: {sim_dir.name}")
        return str(sim_dir), None
        
    except Exception as e:
        print(f"处理失败: {sim_dir}, 错误: {str(e)}")
        return None, str(sim_dir)

def batch_compress_dataset(dataset_dir, output_dir, splits=16):
    """批量压缩数据集"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有需要处理的目录 (直接查找 sim_ 开头的目录)
    sim_dirs = list(dataset_path.rglob('sim_*'))
    # 过滤掉非目录
    sim_dirs = [d for d in sim_dirs if d.is_dir()]
    
    # 调整进程数和优先级
    num_cores = mp.cpu_count()
    print(f"系统总核心数: {num_cores}")
    max_workers = num_cores - 4  # 使用更多核心
    
    # 设置进程优先级
    import os
    os.nice(10)  # 提高进程优先级
    
    # 使用maxtasksperchild限制内存增长
    pool = mp.Pool(max_workers, maxtasksperchild=1)
    
    print(f"使用 {max_workers} 个进程进行并行处理...")
    
    # 创建偏函数，固定output_path和splits参数
    process_func = partial(process_single_dir, output_path, splits)
    
    # 并行处理所有目录
    
    # 获取所有需要处理的目录 (直接查找 sim_ 开头的目录)
    sim_dirs = list(dataset_path.rglob('sim_*'))
    # 过滤掉非目录
    sim_dirs = [d for d in sim_dirs if d.is_dir()]
    batch_size = max(len(sim_dirs) // max_workers, 1)  # 每个进程处理的目录数
    
    # 分批处理目录
    processed_dirs = []
    skipped_dirs = []
    
    # 使用position参数固定进度条位置
    with tqdm(total=len(sim_dirs), desc="处理进度", position=0, leave=True) as pbar:
        for i in range(0, len(sim_dirs), batch_size):
            batch = sim_dirs[i:i + batch_size]
            results = pool.map(process_func, batch)
            
            # 处理每批结果并立即更新列表
            for processed, skipped in results:
                if processed:
                    processed_dirs.append(processed)
                    pbar.set_postfix({'成功': len(processed_dirs)})
                if skipped:
                    skipped_dirs.append(skipped)
                    pbar.set_postfix({'成功': len(processed_dirs), '跳过': len(skipped_dirs)})
            
            # 更新进度
            pbar.update(len(batch))
            
      
    
    pool.close()
    pool.join()
    
    # 输出最终统计信息
    print("\n处理完成！统计信息：")
    print(f"成功处理目录数: {len(processed_dirs)}")
    print(f"跳过目录数: {len(skipped_dirs)}")
    
    if skipped_dirs:
        print("\n以下目录被跳过：")
        for dir_path in skipped_dirs:
            print(f"- {dir_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量压缩数据集")
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--output", type=str, required=True, help="压缩输出目录")
    parser.add_argument("--splits", type=int, default=16, help="每个仿真的分片数量")
    
    args = parser.parse_args()
    
    batch_compress_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output,
        splits=args.splits
    )

if __name__ == '__main__':
    sys.exit(main())