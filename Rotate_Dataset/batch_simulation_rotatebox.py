#!/usr/bin/env python3
import os
import sys
import numpy as np
from glob import glob
from calculate_rotation_normal import calculate_normal_from_angles, calculate_rotation_sequences
from create_physics_fueltank_zh_rotatebox import create_custom_simulation

def batch_run_simulations(fluid_models_dir, dynamic_models_dir, output_base_dir):
    """批量运行物理仿真"""
    # 定义参数范围
    num_repeats = 5
    rotation_times = [3.0]
    # 仅绕X轴旋转
    roll_angles = [0]
    pitch_angles = [30, 45, 60]
    sequences = {
        'constant_velocity': 0,
    }
    # rotation_times = [1.0]
    # roll_angles = [0]
    # pitch_angles = [0]
    # sequences = {
    #     'constant_velocity': 0,
    # }
    
    # 获取所有动态模型文件
    dynamic_models = sorted(glob(os.path.join(dynamic_models_dir, '*.obj')))
    if not dynamic_models:
        print(f"错误：在 {dynamic_models_dir} 中未找到动态模型文件")
        return
    
    # 获取所有流体模型文件
    fluid_models = sorted(glob(os.path.join(fluid_models_dir, '*.obj')))
    if not fluid_models:
        print(f"错误：在 {fluid_models_dir} 中未找到流体模型文件")
        return
        
    total_sims = len(rotation_times) * len(roll_angles) * len(pitch_angles) * len(fluid_models) * len(sequences) * len(dynamic_models) * num_repeats
    current_sim = 0
    
    # 找到最后一个已完成的仿真
    last_completed = None
    
    # 第一次遍历找到最后一个已完成的仿真
    for dynamic_model in dynamic_models:
        dynamic_model_name = os.path.splitext(os.path.basename(dynamic_model))[0]
        dynamic_dir = os.path.join(output_base_dir, f'dynamic_{dynamic_model_name}')
        for time in rotation_times:
            time_dir = os.path.join(dynamic_dir, f'time_{time}s')
            for roll in roll_angles:
                for pitch in pitch_angles:
                    angle_dir = os.path.join(time_dir, f'roll_{roll}_pitch_{pitch}')
                    for fluid_model in fluid_models:
                        model_name = os.path.splitext(os.path.basename(fluid_model))[0]
                        model_dir = os.path.join(angle_dir, model_name)
                        for seq_name, seq_type in sequences.items():
                            sim_output_dir_base = os.path.join(model_dir, seq_name)
                            for run_idx in range(num_repeats):
                                seed = 42 + run_idx
                                expected_sim_dir = os.path.join(sim_output_dir_base, f'sim_{seed:04d}')
                                if os.path.exists(expected_sim_dir) and len(os.listdir(expected_sim_dir)) > 0:
                                    last_completed = expected_sim_dir

    # 正常的仿真循环
    for dynamic_model in dynamic_models:
        dynamic_model_name = os.path.splitext(os.path.basename(dynamic_model))[0]
        dynamic_dir = os.path.join(output_base_dir, f'dynamic_{dynamic_model_name}')
        
        print(f"\n处理动态模型: {dynamic_model_name}")
        
        for time in rotation_times:
            time_dir = os.path.join(dynamic_dir, f'time_{time}s')
            for roll in roll_angles:
                for pitch in pitch_angles:
                    # 获取旋转法向量和序列
                    normal = calculate_normal_from_angles(roll, pitch)
                    const_vel_seq, const_acc_seq, _, _ = calculate_rotation_sequences(pitch, roll, time=time, t_start=0.0)
                    
                    # 创建角度组合目录
                    angle_dir = os.path.join(time_dir, f'roll_{roll}_pitch_{pitch}')
                    
                    # 处理每个流体模型
                    for fluid_model in fluid_models:
                        model_name = os.path.splitext(os.path.basename(fluid_model))[0]
                        model_dir = os.path.join(angle_dir, model_name)
                        
                        for seq_name, seq_type in sequences.items():
                            sim_output_dir_base = os.path.join(model_dir, seq_name)
                            
                            for run_idx in range(num_repeats):
                                current_sim += 1
                                seed = 42 + run_idx
                                expected_sim_dir = os.path.join(sim_output_dir_base, f'sim_{seed:04d}')
                                
                                # 检查是否是最后一个已完成的仿真
                                if expected_sim_dir == last_completed:
                                    import shutil
                                    shutil.rmtree(expected_sim_dir)
                                    print(f"删除最后一个已完成的仿真: {expected_sim_dir}")
                                elif os.path.exists(expected_sim_dir) and len(os.listdir(expected_sim_dir)) > 0:
                                    print(f"跳过已完成的仿真: 动态模型={dynamic_model_name}, 时间={time}s, 横滚={roll}°, 俯仰={pitch}°, 流体模型={model_name}, 序列={seq_name}, 运行={run_idx}")
                                    continue
                                    
                                print(f"\n进度: [{current_sim}/{total_sims}]")
                                print(f"处理: 动态模型={dynamic_model_name}, 时间={time}s, 横滚={roll}°, 俯仰={pitch}°")
                                print(f"流体模型: {model_name}, 序列: {seq_name}, 运行: {run_idx}")
                                
                                sequence = const_vel_seq if seq_type == 0 else const_acc_seq
                                # 运行仿真
                                create_custom_simulation(
                                    dynamic_model_path=dynamic_model,
                                    rotation_sequence=sequence,
                                    normal_vector=normal,
                                    fluid_model_path=fluid_model,
                                    output_dir=sim_output_dir_base,
                                    seed=seed
                                )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量运行物理仿真")
    parser.add_argument("--fluid-dir", type=str, required=True, help="流体模型目录")
    parser.add_argument("--dynamic-dir", type=str, required=True, help="动态模型目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    batch_run_simulations(
        fluid_models_dir=args.fluid_dir,
        dynamic_models_dir=args.dynamic_dir,
        output_base_dir=args.output
    )

if __name__ == '__main__':
    sys.exit(main())