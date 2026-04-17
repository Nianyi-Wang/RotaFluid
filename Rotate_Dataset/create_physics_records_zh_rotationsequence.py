#!/usr/bin/env python3
"""This script creates compressed records for training the network"""
import os
import sys
import json
import argparse
import numpy as np
from glob import glob
import struct

from create_physics_scenes_fueltank import PARTICLE_RADIUS
from physics_data_helper import *
from compare_rotation import read_rb_bin_matrices, calculate_rotation_matrices, load_json_config  # 导入计算旋转矩阵的函数


def create_scene_files(scene_dir, scene_id, outfileprefix, splits=16):

    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
        scene_dict = json.load(f)

    # dynamic_box_bgeo_path = "/home/zh/fueltank_datasets/models/fueltank/dynamic_box.bgeo"
    # box, box_normals = numpy_from_bgeo(dynamic_box_bgeo_path)
    box, box_normals = numpy_from_bgeo(os.path.join(scene_dir, 'box.bgeo'))
    
    # 获取刚体数据目录
    rigid_body_dir = os.path.join(scene_dir, 'rigid_bodies')
    
    partio_dir = os.path.join(scene_dir, 'partio')
    fluid_ids = get_fluid_ids_from_partio_dir(partio_dir)
    num_fluids = len(fluid_ids)
    fluid_id_bgeo_map = {
        k: get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids
    }

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = list(range(len(v)))
        if len(v) != len(frames):
            raise Exception(
                'number of frames for fluid {} ({}) is different from {}'.
                format(k, len(v), len(frames)))

    sublists = np.array_split(frames, splits)

    boring = False  # no fluid and rigid bodies dont move
    last_max_velocities = [1] * 20

    # 预计算box和box_normals的数据类型转换，只在第一帧使用
    box_float32 = box.astype(np.float32)
    box_normals_float32 = box_normals.astype(np.float32)
    
    # 预处理流体参数
    fluid_params = {
        flid: {
            'viscosity': scene_dict[flid]['viscosity'],
            'density': scene_dict[flid]['density0']
        }
        for flid in fluid_ids
    }

    def standardize_target_sequence(target_sequence):
        """标准化目标序列，确保所有序列具有相同的结构
        Args:
            target_sequence: 原始序列
        Returns:
            标准化后的序列
        """
        if len(target_sequence) == 8:  # 短序列
            # 添加最后的时间点和速度
            return target_sequence + [6.0, 0.0]
        return target_sequence
    
    # 在create_scene_files函数中修改
        # 读取旋转配置
    scene_json_path = os.path.join(scene_dir, 'scene.json')
    axis, target_sequence = load_json_config(scene_json_path)
        # 标准化序列
    target_sequence = standardize_target_sequence(target_sequence)
    
    for sublist_i, sublist in enumerate(sublists):
        if boring:
            break

        outfilepath = outfileprefix + '_{0:02d}.msgpack.zst'.format(sublist_i)
        if not os.path.isfile(outfilepath):
            # 预分配数据列表
            data = [None] * len(sublist)
            
            # 一次性读取所有旋转矩阵
            all_positions, all_rotations = read_rb_bin_matrices(rigid_body_dir)
            
            for i, frame_i in enumerate(sublist):
                feat_dict = {}
                
                # 使用预先读取的数据
                pos = all_positions[frame_i]
                rot_matrix = all_rotations[frame_i]
                
                # 在第一帧中添加旋转配置
                if frame_i == sublist[0]:
                    feat_dict['box'] = box.astype(np.float32)
                    feat_dict['box_normals'] = box_normals.astype(np.float32)
                    feat_dict['rotation_axis'] = np.array(axis, dtype=np.float32)
                    feat_dict['rotation_sequence'] = np.array(target_sequence, dtype=np.float32)
                
                # 所有帧都保存位置和旋转矩阵
                feat_dict['box_pos'] = np.array(pos, dtype=np.float32)
                feat_dict['box_rot'] = rot_matrix.astype(np.float32)
                feat_dict['frame_id'] = np.int64(frame_i)
                feat_dict['scene_id'] = scene_id

                pos = []
                vel = []
                mass = []
                viscosity = []

                sizes = np.array([0, 0, 0, 0], dtype=np.int32)

                # 优化流体数据处理
                pos_list = []
                vel_list = []
                mass_list = []
                visc_list = []
                total_size = 0

                for flid in fluid_ids:
                    bgeo_path = fluid_id_bgeo_map[flid][frame_i]
                    pos_, vel_ = numpy_from_bgeo(bgeo_path)
                    particles = pos_.shape[0]
                    total_size += particles
                    
                    pos_list.append(pos_)
                    vel_list.append(vel_)
                    visc_list.append(np.full(particles, fluid_params[flid]['viscosity']))
                    mass_list.append(np.full(particles, fluid_params[flid]['density']))

                # 一次性连接所有数组
                feat_dict['pos'] = np.concatenate(pos_list, axis=0).astype(np.float32)
                feat_dict['vel'] = np.concatenate(vel_list, axis=0).astype(np.float32)
                feat_dict['viscosity'] = np.concatenate(visc_list)
                feat_dict['m'] = np.concatenate(mass_list) * (2 * PARTICLE_RADIUS)**3

                # 使用正确的索引i存储数据
                data[i] = feat_dict

            create_compressed_msgpack(data, outfilepath)


def create_compressed_msgpack(data, outfilepath):
    import zstandard as zstd
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()

    compressor = zstd.ZstdCompressor(level=22)
    with open(outfilepath, 'wb') as f:
        print('writing', outfilepath)
        f.write(compressor.compress(msgpack.packb(data, use_bin_type=True)))


def compress_single_scene(input_dir, output_dir, prefix_name, splits=16):
    """外部调用函数，压缩单个场景目录
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        prefix_name: 保存文件的前缀名
        splits: 分片数量，默认16
    """
    os.makedirs(output_dir, exist_ok=True)
    outfileprefix = os.path.join(output_dir, prefix_name)
    create_scene_files(input_dir, os.path.basename(input_dir), outfileprefix, splits)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Creates compressed msgpacks for directories with SplishSplash scenes")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to the input directory with the simulation data")
    parser.add_argument(
        "--splits",
        type=int,
        default=16,
        help="The number of files to generate per scene (default=16)")

    args = parser.parse_args()
    os.makedirs(args.output)

    outdir = args.output

    scene_dirs = sorted(glob(os.path.join(args.input, '*')))
    print(scene_dirs)

    for scene_dir in scene_dirs:
        print(scene_dir)
        scene_name = os.path.basename(scene_dir)
        print(scene_name)
        outfileprefix = os.path.join(outdir, scene_name)
        create_scene_files(scene_dir, scene_name, outfileprefix, args.splits)


if __name__ == '__main__':
    main()
