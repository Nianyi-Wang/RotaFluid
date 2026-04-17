#!/usr/bin/env python3
"""This script generates random fluid sequences with SPlisHSPlasH."""
import os
import argparse
import sys
import json
import numpy as np
from copy import deepcopy
import tempfile
import subprocess
from shutil import copyfile
import open3d as o3d
import math

from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN

SCRIPT_DIR = os.path.dirname(__file__)
PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

MAX_RIGID_START_VELOCITY_XZ = 2.0
MAX_RIGID_START_VELOCITY_Y = 2.0

# 配置参数
# 简单模型仿真12s，复杂模型15s
default_configuration = {
    "pause": False, 
    "particleRadius": 0.025,
    "stopAt": 6.0,
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 1000,
    "simulationMethod": 4,
    "gravitation": [0, -9.81, 0],
    "cflMethod": 1,
    "cflFactor": 1,
    "cflMaxTimeStepSize": 0.005,
    "maxIterations": 100,
    "maxError": 0.01,
    "maxIterationsV": 100,
    "maxErrorV": 0.1,
    "stiffness": 50000,
    "exponent": 7,
    "velocityUpdateMethod": 0,
    "enableDivergenceSolver": True,
    "enablePartioExport": True,
    "enableRigidBodyExport": True,
    "dataExportFPS": 50.0,
    "stateExportFPS": 50.0,
    "partioAttributes": "density;velocity",
    "boundaryHandlingMethod": 0
}
# boundaryHandlingMethod  0 ：粒子形成边界 ，2：sdf边界，油箱孔径小用0 ，2 生成更平滑噪声更小

default_simulation = {
    "contactTolerance": 0.0125
}

default_fluid = {
    "surfaceTension": 0.2,
    "surfaceTensionMethod": 0,
    "viscosity": 0.01,
    "viscosityMethod": 3,
    "viscoMaxIter": 200,
    "viscoMaxError": 0.05,
    "density0": 1000
}

default_rigidbody = {
    "translation": [0, 0, 0],
    "rotationAxis": [1, 0, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": True,
    "restitution": 0.6,
    "friction": 0.0,
    "mapThickness": 0.0,
    "mapResolution": [200, 200, 200],
    "mapInvert": True
}
    # "mapThickness": 0.0,
    # "mapResolution": [64, 64, 64] boundaryHandlingMethod为2时启用
# 油箱有厚度容器
default_joint = {
    "position": [0, 0, 0],
    "axis": [1, 0, 0],
    "repeatSequence": 0
}

# 无厚度容器
# default_joint = {
#     "position": [0, 2, 0],
#     "axis": [1, 0, 0],
#     "repeatSequence": 0
# }


# default_AnimationFields = [
#     {
#         "particleField": "velocity",
#         "expression_x": "",
#         "expression_y": "",
#         "expression_z": ""
#     }
# ]

def obj_volume_to_particles(objpath, scale=1, radius=None):
    """Convert obj file to particles"""
    if radius is None:
        radius = default_configuration["particleRadius"]
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        if isinstance(scale, (int, float)):
            scale_str = f"{scale},{scale},{scale}"
        else:
            scale_str = f"{scale[0]},{scale[1]},{scale[2]}"
        radius_str = str(radius)
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)

def obj_surface_to_particles(objpath, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    # Increase multiplier to improve sampling precision 4.0 
    # 1.9会略少导致粒子泄露，采用4.0可以避免
    num_points = int(2.2 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals

# def create_scene_with_joint(output_dir, seed, options):
#     """Creates a scene with rigid bodies, fluid, and a hinge joint"""
#     np.random.seed(seed)
    
#     sim_directory = os.path.join(output_dir, f'sim_{seed:04d}')
#     os.makedirs(sim_directory, exist_ok=False)

#     # 创建场景配置
#     scene = {
#         'Configuration': default_configuration,
#         'Simulation': default_simulation,
#         'RigidBodies': [],
#         'FluidModels': [],
#         'TargetVelocityMotorHingeJoints': [],
#         "AnimationFields": default_AnimationFields
#     }

#     # 添加边界盒子
#     box_obj = os.path.join(SCRIPT_DIR, 'models', 'fueltank', 'wallbox.obj')
#     # box_output_path = os.path.join(sim_directory, 'box.bgeo')
#     # bb, bb_normals = obj_surface_to_particles(box_obj)
#     # write_bgeo_from_numpy(box_output_path, bb, bb_normals)

#     box_obj_output_path = os.path.join(sim_directory, 'box.obj')
#     copyfile(box_obj, box_obj_output_path)

#     boundary = deepcopy(default_rigidbody)
#     boundary.update({
#         'id': 0,
#         'geometryFile': os.path.basename(os.path.abspath(box_obj_output_path)),
#         'scale': [1.0, 1.0, 1.0],
#         'isDynamic': False,
#         'isWall': True,
#         "mapResolution": [10, 10, 10]
#     })
#     scene['RigidBodies'].append(boundary)

#     # 添加动态刚体
#     dynamic_box_obj_path = os.path.join(SCRIPT_DIR, 'models', 'fueltank', 'box.obj')
#     dynamic_box_output_path = os.path.join(sim_directory, 'dynamic_box.obj')
#     copyfile(dynamic_box_obj_path, dynamic_box_output_path)
#     dynamic_box_bgeo_path =  os.path.join(sim_directory, 'box.bgeo')
#     # 为动态刚体生成bgeo文件
#     dynamic_bb, dynamic_bb_normals = obj_surface_to_particles(dynamic_box_obj_path)
#     write_bgeo_from_numpy(dynamic_box_bgeo_path, dynamic_bb, dynamic_bb_normals)
#     dynamic_body = deepcopy(default_rigidbody)
#     # 有厚度容器
#     dynamic_body.update({
#         'id': 1,
#         'geometryFile': os.path.basename(os.path.abspath(dynamic_box_output_path)),
#         'isDynamic': True,
#         'isWall': False,
#         'density':99999999999,
#         'mapInvert': True
#     })
 

#     # 无厚度容器
#     # dynamic_body.update({
#     #     'id': 1,
#     #     'geometryFile': os.path.basename(os.path.abspath(dynamic_box_output_path)),
#     #     'isDynamic': True,
#     #     'isWall': False,
#     #     'density': 99999999999,
#     #     'mapInvert': True
#     # })
#     scene['RigidBodies'].append(dynamic_body)

#     # 添加铰链关节
#     joint = deepcopy(default_joint)
#     joint.update({
#         'bodyID1': 1,
#         'bodyID2': 0,
#         'targetSequence': [0, 0, 3, 0, 6, 0]
#     })
#     scene['TargetVelocityMotorHingeJoints'].append(joint)

#     # 添加流体
#     fluid_obj = os.path.join(SCRIPT_DIR, 'models','fueltank', 'fluid_4.obj')
#     fluid_particles = obj_volume_to_particles(fluid_obj, scale=1)[0]
#     # 创建零速度数组
#     fluid_velocities = np.zeros_like(fluid_particles)
    
#     fluid_output_path = os.path.join(sim_directory, 'fluid0.bgeo')
#     write_bgeo_from_numpy(fluid_output_path, fluid_particles, fluid_velocities)
    
#     fluid_model = {
#         'translation': [0.0, 0.0, 0.0],
#         'scale': [1.0, 1.0, 1.0],
#         'id': 'fluid0',
#         'particleFile': 'fluid0.bgeo'  # 使用相对路径
#     }
#     scene['FluidModels'].append(fluid_model)

#     # 添加流体属性
#     scene['fluid0'] = default_fluid

#     # 保存场景配置
#     scene_output_path = os.path.join(sim_directory, 'scene.json')
#     with open(scene_output_path, 'w') as f:
#         json.dump(scene, f, indent=4)

#     # 运行模拟器
#     run_simulator(os.path.abspath(scene_output_path), sim_directory)

def apply_rotation(vertex, rotation_matrix):
    """Applies rotation to a vertex or normal."""
    v = np.array(vertex, dtype=np.float32)
    return rotation_matrix.dot(v)

def run_simulator(scene, output_dir):
    """运行模拟器"""
    status = subprocess.run([
        SIMULATOR_BIN,
        '--no-cache',
        '--no-gui',
        '--no-initial-pause',
        '--output-dir', output_dir,
        scene
    ])

def rotate_obj_file(input_filename, output_filename, rotation_matrix):
    """Rotate vertices and normals in an OBJ file and save to a new file."""
    # 创建输出文件所在的目录
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            if line.startswith('v ') or line.startswith('vn '):  # Process vertices and vertex normals
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                rotated_vertex = apply_rotation(vertex, rotation_matrix)
                file.write(f"{parts[0]} {rotated_vertex[0]} {rotated_vertex[1]} {rotated_vertex[2]}\n")
            elif line.startswith('vt '):  # Process texture coordinates (no rotation)
                file.write(line)
            elif line.startswith('f '):  # Process faces (no change)
                file.write(line)
            else:
                # Write other lines as-is
                file.write(line)
    return output_filename


def main():
    # parser = argparse.ArgumentParser(description="Creates physics sim data with joints")
    # parser.add_argument("--output", type=str, required=True,
    #                   help="The path to the output directory")
    # parser.add_argument("--seed", type=int, required=True,
    #                   help="The random seed for initialization")

    # args = parser.parse_args()
    # os.makedirs(args.output, exist_ok=True)
    # 设置参数
    dynamic_model = "/home/zh/fueltank_datasets/models/fueltank/tank911.obj"
    fluid_model = "/home/zh/fueltank_datasets/models/fueltank/fluid_50_911.obj"
    # rotation_seq =   [0.0,0.0, 1.0,0.0, 1.01,0.27596,2.0,0.27596,2.01,0.0]  # [t0, s0, t1, s1, t2, s2]
    rotation_seq =   [0.0,
                0.0,
                0.0,
                0.0,
                0.01,
                0.349066,
                3.0,
                0.349066,
                3.01,
                0.0]  # [t0, s0, t1, s1, t2, s2]
    normal_vec =  [1.0, 0.0, 0.0] # 旋转轴法向量
    output_dir = "/home/zh/fueltank_datasets/output/rotate_911_test_025"
    os.makedirs(output_dir, exist_ok=True)

    # 创建仿真并获取输出目录路径
    sim_dir = create_custom_simulation(
        dynamic_model_path=dynamic_model,
        rotation_sequence=rotation_seq,
        normal_vector=normal_vec,
        fluid_model_path=fluid_model,
        output_dir=output_dir,
        seed=42
    )
    # create_scene_with_joint(args.output, args.seed, args)

def create_custom_simulation(dynamic_model_path, rotation_sequence, normal_vector, 
                           fluid_model_path, output_dir, seed=0):
    """创建自定义物理仿真场景并运行
    
    Args:
        dynamic_model_path: 动态模型的obj文件路径
        rotation_sequence: 旋转序列 [t0, w0, t1, w1, t2, w2]
        normal_vector: 旋转法向量 [nx, ny, nz]
        fluid_model_path: 流体模型的obj文件路径
        output_dir: 输出目录
        seed: 随机种子
    """
    np.random.seed(seed)
    
    sim_directory = os.path.join(output_dir, f'sim_{seed:04d}')
    os.makedirs(sim_directory, exist_ok=False)

    # 创建场景配置
    scene = {
        'Configuration': default_configuration,
        'Simulation': default_simulation,
        'RigidBodies': [],
        'FluidModels': [],
        'TargetVelocityMotorHingeJoints': []
    }

    # 旋转矩阵
    random_angle_x = np.random.randint(-90, 91)
    random_angle_z = np.random.randint(-90, 91)
    random_angle_y = np.random.randint(-90, 91)
    angle_x = math.radians(random_angle_x)
    angle_y = math.radians(random_angle_y)
    angle_z = math.radians(random_angle_z)
    rand_R_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ]) #绕z轴
    rand_R_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ]) #绕x轴
    rand_R_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ]) #绕y轴
    rand_R = rand_R_x @ rand_R_y @ rand_R_z

    # 保存旋转信息
    rotation_info = {
        'seed': seed,
        'angles_deg': {
            'x': random_angle_x,
            'y': random_angle_y,
            'z': random_angle_z
        },
        'angles_rad': {
            'x': angle_x,
            'y': angle_y,
            'z': angle_z
        },
        'matrices': {
            'R_x': rand_R_x.tolist(),
            'R_y': rand_R_y.tolist(),
            'R_z': rand_R_z.tolist(),
            'R_combined': rand_R.tolist()
        }
    }
    with open(os.path.join(sim_directory, 'rotation_info.json'), 'w') as f:
        json.dump(rotation_info, f, indent=4)

    # 添加边界盒子
    box_obj = os.path.join(SCRIPT_DIR, 'models', 'fueltank', 'wallbox.obj')
    box_obj_output_path = os.path.join(sim_directory, 'wallbox.obj')
    # box_output_path = os.path.join(sim_directory, 'box.bgeo')
    # bb, bb_normals = obj_surface_to_particles(box_obj)
    # write_bgeo_from_numpy(box_output_path, bb, bb_normals)

    copyfile(box_obj, box_obj_output_path)
    boundary = deepcopy(default_rigidbody)
    boundary.update({
        'id': 0,
        'geometryFile': os.path.basename(os.path.abspath(box_obj_output_path)),
        'isDynamic': False,
        'isWall': True,
        "scale": [1.0, 2.0, 1.0],
        "mapThickness": 0.0,
        "mapResolution": [10, 10, 10]
    })
    scene['RigidBodies'].append(boundary)

    # 添加动态刚体，并生成对应bgeo油箱文件
    # bb_obj = np.random.choice(dynamic_model_path)
    # 旋转盒子
    bb_obj = rotate_obj_file(dynamic_model_path, sim_directory+"/box/" +  f"sim_{seed:04}" + '_rotated_box.obj', rand_R)
    # convert bounding box to particles
    dynamic_box_output_path = os.path.join(sim_directory, 'dynamic_box.obj')
    copyfile(bb_obj, dynamic_box_output_path)
    box_output_path = os.path.join(sim_directory, 'box.bgeo')
    bb, bb_normals = obj_surface_to_particles(dynamic_box_output_path)
    write_bgeo_from_numpy(box_output_path, bb, bb_normals)
    
    dynamic_body = deepcopy(default_rigidbody)
    # 有厚度容器    simple false 69r true
    dynamic_body.update({
        'id': 1,
        'geometryFile': os.path.basename(os.path.abspath(dynamic_box_output_path)),
        'particleFile': 'box.bgeo',
        'isDynamic': True,
        'isWall': False,
        'density': 999999999999,
        'mapInvert': True,
        'rotationAxis': normal_vector,
        'mapResolution': [128, 128, 128]
    })

    # 无厚度容器
    # dynamic_body.update({
    #     'id': 1,
    #     'geometryFile': os.path.basename(os.path.abspath(dynamic_box_output_path)),
    #     'isDynamic': True,
    #     'isWall': False,
    #     'density': 999999999999,
    #     'mapInvert': True,
    #     'rotationAxis': normal_vector
    # })
    scene['RigidBodies'].append(dynamic_body)

    # 添加铰链关节
    joint = deepcopy(default_joint)
    joint.update({
        'bodyID1': 1,
        'bodyID2': 0,
        'axis': normal_vector,
        'targetSequence': rotation_sequence
    })
    scene['TargetVelocityMotorHingeJoints'].append(joint)

    # 添加流体，旋转流体
    # fluid_obj = np.random.choice(fluid_model_path)
    # print("流体块obj：", fluid_obj)
        # fluid_shapes.remove(fluid_model_path) # 为了液体不重复

    fluid_obj = rotate_obj_file(fluid_model_path,
                            os.path.join(sim_directory, f'sim_{seed:04d}_rotated_fluid.obj'),
                            rand_R)


    fluid_particles = obj_volume_to_particles(fluid_obj, scale=1)[0]
    fluid_velocities = np.zeros_like(fluid_particles)
    
    fluid_output_path = os.path.join(sim_directory, 'fluid0.bgeo')
    write_bgeo_from_numpy(fluid_output_path, fluid_particles, fluid_velocities)
    
    fluid_model = {
        'translation': [0.0, 0.0, 0.0],
        'scale': [1.0, 1.0, 1.0],
        'id': 'fluid0',
        'particleFile': 'fluid0.bgeo'
    }
    scene['FluidModels'].append(fluid_model)
    scene['fluid0'] = default_fluid

    # 保存场景配置
    scene_output_path = os.path.join(sim_directory, 'scene.json')
    with open(scene_output_path, 'w') as f:
        json.dump(scene, f, indent=4)

    # 运行模拟器
    run_simulator(os.path.abspath(scene_output_path), sim_directory)
    
    return sim_directory

if __name__ == '__main__':
    sys.exit(main())