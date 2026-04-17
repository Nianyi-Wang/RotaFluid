"""Functions for reading the compressed training/validation data records"""
import os
import sys
import numpy as np
from glob import glob
# import dataflow
from tensorpack import dataflow
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class PhysicsSimDataFlow(dataflow.RNGDataFlow):
    """Data flow for msgpacks generated from SplishSplash simulations.
    """

    def __init__(self, files, random_rotation=False, shuffle=False, window=2):
        if not len(files):
            raise Exception("List of files must not be empty")
        if window < 1:
            raise Exception("window must be >=1 but is {}".format(window))
        self.files = files
        self.random_rotation = random_rotation
        self.shuffle = shuffle
        self.window = window

    def __iter__(self):
        decompressor = zstd.ZstdDecompressor()
        files_idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(files_idxs)

        for file_i in files_idxs:
            with open(self.files[file_i], 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()),
                                       raw=False)

            data_idxs = np.arange(len(data) - self.window + 1)
            if self.shuffle:
                self.rng.shuffle(data_idxs)
            
            # 读取第一帧的信息
            box = data[0]['box']
            box_normals = data[0]['box_normals']
            rotation_axis =  data[0]['rotation_axis']
            rotation_sequence =  data[0]['rotation_sequence']

            for data_i in data_idxs:
                sample = {}
                # 生成随机旋转矩阵（如果需要）
                rand_R = np.eye(3, dtype=np.float32)
                if self.random_rotation:
                    angle_rad = self.rng.uniform(0, 2 * np.pi)
                    s = np.sin(angle_rad)
                    c = np.cos(angle_rad)
                    rand_R = np.array([c, 0, s, 0, 1, 0, -s, 0, c],
                                    dtype=np.float32).reshape((3, 3))

                # 直接使用第一帧的数据，不做任何形状检查
                if self.random_rotation:
                    # 如果需要随机旋转，则对第一帧的数据进行旋转
                    sample['box_ref'] = np.matmul(box, rand_R) if box.shape[1] == 3 else box
                    sample['box_normals_ref'] = np.matmul(box_normals, rand_R) if box_normals.shape[1] == 3 else box_normals
                    sample['rotation_axis'] = np.matmul(rotation_axis, rand_R)
                else:
                    # 否则直接使用第一帧的原始数据
                    sample['box_ref'] = box
                    sample['box_normals_ref'] = box_normals
                    sample['rotation_axis'] = rotation_axis
                
                # 旋转序列不需要旋转变换
                sample['rotation_sequence'] = rotation_sequence

                # 处理时间序列数据
                for time_i in range(self.window):
                    item = data[data_i + time_i]
                    
                    # 添加box位置和旋转矩阵
                    pos = item.get('box_pos', np.zeros(3, dtype=np.float32))
                    sample[f'position{time_i}'] = pos.astype(np.float32)  # 确保类型正确
                    sample[f'rotation{time_i}'] = item.get('box_rot', np.eye(3, dtype=np.float32)).astype(np.float32)
                    
                    # 计算当前帧在整个序列中的相对时间位置
                    frame_id = item['frame_id']
                    current_time = frame_id / 50.0  # 假设50fps，转换为秒
                    
                    # 将标量值转换为NumPy数组，以便后续处理
                    sample[f'sim_time{time_i}'] = np.array([current_time], dtype=np.float32)
                    sample[f'frame_id{time_i}'] = np.array([frame_id], dtype=np.float32)  # 转换为浮点数组
                    
                    for k in ('pos', 'vel'):
                        if self.random_rotation:
                            sample[k + str(time_i)] = np.matmul(item[k], rand_R)
                        else:
                            sample[k + str(time_i)] = item[k]

                    # 其他标量值也可以转换为数组
                    for k in ('m', 'viscosity','scene_id'):
                        sample[k + str(time_i)] = item[k]

                yield sample      

def read_data_val(files, **kwargs):
    return read_data(files=files,
                     batch_size=1,
                     repeat=False,
                     shuffle_buffer=None,
                     random_rotation=False,
                     num_workers=1,
                     **kwargs)


def read_data_train(files, batch_size, random_rotation=True, **kwargs):
    return read_data(files=files,
                     batch_size=batch_size,
                     random_rotation=random_rotation,
                     repeat=True,
                     shuffle_buffer=512,
                     **kwargs) #训练时随机旋转，测试时不旋转


def read_data(files=None,
              batch_size=1,
              window=2,
              random_rotation=False,
              repeat=False,
              shuffle_buffer=None,
              num_workers=1,
              cache_data=False):
    print(files[0:20], '...' if len(files) > 20 else '')

    # caching makes only sense if the data is finite
    if cache_data:
        if repeat == True:
            raise Exception("repeat must be False if cache_data==True")
        if random_rotation == True:
            raise Exception("random_rotation must be False if cache_data==True")
        if num_workers != 1:
            raise Exception("num_workers must be 1 if cache_data==True")
    print("random_rotation: "+str(random_rotation))
    df = PhysicsSimDataFlow(
        files=files,
        random_rotation=random_rotation,
        shuffle=True if shuffle_buffer else False,
        window=window,
    )

    if repeat:
        df = dataflow.RepeatedData(df, -1)

    if shuffle_buffer:
        df = dataflow.LocallyShuffleData(df, shuffle_buffer)

    if num_workers > 1:
        df = dataflow.MultiProcessRunnerZMQ(df, num_proc=num_workers)

    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)

    if cache_data:
        df = dataflow.CacheData(df)

    df.reset_state()
    return df


def validate_data_sample(sample):
    required_fields = {
        'box_ref': (np.ndarray, None),  # 修改为可变形状
        'box_normals_ref': (np.ndarray, None),  # 修改为可变形状
        'rotation_axis': (np.ndarray, (3,)),
        'rotation_sequence': (np.ndarray, (10,)),
    }
    
    time_dependent_fields = {
        'rotation': (np.ndarray, (3, 3)),
        'pos': (np.ndarray, None),  # 保持粒子数量可变
        'vel': (np.ndarray, None),
        'position': (np.ndarray, (3,)),  # 修改为一维向量
        'm': (np.ndarray, None),
        'viscosity': (np.ndarray, None),
        'frame_id': (np.ndarray, (1,)),  # 修改为np.ndarray类型，形状为(1,)
        'scene_id': (str, None)
    }
    
    # 检查基本字段
    for field, (dtype, shape) in required_fields.items():
        if field not in sample:
            print(f"错误：缺少必需字段 {field}")
            return False
        if not isinstance(sample[field], dtype):
            print(f"错误：字段 {field} 类型错误，应为 {dtype}")
            return False
        if shape and sample[field].shape != shape:
            print(f"错误：字段 {field} 形状错误，应为 {shape}，实际为 {sample[field].shape}")
            return False
    
    # 检查时间相关字段
    for i in range(2):  # 假设window=2
        for field, (dtype, shape) in time_dependent_fields.items():
            key = f"{field}{i}"
            if key not in sample:
                print(f"错误：缺少时间相关字段 {key}")
                return False
            if not isinstance(sample[key], dtype):
                print(f"错误：字段 {key} 类型错误，应为 {dtype}")
                return False
            if shape and sample[key].shape != shape:
                print(f"错误：字段 {key} 形状错误，应为 {shape}，实际为 {sample[key].shape}")
                return False
    
    # 检查粒子数量一致性
    n_particles = sample['pos0'].shape[0]
    particle_fields = ['pos', 'vel', 'm', 'viscosity']
    for i in range(2):
        for field in particle_fields:
            key = f"{field}{i}"
            if sample[key].shape[0] != n_particles:
                print(f"错误：字段 {key} 粒子数量不一致，应为 {n_particles}，实际为 {sample[key].shape[0]}")
                return False
    
    return True

def test_data_reading(dataset_dir, num_samples=5):
    import glob
    files = glob.glob(os.path.join(dataset_dir, 'train/*.zst'))
    if not files:
        print(f"错误：在 {dataset_dir}/train/ 中未找到.zst文件")
        return
    
    # 按文件名排序并打印
    files.sort()
    # print("数据文件列表:")
    # for f in files:
    #     print(f"- {os.path.basename(f)}")
    
    print(f"\n开始测试数据读取，将检查 {num_samples} 个样本...")
    df = PhysicsSimDataFlow(files=files, random_rotation=False, shuffle=False)
    
    current_file = None
    for i, sample in enumerate(df):
        if i >= num_samples:
            break
            
        # 获取当前文件名和基本信息
        scene_id = sample['scene_id0']
        frame_id = sample['frame_id0'][0]  # 从数组中获取标量值
        current_time = sample['sim_time0'][0]
        
        # 打印当前正在处理的文件名
        file_name = files[int(frame_id) // 50]  # 确保是整数索引
        if current_file != file_name:
            current_file = file_name
            print(f"\n正在处理文件: {os.path.basename(current_file)}")
        
        print(f"\n当前样本信息:")
        print(f"Scene ID: {scene_id}")
        print(f"Frame ID: {frame_id}")
        print(f"Simulation Time: {current_time:.3f}s")
        
        # 如果有连续帧，显示时间序列
        if 'sim_time1' in sample:
            next_time = sample['sim_time1'][0]
            print(f"Next Frame Time: {next_time:.3f}s")
            print(f"Time Step: {next_time - current_time:.3f}s")
        
        # 显示对应的旋转序列时间点
        # if 'rotation_sequence' in sample:
        #     seq = sample['rotation_sequence']
        #     print("\n旋转序列时间点:")
        #     for i in range(0, len(seq), 2):
        #         print(f"t={seq[i]:.3f}s: 角速度={seq[i+1]:.3f}")
        
        # 检查旋转轴和旋转序列
        if 'rotation_axis' in sample:
            print(f"旋转轴 (rotation_axis): {sample['rotation_axis']}")
        else:
            print("警告：未找到旋转轴数据")
            
        if 'rotation_sequence' in sample:
            print(f"旋转序列 (rotation_sequence): {sample['rotation_sequence']}")
            print(f"旋转序列形状: {sample['rotation_sequence'].shape}")
        else:
            print("警告：未找到旋转序列数据")
        
        if validate_data_sample(sample):
            print("√ 样本数据格式正确")
            print(f"  - 粒子数量: {sample['pos0'].shape[0]}")
            print(f"  - 场景ID: {sample['scene_id0']}")
            print(f"  - 帧ID: {sample['frame_id0']}-{sample['frame_id1']}")
            
            # 检查旋转矩阵和位置向量
            print(f"  - 旋转矩阵 (rotation0): {sample['rotation0'].shape}")
            # 输出旋转矩阵和位置
            print(f"    rotation0:\n{sample['rotation0']}")
            print(f"    position0:\n{sample['position0']}")
            print(f"  - 位置向量 (position0): {sample['position0'].shape}")
        else:
            print("× 样本数据格式有误")
            
        # 打印更详细的数据结构信息
        print("\n样本包含的所有键:")
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, np.ndarray):
                print(f"  - {key}: 形状={value.shape}, 类型={value.dtype}")
            else:
                print(f"  - {key}: 类型={type(value)}")

if __name__ == '__main__':
    # 使用示例
    dataset_dir = "/home/zh/DualFluidNet_code_update/datasets/default_dataset_15s"
    test_data_reading(dataset_dir)
