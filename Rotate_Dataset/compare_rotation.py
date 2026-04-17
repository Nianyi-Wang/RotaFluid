import json
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_json_config(json_path):
    """加载JSON配置文件"""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # 提取旋转轴和目标序列
    joint_info = config["TargetVelocityMotorHingeJoints"][0]
    axis = joint_info["axis"]
    target_sequence = joint_info["targetSequence"]
    
    return axis, target_sequence

def calculate_rotation_matrices(axis, time_vel_sequence, fps=50, duration=30.0):
    """计算旋转矩阵序列 - 使用四元数累积旋转"""
    # 规范化旋转轴
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # 分离时间和角速度序列
    time_sequence = time_vel_sequence[::2]
    velocity_sequence = time_vel_sequence[1::2]
    
    # 计算总帧数
    total_frames = int(fps * duration) + 1
    rotation_matrices = []
    angles = []
    
    # 使用四元数表示当前旋转
    current_quat = Rotation.identity()
    dt = 1.0 / fps
    
    for frame in range(total_frames):
        t = frame * dt
        # 获取当前角速度
        angular_vel = calculate_angular_velocity(t, time_sequence, velocity_sequence)
        
        # 计算这一步的旋转增量（使用四元数）
        angle_increment = angular_vel * dt
        delta_rotation = Rotation.from_rotvec(angle_increment * axis)
        
        # 累积旋转（四元数乘法）
        current_quat = delta_rotation * current_quat
        
        # 保存当前旋转矩阵
        rotation_matrices.append(current_quat.as_matrix())
        
        # 计算等效的总角度（用于可视化）
        rotvec = current_quat.as_rotvec()
        angle = np.linalg.norm(rotvec)
        angles.append(angle)
    
    return np.array(rotation_matrices), np.array(angles)

def calculate_rotation_matrices_rk4(axis, time_vel_sequence, fps=50, duration=30.0):
    """使用RK4积分方法计算旋转矩阵序列"""
    # 规范化旋转轴
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # 分离时间和角速度序列
    time_sequence = time_vel_sequence[::2]
    velocity_sequence = time_vel_sequence[1::2]
    
    # 计算总帧数
    total_frames = int(fps * duration) + 1
    rotation_matrices = []
    angles = []
    
    # 使用四元数表示当前旋转
    current_quat = Rotation.identity()
    dt = 1.0 / fps
    
    for frame in range(total_frames):
        t = frame * dt
        
        # RK4积分
        k1 = calculate_angular_velocity(t, time_sequence, velocity_sequence)
        k2 = calculate_angular_velocity(t + dt/2, time_sequence, velocity_sequence)
        k3 = calculate_angular_velocity(t + dt/2, time_sequence, velocity_sequence)
        k4 = calculate_angular_velocity(t + dt, time_sequence, velocity_sequence)
        
        # 计算角度增量
        angle_increment = (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 创建增量旋转
        delta_rotation = Rotation.from_rotvec(angle_increment * axis)
        
        # 累积旋转
        current_quat = delta_rotation * current_quat
        
        # 保存当前旋转矩阵
        rotation_matrices.append(current_quat.as_matrix())
        
        # 计算等效的总角度
        rotvec = current_quat.as_rotvec()
        angle = np.linalg.norm(rotvec)
        angles.append(angle)
    
    return np.array(rotation_matrices), np.array(angles)

def calculate_angular_velocity(t, time_sequence, velocity_sequence):
    """计算给定时间的角速度（线性插值）"""
    if t <= time_sequence[0]:
        return velocity_sequence[0]
    if t >= time_sequence[-1]:
        return velocity_sequence[-1]
    
    for i in range(len(time_sequence)-1):
        t0, t1 = time_sequence[i], time_sequence[i+1]
        v0, v1 = velocity_sequence[i], velocity_sequence[i+1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return v0 + alpha * (v1 - v0)
    return 0.0

def read_rb_bin_matrices(rb_bin_folder):
    """读取rigid_bodies文件夹中所有bin文件的旋转矩阵和位置
    
    Args:
        rb_bin_folder: rigid_bodies文件夹路径
        
    Returns:
        positions: 所有帧的位置数组 [num_frames, 3]
        matrices: 所有帧的旋转矩阵数组 [num_frames, 3, 3]
    """
    def extract_number(filename):
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    bin_files = sorted([f for f in os.listdir(rb_bin_folder) if f.endswith('.bin')], key=extract_number)
    
    positions = []
    matrices = []
    
    for bin_file in bin_files:
        file_path = os.path.join(rb_bin_folder, bin_file)
        
        with open(file_path, 'rb') as f:
            # 第一帧的特殊格式
            if bin_file == bin_files[0]:
                # 读取边界模型数量
                num_boundary_models = struct.unpack('I', f.read(4))[0]
                
                # 对每个边界模型，读取信息
                for i in range(num_boundary_models):
                    # 读取网格文件名长度
                    filename_length = struct.unpack('I', f.read(4))[0]
                    f.read(filename_length)  # 读取文件名
                    f.read(12)  # 读取缩放信息
                    f.read(1)   # 读取是否为墙体
                    f.read(16)  # 读取颜色信息
            
            try:
                # 跳过第一个模型的数据
                f.read(12)  # 跳过位置
                f.read(36)  # 跳过旋转矩阵
                
                # 读取第二个模型的数据
                pos = struct.unpack('fff', f.read(12))
                rot = struct.unpack('fffffffff', f.read(36))
                
                positions.append(np.array(pos))
                matrices.append(np.array(rot).reshape(3, 3))
                
            except Exception as e:
                print(f"警告: 无法读取文件 {bin_file} 中的数据: {e}")
    
    return np.array(positions), np.array(matrices)

def compare_matrices(calculated_matrices, bin_matrices):
    """比较计算的旋转矩阵和bin文件中的旋转矩阵"""
    # 确保矩阵数量相同
    min_frames = min(len(calculated_matrices), len(bin_matrices))
    
    # 计算每帧的误差
    errors = []
    for i in range(min_frames):
        # 计算Frobenius范数作为误差度量
        error = np.linalg.norm(calculated_matrices[i] - bin_matrices[i], 'fro')
        errors.append(error)
    
    return np.array(errors)

def calculate_and_plot_angular_changes(time_vel_sequence, bin_matrices, fps, axis, output_dir, duration=4.0):
    """计算角速度和角度变化并绘图
    
    Args:
        time_vel_sequence: [t0, v0, t1, v1, ...] 格式的速度序列
        bin_matrices: 实际的旋转矩阵序列
        fps: 帧率
        axis: 旋转轴
        output_dir: 图片保存目录
        duration: 仿真持续时间
    """
    # 准备绘图数据 - 理论值
    dt = 0.001  # 1ms分辨率
    times = []
    velocities = []
    angles = []
    
    current_angle = 0.0
    
    # 提取时间点和速度点
    seq_times = time_vel_sequence[0::2]
    seq_vels = time_vel_sequence[1::2]
    
    for t in np.arange(0, duration, dt):
        times.append(t)
        
        # 计算当前角速度
        vel = calculate_angular_velocity(t, seq_times, seq_vels)
        velocities.append(vel)
        
        # 积分计算角度
        current_angle += vel * dt
        angles.append(np.degrees(current_angle))

    # 准备绘图数据 - 实际值
    actual_times = []
    actual_angles_rad = []
    
    # 规范化旋转轴
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    for i, matrix in enumerate(bin_matrices):
        t = i / fps
        if t > duration:
            break
        actual_times.append(t)
        
        # 计算带符号的角度
        r = Rotation.from_matrix(matrix)
        rotvec = r.as_rotvec()
        
        # 投影到旋转轴上获取带符号的角度 [-pi, pi]
        angle_rad = np.dot(rotvec, axis)
        actual_angles_rad.append(angle_rad)
        
    # 解包角度以处理超过 +/- pi 的情况
    actual_angles_unwrapped = np.unwrap(actual_angles_rad)
    actual_angles = np.degrees(actual_angles_unwrapped)
    
    # 计算实际角速度 (数值微分)
    actual_velocities = np.zeros(len(actual_angles))
    if len(actual_angles) > 1:
        # 使用梯度计算角速度 (radians -> radians/s)
        actual_velocities = np.gradient(actual_angles_unwrapped, 1.0/fps)
        
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 绘制角速度
    ax1.plot(times, velocities, 'b--', label='Target Angular Velocity', linewidth=2)
    ax1.plot(actual_times, actual_velocities, 'g-', label='Actual Angular Velocity', alpha=0.7)
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('Rotation Sequence Analysis')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制角度
    ax2.plot(times, angles, 'r--', label='Target Angle', linewidth=2)
    ax2.plot(actual_times, actual_angles, 'orange', label='Actual Angle', alpha=0.7)
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    # 角速度
    output_file = os.path.join(output_dir, 'rotation_angular_velocity.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    # Set font at the beginning of main function
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'Arial']  # Try multiple fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    # 路径设置
    base_path = "/home/zh/fueltank_datasets/output/rotate_69r_test_025_baoxian_v5/sim_0042"
    bgeo_path = "/home/zh/fueltank_datasets/output/rotate_69r_test_025_baoxian_v5/sim_0042/box.bgeo"
    json_path = os.path.join(base_path, "scene.json")
    rb_bin_folder = os.path.join(base_path, "rigid_bodies")
    
    # 加载JSON配置
    axis, target_sequence = load_json_config(json_path)
    print(f"旋转轴: {axis}")
    print(f"目标序列: {target_sequence}")
    
    # 读取bin文件中的旋转矩阵和位置
    positions, bin_matrices = read_rb_bin_matrices(rb_bin_folder)
    print(f"从bin文件读取到 {len(bin_matrices)} 个旋转矩阵")
    print(f"从bin文件读取到 {len(positions)} 个位置数据")
    
    # 分析position变化
    if len(positions) > 0:
        print(f"\nPosition分析:")
        print(f"第一帧位置: {positions[0]}")
        if len(positions) > 1:
            print(f"最后一帧位置: {positions[-1]}")
            
            # 计算每一帧相对于第一帧的位移
            displacements = positions - positions[0]
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
            
            # 计算相邻帧之间的位移
            frame_to_frame_displacements = np.diff(positions, axis=0)
            frame_to_frame_magnitudes = np.linalg.norm(frame_to_frame_displacements, axis=0)
            
            print(f"最大总位移: {np.max(displacement_magnitudes):.6f}")
            print(f"平均总位移: {np.mean(displacement_magnitudes):.6f}")
            print(f"最大帧间位移: {np.max(frame_to_frame_magnitudes):.6f}")
            print(f"平均帧间位移: {np.mean(frame_to_frame_magnitudes):.6f}")
            
            # 检查position是否基本保持不变
            if np.max(displacement_magnitudes) < 1e-6:
                print("Position基本保持不变（误差小于1e-6）")
            else:
                print("Position存在明显变化")
            
            # 可视化position变化
            plt.figure(figsize=(15, 10))
            
            # 子图1: X, Y, Z坐标随时间的变化
            plt.subplot(2, 3, 1)
            plt.plot(positions[:, 0], label='X')
            plt.plot(positions[:, 1], label='Y')
            plt.plot(positions[:, 2], label='Z')
            plt.title('Position Components vs Frame')
            plt.xlabel('Frame')
            plt.ylabel('Position')
            plt.legend()
            plt.grid(True)
            
            # 子图2: 总位移大小随时间的变化
            plt.subplot(2, 3, 2)
            plt.plot(displacement_magnitudes)
            plt.title('Total Displacement from Initial Position')
            plt.xlabel('Frame')
            plt.ylabel('Displacement Magnitude')
            plt.grid(True)
            
            # 子图3: 帧间位移大小
            plt.subplot(2, 3, 3)
            plt.plot(frame_to_frame_magnitudes)
            plt.title('Frame-to-Frame Displacement')
            plt.xlabel('Frame')
            plt.ylabel('Displacement Magnitude')
            plt.grid(True)
            
            # 子图4: 位移的X, Y, Z分量
            plt.subplot(2, 3, 4)
            plt.plot(displacements[:, 0], label='X displacement')
            plt.plot(displacements[:, 1], label='Y displacement')
            plt.plot(displacements[:, 2], label='Z displacement')
            plt.title('Displacement Components')
            plt.xlabel('Frame')
            plt.ylabel('Displacement')
            plt.legend()
            plt.grid(True)
            
            # 子图5: 3D轨迹图
            plt.subplot(2, 3, 5)
            plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7)
            plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
            plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
            plt.title('Position Trajectory (X-Y plane)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            
            # 子图6: 统计信息
            plt.subplot(2, 3, 6)
            stats_text = f"""Position Statistics:
    Max total displacement: {np.max(displacement_magnitudes):.2e}
    Mean total displacement: {np.mean(displacement_magnitudes):.2e}
    Max frame-to-frame: {np.max(frame_to_frame_magnitudes):.2e}
    Mean frame-to-frame: {np.mean(frame_to_frame_magnitudes):.2e}
    Total frames: {len(positions)}
    Position range X: [{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}]
    Position range Y: [{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]
    Position range Z: [{np.min(positions[:, 2]):.3f}, {np.max(positions[:, 2]):.3f}]"""
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_path, 'position_analysis.png'), dpi=300, bbox_inches='tight')
            
            # 输出每隔100帧的position信息
            print(f"\n每隔100帧的position变化:")
            for i in range(0, len(positions), 100):
                if i == 0:
                    print(f"帧 {i}: {positions[i]} (初始位置)")
                else:
                    displacement = positions[i] - positions[0]
                    displacement_mag = np.linalg.norm(displacement)
                    print(f"帧 {i}: {positions[i]}, 位移: {displacement}, 位移大小: {displacement_mag:.6f}")
    
    # 检查是否成功读取了矩阵
    if len(bin_matrices) == 0:
        print("错误: 未能从bin文件中读取到任何旋转矩阵")
        return
    
    # 使用不同方法计算旋转矩阵
    # 注意：calculate_rotation_matrices 已经使用四元数方法了，不需要额外的 calculate_rotation_matrices_quat
    calculated_matrices, angles = calculate_rotation_matrices(axis, target_sequence,duration=45.0)
    calculated_matrices_rk4, angles_rk4 = calculate_rotation_matrices_rk4(axis, target_sequence,duration=45.0)
    
    print(f"计算得到 {len(calculated_matrices)} 个旋转矩阵")
    
    # 计算最小帧数
    min_frames = min(len(calculated_matrices), len(bin_matrices), len(calculated_matrices_rk4))
    
    # 计算不同方法的误差
    errors = compare_matrices(calculated_matrices[:min_frames], bin_matrices[:min_frames])
    errors_rk4 = compare_matrices(calculated_matrices_rk4[:min_frames], bin_matrices[:min_frames])
    
    # 添加两种方法之间的误差对比
    errors_between_methods = compare_matrices(calculated_matrices[:min_frames], calculated_matrices_rk4[:min_frames])
    
    # 比较不同方法的误差
    # Use English labels for the first plot
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label='Euler Integration + Quaternion vs Bin File')
    plt.plot(errors_rk4, label='RK4 Integration + Quaternion vs Bin File')
    plt.plot(errors_between_methods, label='Euler vs RK4')
    plt.title('Rotation Matrix Error Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'rotation_methods_comparison.png'))
    
    # 新增：直接计算并绘图
    print("\n生成角速度和角度变化图...")
    # 假设fps为50，与create_physics脚本中设置的一致
    calculate_and_plot_angular_changes(target_sequence, bin_matrices, 50, axis, base_path, duration=2.0)

    # 输出统计信息
    print("\n误差统计信息:")
    print(f"欧拉积分 + 四元数 vs Bin文件:")
    print(f"  平均误差: {np.mean(errors)}")
    print(f"  最大误差: {np.max(errors)}")
    print(f"  最大误差出现在第 {np.argmax(errors)} 帧")
    print(f"  最小误差: {np.min(errors)}")
    
    print(f"\nRK4积分 + 四元数 vs Bin文件:")
    print(f"  平均误差: {np.mean(errors_rk4)}")
    print(f"  最大误差: {np.max(errors_rk4)}")
    print(f"  最大误差出现在第 {np.argmax(errors_rk4)} 帧")
    print(f"  最小误差: {np.min(errors_rk4)}")
    
    print(f"\n欧拉积分 vs RK4积分:")
    print(f"  平均误差: {np.mean(errors_between_methods)}")
    print(f"  最大误差: {np.max(errors_between_methods)}")
    print(f"  最大误差出现在第 {np.argmax(errors_between_methods)} 帧")
    print(f"  最小误差: {np.min(errors_between_methods)}")
    
    # 输出前5个矩阵进行对比
    print("\n前5个旋转矩阵对比:")
    for i in range(min(5, len(bin_matrices))):
        print(f"\n帧 {i}:")
        print("计算矩阵:")
        print(calculated_matrices[i])
        print("bin文件矩阵:")
        print(bin_matrices[i])
        print(f"误差: {errors[i]}")
    
    # 每隔300帧输出一次
    print("\n每隔300帧的旋转矩阵对比:")
    for i in range(0, min(len(bin_matrices), len(calculated_matrices)), 300):
        if i >= 5:  # 跳过前5个已经输出的
            print(f"\n帧 {i}:")
            print("计算矩阵:")
            print(calculated_matrices[i])
            print("bin文件矩阵:")
            print(bin_matrices[i])
            print(f"误差: {errors[i]}")
    
    # 输出统计信息
    print(f"\n平均误差: {np.mean(errors)}")
    print(f"最大误差: {np.max(errors)}")
    print(f"最小误差: {np.min(errors)}")
    
    # Remove the second font setting since we already set it at the beginning
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # Remove this line
    # plt.rcParams['axes.unicode_minus'] = False  # Remove this line
    
    # Continue with English labels for remaining plots
    plt.figure(figsize=(12, 6))
    plt.plot(errors)
    plt.title('Rotation Matrix Error')
    plt.xlabel('Frame')
    plt.ylabel('Frobenius Norm Error')
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'rotation_error.png'))
    
    # 可视化角度变化对比
    plt.figure(figsize=(12, 6))
    
    # 计算bin矩阵的角度
    bin_angles = []
    for matrix in bin_matrices:
        # 从旋转矩阵提取角度
        r = Rotation.from_matrix(matrix)
        angle = np.linalg.norm(r.as_rotvec())
        bin_angles.append(angle)
    
    plt.plot(np.degrees(angles[:min_frames]), label='Calculated Angle')  # 使用英文标签
    plt.plot(np.degrees(bin_angles), label='Bin File Angle')
    plt.title('Rotation Angle Comparison')  # 使用英文标题避免字体问题
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'rotation_angle_comparison.png'))
    

    from physics_data_helper import numpy_from_bgeo
    box, box_normals = numpy_from_bgeo(bgeo_path)
    print(f"Box position shape: {box.shape}, Box normals shape: {box_normals.shape}")
    print(f"Box position sample: {box[980:1000]}")
    print(f"Box normals sample: {box_normals[980:1000]}")
    print(f"分析完成，结果已保存到 {base_path}")

if __name__ == "__main__":
    main()