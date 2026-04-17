import numpy as np

def calculate_rotation_matrix(angle_z, angle_y):
    """计算组合旋转矩阵
    
    Args:
        angle_z: z轴旋转角度（弧度）
        angle_y: y轴旋转角度（弧度）
    
    Returns:
        R: 3x3旋转矩阵
    """
    # 创建z轴旋转矩阵
    cos_z = np.cos(angle_z)
    sin_z = np.sin(angle_z)
    R_z = np.array([
        [cos_z, -sin_z, 0.0],
        [sin_z, cos_z, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 创建y轴旋转矩阵
    cos_y = np.cos(angle_y)
    sin_y = np.sin(angle_y)
    R_y = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y]
    ])
    
    # 组合旋转（先z后y）
    R = R_y @ R_z
    return R

# def calculate_normal_from_angles(angle_z_deg, angle_y_deg):
#     """根据目标姿态角度计算旋转轴和旋转角度
    
#     Args:
#         angle_z_deg: z轴目标角度（角度）
#         angle_y_deg: y轴目标角度（角度）
    
#     Returns:
#         normal: 归一化的旋转轴向量 [nx, ny, nz]
#     """
#     # 角度转弧度
#     angle_z = round(np.pi * angle_z_deg / 180.0, 6)
#     angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    
#     # 计算目标方向向量
#     target = np.array([
#         np.sin(angle_z),   # z轴旋转影响xy平面
#         -np.sin(angle_y),  # y轴旋转影响yz平面
#         np.cos(angle_z) * np.cos(angle_y)
#     ])
    
#     # 初始向量
#     initial = np.array([0.0, 0.0, 1.0])
    
#     # 计算旋转轴
#     rotation_axis = np.cross(target, initial)
    
#     if np.allclose(rotation_axis, 0):
#         return [0.0, 0.0, 1.0]
    
#     # 归一化旋转轴
#     rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
#     return [round(n, 6) for n in rotation_axis]

# def calculate_rotation_matrix(angle_x, angle_y, angle_z):
#     """计算三轴组合旋转矩阵
    
#     Args:
#         angle_x: x轴旋转角度（弧度）
#         angle_y: y轴旋转角度（弧度）
#         angle_z: z轴旋转角度（弧度）
    
#     Returns:
#         R: 3x3旋转矩阵
#     """
#     # 创建x轴旋转矩阵
#     cos_x = np.cos(angle_x)
#     sin_x = np.sin(angle_x)
#     R_x = np.array([
#         [1.0, 0.0, 0.0],
#         [0.0, cos_x, -sin_x],
#         [0.0, sin_x, cos_x]
#     ])
    
#     # 创建y轴旋转矩阵
#     cos_y = np.cos(angle_y)
#     sin_y = np.sin(angle_y)
#     R_y = np.array([
#         [cos_y, 0.0, sin_y],
#         [0.0, 1.0, 0.0],
#         [-sin_y, 0.0, cos_y]
#     ])
    
#     # 创建z轴旋转矩阵
#     cos_z = np.cos(angle_z)
#     sin_z = np.sin(angle_z)
#     R_z = np.array([
#         [cos_z, -sin_z, 0.0],
#         [sin_z, cos_z, 0.0],
#         [0.0, 0.0, 1.0]
#     ])
    
#     # 组合旋转（先x后y最后z）
#     R = R_z @ R_y @ R_x
#     return R

def calculate_normal_from_angles(angle_z_deg, angle_x_deg, angle_y_deg=0):
    """根据三个轴的目标角度计算旋转轴和旋转角度
    
    Args:
        angle_z_deg: z轴目标角度（角度）
        angle_x_deg: x轴目标角度（角度）
        angle_y_deg: y轴目标角度（角度），默认为0
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 分别计算每个轴的旋转向量
    x_axis = np.array([1.0, 0.0, 0.0]) * np.sin(angle_x)
    y_axis = np.array([0.0, 1.0, 0.0]) * np.sin(angle_y)
    z_axis = np.array([0.0, 0.0, 1.0]) * np.sin(angle_z)
    
    # 组合旋转向量
    rotation_axis = x_axis + y_axis + z_axis
    
    if np.allclose(rotation_axis, 0):
        return [0.0, 0.0, 1.0]
    
    # 归一化旋转轴
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    return [round(n, 6) for n in rotation_axis]

def calculate_angular_acceleration(angle_x, angle_y, time):
    """计算达到目标角度所需的角加速度，包含加速和减速两个阶段"""
    # 计算总旋转角度
    total_angle = np.sqrt(angle_x**2 + angle_y**2)
    
    # 调整加速和减速时间比例为7:3
    # 加速时间更长，确保平稳加速
    t_acc = time * 0.7
    t_dec = time * 0.3
    
    # 计算加速阶段位移（占总位移的70%）
    s_acc = total_angle * 0.7
    # 加速阶段加速度：s = 1/2 * a * t^2
    alpha_acc = 2 * s_acc / (t_acc * t_acc)
    
    # 计算交接处速度
    v_mid = alpha_acc * t_acc
    
    # 计算减速阶段加速度
    s_dec = total_angle * 0.3  # 剩余30%的位移
    # 使用末速度为0的运动方程：
    # s_dec = v_mid * t_dec + 1/2 * alpha_dec * t_dec^2
    # 0 = v_mid + alpha_dec * t_dec
    alpha_dec = -v_mid / t_dec
    
    return alpha_acc, alpha_dec, t_acc

def calculate_rotation_sequences(angle_z_deg, angle_x_deg, time=0.5,t_start = 1.0,angle_y_deg=0):
    """计算三种不同的旋转序列：匀速、匀加速和匀减速
    
    Args:
        angle_z_deg: z轴目标旋转角度（角度）
        angle_x_deg: x轴目标旋转角度（角度）
        angle_y_deg: y轴目标旋转角度（角度），默认为0
        time: 总运动时间（秒），默认为1秒
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 计算总旋转角度和平均角速度
    total_angle = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    angular_velocity = round(total_angle / time, 6)
    
    # 时间点设置
    t_mid = round(time / 2, 6)+t_start
    t_end = round(time, 6)+t_start
    
    # 1. 匀速序列：保持恒定角速度
    constant_velocity_seq = [
        0.0,
        0.0,
        t_start,
        0.0,
        t_start+0.01,
        round(angular_velocity, 6),
        round(t_end, 6),
        round(angular_velocity, 6),
        round(t_end + 0.01, 6),
        0.000000
    ]
    
    # 2. 匀加速序列
    max_acc_vel = round(2 * angular_velocity, 6)
    constant_acc_seq = [
        0.0,
        0.0,
        t_start,
        0.000000,
        round(t_end, 6),
        max_acc_vel,
        round(t_end + 0.01, 6),
        0.000000,
        12.0,
        0.0
    ]
    
    # 3. 匀减速序列
    initial_vel = round(2 * angular_velocity, 6)
    constant_dec_seq = [
        t_start,
        initial_vel,
        round(t_mid, 6),
        round(initial_vel/2, 6),
        round(t_end, 6),
        0.000000
    ]
    
    return constant_velocity_seq, constant_acc_seq, constant_dec_seq, angular_velocity

def calculate_triangle_acc_dec_sequence(angle_z_deg, angle_x_deg, time=0.5, t_start=0.0, angle_y_deg=0):
    """计算先匀加速后匀减速的旋转序列（三角形速度分布），先正向旋转到目标角度，再反向旋转回0
    
    Args:
        angle_z_deg: z轴目标旋转角度（角度）
        angle_x_deg: x轴目标旋转角度（角度）
        angle_y_deg: y轴目标旋转角度（角度），默认为0
        time: 总运动时间（秒）
        t_start: 开始时间
    
    Returns:
        acc_dec_seq: 旋转序列 [t, v, t, v, ...]
        max_velocity: 最大角速度
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 计算总旋转角度 (单程)
    total_angle = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 我们需要完成 0 -> Angle -> 0 的过程
    # 总时间为 time
    # 分为两段：0 -> Angle (耗时 time/2) 和 Angle -> 0 (耗时 time/2)
    
    # 第一段：0 -> Angle
    # 三角形速度分布，位移为 total_angle，时间为 time/2
    # Area = 1/2 * v_peak * (time/2) = total_angle
    # v_peak = 4 * total_angle / time
    
    max_velocity = round(4 * total_angle / time, 6)
    
    # 时间点
    t_1 = round(t_start + time / 4, 6)      # 达到正向最大速度
    t_2 = round(t_start + time / 2, 6)      # 速度回0 (到达最大角度)
    t_3 = round(t_start + 3 * time / 4, 6)  # 达到反向最大速度
    t_4 = round(t_start + time, 6)          # 速度回0 (回到0角度)
    
    # 构造序列 [t, v, t, v, ...]
    acc_dec_seq = [
        0.0, 0.0,
        t_start, 0.0,
        t_1, max_velocity,
        t_2, 0.0,
        t_3, -max_velocity,
        t_4, 0.0
    ]
    
    return acc_dec_seq, max_velocity

def calculate_cycle_rotation_sequence(angle_z_deg, angle_x_deg, time=2.0, t_start=0.0, angle_y_deg=0):
    """计算先匀加速后匀减速的旋转序列（三角形速度分布），先正向旋转到目标角度，再反向旋转回0
    
    Args:
        angle_z_deg: z轴目标旋转角度（角度）
        angle_x_deg: x轴目标旋转角度（角度）
        angle_y_deg: y轴目标旋转角度（角度），默认为0
        time: 总运动时间（秒）
        t_start: 开始时间
    
    Returns:
        cyc_rotate_seq: 旋转序列 [t, v, t, v, ...]
        max_velocity: 最大角速度
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 计算总旋转角度 (单程)
    total_angle = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 我们需要完成 0 -> Angle -> 0 的过程
    # 总时间为 time
    # 分为两段：0 -> Angle (耗时 time/2) 和 Angle -> 0 (耗时 time/2)
    
    # 第一段：0 -> Angle
    # 三角形速度分布，位移为 total_angle，时间为 time/2
    # Area = 1/2 * v_peak * (time/2) = total_angle
    # v_peak = 4 * total_angle / time
    
    max_velocity = round(4 * total_angle / time, 6)
    
    # 时间点
    t_1 = round(t_start + time / 4, 3)      # 达到正向最大速度
    t_2 = round(t_start + time / 2, 3)      # 速度回0 (到达最大角度)
    t_3 = round(t_start + 3 * time / 4, 3)  # 达到反向最大速度
    t_4 = round(t_start + time, 3)          # 速度回0 (回到0角度)
    t_5 = round(t_start + time + time / 4, 3)      # 达到反向最大速度
    t_6 = round(t_start + time + time / 2, 3)      # 速度回0 (到达最大角度)
    t_7 = round(t_start + time + 3 * time / 4, 3)  # 达到正向最大速度
    t_8 = round(t_start + 2 * time, 3)          # 速度回0 (回到0角度)
    
    # 构造序列 [t, v, t, v, ...]
    cyc_rotate_seq = [
        0.0, 0.0,
        t_start, 0.0,
        t_1, max_velocity,
        t_2, 0.0,
        t_3, -max_velocity,
        t_4, 0.0,
        t_5, -max_velocity,
        t_6, 0.0,
        t_7, max_velocity,
        t_8, 0.0
    ]
    
    return cyc_rotate_seq, max_velocity

def calculate_cycle_sequence(angle_z_deg, angle_x_deg, time=0.5, t_start=0.0, angle_y_deg=0):
    """计算往复循环旋转序列：0 -> Max -> 0 -> -Max -> 0
    
    Args:
        angle_z_deg: z轴目标旋转角度（角度）
        angle_x_deg: x轴目标旋转角度（角度）
        angle_y_deg: y轴目标旋转角度（角度），默认为0
        time: 总运动时间（秒）
        t_start: 开始时间
    
    Returns:
        cycle_seq: 旋转序列 [t, v, t, v, ...]
        velocity: 恒定速度大小
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 计算最大偏转角度（振幅）
    amplitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 计算速度大小
    # 总路程 = 4 * amplitude (0->Max->0->-Max->0)
    # 速度 = 总路程 / 时间
    velocity = round(4 * amplitude / time, 6)
    
    # 时间点设置
    t_quarter = time / 4.0
    t1 = t_start
    t2 = t_start + t_quarter
    t3 = t_start + 3 * t_quarter
    t4 = t_start + time
    
    eps = 0.0001
    
    # 构造序列 [t, v, t, v, ...]
    # 0 -> Max: +V
    # Max -> -Max: -V (跨越0点)
    # -Max -> 0: +V
    
    cycle_seq = [
        0.0, 0.0,
        t1, 0.0,
        round(t1 + eps, 6), velocity,
        round(t2 - eps, 6), velocity,
        round(t2 + eps, 6), -velocity,
        round(t3 - eps, 6), -velocity,
        round(t3 + eps, 6), velocity,
        round(t4 - eps, 6), velocity,
        round(t4 + eps, 6), 0.0
    ]
    
    return cycle_seq, velocity

def calculate_sine_sequence(angle_z_deg, angle_x_deg, time=0.5, t_start=0.0, angle_y_deg=0, steps=40):
    """计算正弦运动序列：角度按正弦变化 0 -> Max -> 0 -> -Max -> 0
    
    Args:
        angle_z_deg: z轴目标旋转角度（角度）
        angle_x_deg: x轴目标旋转角度（角度）
        angle_y_deg: y轴目标旋转角度（角度），默认为0
        time: 总运动时间（秒）
        t_start: 开始时间
        steps: 离散采样点数
    
    Returns:
        sine_seq: 旋转序列 [t, v, t, v, ...]
        max_velocity: 最大角速度
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 计算振幅
    amplitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 角速度公式: omega(t) = A * (2*pi/T) * cos(2*pi * (t-t_start)/T)
    omega_max = amplitude * (2 * np.pi / time)
    
    sine_seq = [0.0, 0.0]
    
    # 生成余弦速度曲线
    for i in range(steps + 1):
        t_curr = t_start + time * i / steps
        phase = 2 * np.pi * i / steps
        velocity = omega_max * np.cos(phase)
        
        # 在t_start前保持0
        if i == 0:
            if t_start > 0:
                sine_seq.extend([t_start, round(velocity, 6)])
            else:
                # 如果t_start=0，直接添加第一个点（覆盖0.0, 0.0后面的点）
                # 但由于我们已经添加了0.0, 0.0，这里直接添加即可
                sine_seq.extend([round(t_curr, 6), round(velocity, 6)])
        else:
             sine_seq.extend([round(t_curr, 6), round(velocity, 6)])
            
    # 结束时归零
    sine_seq.extend([round(t_start + time + 0.01, 6), 0.0])
    
    return sine_seq, omega_max

def calculate_harmonic_motion_sequence(angle_z_deg, angle_x_deg, time=2.0, t_start=0.0, angle_y_deg=0, samples=40):
    """计算简谐运动(正弦运动)的旋转速度序列
    
    对应运动轨迹: Angle(t) = Amplitude * sin(2*pi * t / T)
    对应速度轨迹: Velocity(t) = Amplitude * (2*pi/T) * cos(2*pi * t / T)
    
    运动过程:
    T=0.00: 0度 (速度最大)
    T=0.25: 正向最大 (速度为0)
    T=0.50: 0度 (速度反向最大)
    T=0.75: 反向最大 (速度为0)
    T=1.00: 0度 (速度最大)
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 振幅
    amplitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 角频率 omega = 2pi / T
    omega = 2 * np.pi / time
    
    # 最大速度 (在位置为0时达到) v_max = A * w
    max_velocity = round(amplitude * omega, 6)
    
    # 生成序列
    # 初始状态
    sequence = [0.0, 0.0]
    
    # 如果有延迟启动, 确保在t_start之前速度为0
    if t_start > 0.001:
        sequence.extend([round(t_start - 0.001, 6), 0.0])
    
    # 采样点生成
    for i in range(samples + 1):
        # 当前时间比例 0.0 -> 1.0
        ratio = i / samples
        
        # 实际时间点
        t_curr = round(t_start + time * ratio, 6)
        
        # 计算速度 v = A * w * cos(w*t)
        # 也就是 v = max_velocity * cos(2*pi * ratio)
        vel = round(max_velocity * np.cos(2 * np.pi * ratio), 6)
        
        sequence.extend([t_curr, vel])

    # 结束后停止
    sequence.extend([round(t_start + time + 0.001, 6), 0.0])
    
    return sequence, max_velocity

def convert_angle(angle, to_radians=True):
    """角度和弧度的转换
    
    Args:
        angle: 输入角度值（角度或弧度）
        to_radians: True表示角度转弧度，False表示弧度转角度
    
    Returns:
        转换后的角度值
    """
    if to_radians:
        return round(np.pi * angle / 180.0, 6)
    else:
        return round(180.0 * angle / np.pi, 6)

def calculate_linear_acc_dec_harmonic_sequence(angle_z_deg, angle_x_deg, time=2.0, t_start=0.0, angle_y_deg=0):
    """计算分段匀加速/匀减速的循环运动序列
    
    对应的位置轨迹 Angle(t) 虽然类似正弦, 但由抛物线拼接而成
    对应的速度轨迹 Velocity(t) 是折线(三角形/锯齿形状)
    
    运动过程 (假设从0开始, time=T):
    [0, T/4]: 0 -> A  (实际上这部分通常做不到纯匀加速直达, 因为要满足0处速度最大)
    
    为了模仿简谐运动的 0->Max->0->-Max->0 轨迹，且使用匀变速：
    
    区间1 [0, T/4]: 从平衡位置(0度)运动到最大正向角度(+A)
        - 速度从 V_max 匀减速到 0
    区间2 [T/4, T/2]: 从最大正向(+A)运动回平衡位置(0度)
        - 速度从 0 匀加速(反向)到 -V_max
    区间3 [T/2, 3T/4]: 从平衡位置(0度)运动到最大负向角度(-A)
        - 速度从 -V_max 匀减速(反向)到 0
    区间4 [3T/4, T]: 从最大负向(-A)运动回平衡位置(0度)
        - 速度从 0 匀加速(正向)到 +V_max
        
    计算 V_max:
    单个区间(如0->T/4)的位移是振幅 A
    位移 S = (v_init + v_final) * t / 2
    A = (V_max + 0) * (T/4) / 2 = V_max * T / 8
    => V_max = 8 * A / T
    
    对比：正弦运动的 V_max = 2*pi*A/T ≈ 6.28*A/T
    匀变速运动的 V_max = 8*A/T
    匀变速为了达到同样振幅，峰值速度需要更大一点。
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    # 振幅 A
    amplitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 最大速度 V_max = 8 * A / T
    max_velocity = round(8 * amplitude / time, 6)
    
    # 关键时间点
    t_0 = t_start
    t_1 = round(t_start + time * 0.25, 6) # +Max Angle, Vel=0
    t_2 = round(t_start + time * 0.50, 6) # 0 Angle, Vel=-Max
    t_3 = round(t_start + time * 0.75, 6) # -Max Angle, Vel=0
    t_4 = round(t_start + time * 1.00, 6) # 0 Angle, Vel=+Max
    
    sequence = [
        0.0, 0.0,           # 初始静止
        t_0, 0.0,           # 启动前静止
        round(t_0 + 0.001, 6), max_velocity, # 启动瞬间突变到 V_max (理想情况) 或极短时间加速
        t_1, 0.0,           # 匀减速到 0 (到达最高点)
        t_2, -max_velocity, # 反向匀加速到 -V_max (回到原点)
        t_3, 0.0,           # 反向匀减速到 0 (到达最低点)
        t_4, max_velocity,  # 正向匀加速到 +V_max (回到原点)
        round(t_4 + 0.001, 6), 0.0 # 瞬间停止
    ]
    
    return sequence, max_velocity

def calculate_continuous_harmonic_sequence(angle_z_deg, angle_x_deg, time=2.0, t_start=0.0, angle_y_deg=0, samples=100):
    """计算连续简谐运动序列（起止速度为0，运动平滑）
    
    运动过程: 0 -> A -> 0
    速度曲线: 由两段正弦波组成
    [0, T/2]: 0 -> A (正向半波)
    [T/2, T]: A -> 0 (反向半波)
    
    最大速度 V_max = 2*pi*A/T (对于完整周期0->A->0->-A->0)
    这里只有一半，0->A对应 sin(pi*t/(T/2)) 积分后是 A
    A = \int_0^{T/2} V_max * sin(pi*t/(T/2)) dt = V_max * (T/2)/pi * [-cos]... = V_max * T/pi
    => V_max = pi * A / T * 2 ?
    
    let's derive:
    Target: 0 -> A in T/2.
    Vel(t) = V_max * sin(2*pi*t/T) ? No, that starts 0 ends 0 at T/2.
    Let omega = 2*pi/T.
    V(t) = V_max * sin(omega * t) for t in [0, T/2] is positive bump.
    Displacement = integral(V_max * sin(omega*t)) from 0 to T/2
                 = V_max/omega * [-cos(omega*t)]_0^{T/2}
                 = V_max/omega * (-(-1) - (-1)) = 2 * V_max / omega
                 = 2 * V_max / (2*pi/T) = V_max * T / pi
    We want Displacement = A.
    So A = V_max * T / pi  => V_max = pi * A / T.
    
    So for phase 1 (0 -> A): V(t) = (pi*A/T) * sin(2*pi*t/T)  (t in 0..T/2)
       Wait, if T is the total time for 0->A->0.
       Then 0->A takes T/2 time.
       So the "period" of the sine wave for 0->A is T.
       V(t) = V_max * sin(2*pi * t / T).
       At t=T/2, V=0. This matches.
       
    Phase 2 (A -> 0): V(t) should be negative.
       V(t) = -V_max * sin(2*pi * (t - T/2) / T) 
            = V_max * sin(2*pi * t / T)  (since sin(x + pi) = -sin(x))
            
    So actually it's just one full sine wave period for velocity?
    V(t) = V_max * sin(2*pi * t / T)
    Let's check displacement over T:
    Integral_0^T V_max * sin(2*pi*t/T) dt = 0.
    So it goes 0 -> A -> 0. Correct.
    The midpoint A is at T/2.
    Displacement at T/2 = Integral_0^{T/2} V_max * sin(2*pi*t/T) dt = V_max * T / pi.
    We need this to be Amplitude.
    So V_max = pi * Amplitude / T.
    """
    # 角度转弧度
    angle_x = round(np.pi * angle_x_deg / 180.0, 6)
    angle_y = round(np.pi * angle_y_deg / 180.0, 6)
    angle_z = round(np.pi * angle_z_deg / 180.0, 6)
    
    amplitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    # 理论最大速度 V_max = pi * A / T
    max_velocity = round(amplitude * np.pi / time, 6)
    
    sequence = []
    
    # 初始点
    sequence.extend([0.0, 0.0])
    if t_start > 0:
        sequence.extend([t_start, 0.0])
        
    for i in range(samples + 1):
        ratio = i / samples
        t_global = t_start + time * ratio
        t_local = time * ratio
        
        # V(t) = V_max * sin(2*pi * t / T)
        # 0 -> T/2: positive velocity (0 -> A)
        # T/2 -> T: negative velocity (A -> 0)
        v = max_velocity * np.sin(2 * np.pi * t_local / time)
            
        sequence.extend([round(t_global, 6), round(v, 6)])
        
    # 确保结束为0
    sequence.extend([round(t_start + time + 0.001, 6), 0.0])
    
    return sequence, max_velocity

if __name__ == '__main__':
    time = 4.0  # 3秒完成旋转
    angle_z = 30   # z轴旋转（俯仰）
    angle_x = 0  # x轴旋转（横滚）

    # y轴默认为0，不需要显式传入
    normal = calculate_normal_from_angles(angle_z, angle_x)
    print(f"目标旋转角度:")
    print(f"z轴: {angle_z:.4f}°")
    print(f"x轴: {angle_x:.4f}°")
    print(f"时间: {time:.4f}")
    print(f"\旋转轴: {normal}")
    
    # 计算三种旋转序列
    const_vel_seq, const_acc_seq, const_dec_seq, ang_vel = calculate_rotation_sequences(angle_z, angle_x, time)
    
    print(f"\n平均角速度: {ang_vel:.4f} rad/s, {np.degrees(ang_vel):.4f} °/s")
    print(f"\n匀速序列 (rad/s):")
    print(const_vel_seq)
    print(f"\n匀加速序列 (rad/s):")
    print(const_acc_seq)
    print(f"\n匀减速序列 (rad/s):")
    print(const_dec_seq)
    
    # 计算匀加速再匀减速的旋转序列
    tri_acc_dec_seq, max_vel = calculate_triangle_acc_dec_sequence(angle_z, angle_x, time)
    print(f"\n匀加速再匀减速序列 (rad/s):")
    print(tri_acc_dec_seq)

    # 计算cycle_rotation的旋转序列
    cycle_rotation_seq, cycle_rotation_vel = calculate_cycle_rotation_sequence(angle_z, angle_x, time)
    print(f"\n周期旋转序列 (rad/s):")
    print(cycle_rotation_seq)

    # 计算往复循环旋转序列
    # cycle_seq, cycle_vel = calculate_cycle_sequence(angle_z, angle_x, time)
    # print(f"\n往复循环旋转序列 (rad/s):")
    # print(cycle_seq)
    # print(f"循环速度: {cycle_vel:.4f} rad/s")
    
    # 计算往复循环的旋转序列
    # cycle_seq, velocity = calculate_cycle_sequence(angle_z, angle_x, time)
    # print(f"\n往复循环旋转序列 (rad/s):")
    # print(cycle_seq)

    # # 计算正弦旋转序列
    # sine_seq, sine_max_vel = calculate_sine_sequence(angle_z, angle_x, time, steps=20)
    # print(f"\n正弦旋转序列 (rad/s):")
    # print(sine_seq)
    # print(f"最大角速度: {sine_max_vel:.4f} rad/s")
    
    # 计算正弦运动序列
    # sine_seq, max_sine_vel = calculate_sine_sequence(angle_z, angle_x, time)
    # print(f"\n正弦运动序列 (rad/s):")
    # print(sine_seq)
    # print(f"正弦最大速度: {max_sine_vel:.4f} rad/s")
    
    # 计算简谐运动序列
    harm_seq, max_harm_vel = calculate_harmonic_motion_sequence(angle_z, angle_x, time, samples=40)
    print(f"\n简谐运动序列 (rad/s):")
    print(harm_seq)
    print(f"简谐最大速度: {max_harm_vel:.4f} rad/s")

    # 计算分段匀变速(类简谐)序列
    lin_harm_seq, max_lin_vel = calculate_linear_acc_dec_harmonic_sequence(angle_z, angle_x, time)
    print(f"\n分段匀变速序列 (rad/s):")
    print(lin_harm_seq)
    print(f"匀变速最大速度: {max_lin_vel:.4f} rad/s (系数8/T)")

    # 计算连续匀变速序列(起止0)
    # cont_lin_seq, max_cont_vel = calculate_continuous_linear_acc_dec_sequence(angle_z, angle_x, time)
    # print(f"\n连续匀变速序列 (起止0) (rad/s):")
    # print(cont_lin_seq)
    # print(f"连续匀变速最大速度: {max_cont_vel:.4f} rad/s")

    # 计算连续简谐运动序列(正弦波拼接)
    cont_harm_seq, max_cont_harm_vel = calculate_continuous_harmonic_sequence(angle_z, angle_x, time)
    print(f"\n连续简谐运动序列 (起止0) (rad/s):")
    print(cont_harm_seq)
    print(f"连续简谐最大速度: {max_cont_harm_vel:.4f} rad/s")
     # 将时间加7，所有速度反转
    fan_cont_harm_seq = []
    for i in range(0, len(cont_harm_seq), 2):
        t = cont_harm_seq[i]
        v = cont_harm_seq[i+1]
        fan_cont_harm_seq.extend([round(t + 7, 6), round(-v, 6)])

    print(f"\n反向连续简谐运动序列 (rad/s):")
    print(fan_cont_harm_seq)
    
    # # 计算分段匀加速/匀减速的旋转序列
    # linear_acc_dec_harmonic_seq, max_linear_harm_vel = calculate_linear_acc_dec_harmonic_sequence(angle_z, angle_x, time, t_start=0.0)
    # print(f"\n分段匀加速/匀减速序列 (rad/s):")
    # print(linear_acc_dec_harmonic_seq)
    # print(f"最大速度: {max_linear_harm_vel:.4f} rad/s")
