import pygame
import math
import random
import numpy as np
import time
from enum import Enum
from collections import deque
import json
import threading
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from scipy.spatial import KDTree

# 初始化pygame
pygame.init()
pygame.font.init()

# 屏幕设置
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("自动驾驶仿真系统 - 带PID控制")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BROWN = (139, 69, 19)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)


# 字体设置 - 使用系统字体或指定中文字体文件
def get_chinese_font(size):
    # 尝试加载系统中可能的中文字体
    chinese_fonts = [
        'SimHei',  # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
        'STSong',  # 华文宋体
        'STKaiti',  # 华文楷体
        'STHeiti',  # 华文黑体
        'PingFang SC',  # 苹方 (macOS)
        'Hiragino Sans GB',  # 冬青黑体 (macOS)
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'Noto Sans CJK SC',  # 思源黑体 (Linux)
    ]

    # 尝试加载字体
    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size)
            # 测试字体是否能渲染中文
            test_surface = font.render('测试', True, BLACK)
            if test_surface.get_width() > 0:
                return font
        except:
            continue

    # 如果系统字体都不可用，尝试加载字体文件
    font_paths = [
        os.path.join(os.getcwd(), 'simhei.ttf'),  # 黑体
        os.path.join(os.getcwd(), 'msyh.ttf'),  # 微软雅黑
        '/System/Library/Fonts/PingFang.ttc',  # macOS 苹方
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux 文泉驿
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except:
                continue

    # 如果所有尝试都失败，返回默认字体（可能无法显示中文）
    return pygame.font.SysFont(None, size)


# 创建中文字体
font = get_chinese_font(16)
large_font = get_chinese_font(24)


# 枚举类定义
class TrafficLightState(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    LEFT_GREEN = 3


class VehicleType(Enum):
    SEDAN = 0
    SUV = 1
    TRUCK = 2


class SensorType(Enum):
    CAMERA = 0
    LIDAR = 1
    RADAR = 2
    GPS = 3
    IMU = 4


class PredictionType(Enum):
    CONSTANT_VELOCITY = 0
    CONSTANT_ACCELERATION = 1
    MANEUVERING = 2


class SLAMState(Enum):
    INITIALIZING = 0
    TRACKING = 1
    LOST = 2


class BEVViewMode(Enum):
    NORMAL = 0
    OCCUPANCY_GRID = 1
    ELEVATION = 2


class PIDControllerType(Enum):
    SPEED = 0
    STEERING = 1
    BRAKE = 2


# 数据类定义
@dataclass
class TrackedObject:
    id: int
    position: 'Vector2'
    velocity: 'Vector2'
    acceleration: 'Vector2'
    object_type: str  # "vehicle", "pedestrian", "obstacle"
    size: 'Vector2'
    confidence: float
    last_update: float
    prediction_horizon: List['Vector2'] = None

    def __post_init__(self):
        if self.prediction_horizon is None:
            self.prediction_horizon = []


# PID控制器类
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=10.0, output_limit=1.0,
                 controller_type=PIDControllerType.SPEED):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.integral_limit = integral_limit  # 积分限幅
        self.output_limit = output_limit  # 输出限幅
        self.controller_type = controller_type  # 控制器类型

        # 状态变量
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()
        self.output = 0.0

        # 调试信息
        self.debug_info = {
            "error": 0.0,
            "p_term": 0.0,
            "i_term": 0.0,
            "d_term": 0.0
        }

    def reset(self):
        """重置控制器状态"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()
        self.output = 0.0

        # 重置调试信息
        self.debug_info = {
            "error": 0.0,
            "p_term": 0.0,
            "i_term": 0.0,
            "d_term": 0.0
        }

    def update(self, setpoint, process_variable, dt=None):
        """更新PID控制器

        Args:
            setpoint: 设定值
            process_variable: 过程变量（当前值）
            dt: 时间步长（秒），如果为None则自动计算

        Returns:
            output: 控制器输出
        """
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.01  # 默认时间步长
        self.previous_time = current_time

        # 计算误差
        error = setpoint - process_variable

        # 比例项
        p_term = self.kp * error

        # 积分项（带限幅）
        self.integral += error * dt
        # 积分抗饱和
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral

        # 微分项
        d_error = (error - self.previous_error) / dt if dt > 0 else 0
        d_term = self.kd * d_error

        # 保存当前误差用于下一次计算
        self.previous_error = error

        # 计算输出
        self.output = p_term + i_term + d_term

        # 输出限幅
        if self.output > self.output_limit:
            self.output = self.output_limit
        elif self.output < -self.output_limit:
            self.output = -self.output_limit

        # 保存调试信息
        self.debug_info = {
            "error": error,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term
        }

        return self.output

    def get_debug_info(self):
        """获取调试信息"""
        return self.debug_info


# 向量计算类
class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vector2({self.x:.2f}, {self.y:.2f})"

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self):
        length = self.length()
        if length > 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2()

    def rotate(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector2(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    def to_tuple(self):
        return (self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other):
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))

    def to_numpy(self):
        return np.array([self.x, self.y])


# 3D向量类
class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3()

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])


# 位姿类（位置和方向）
class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position if position else Vector3()
        self.orientation = orientation if orientation else Vector3()  # 欧拉角（roll, pitch, yaw）

    def __repr__(self):
        return f"Pose(pos={self.position}, orient={self.orientation})"

    def to_matrix(self):
        """将位姿转换为4x4齐次变换矩阵"""
        # 计算旋转矩阵（使用欧拉角）
        roll, pitch, yaw = self.orientation.x, self.orientation.y, self.orientation.z

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # ZYX旋转顺序
        rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        # 构建齐次变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [self.position.x, self.position.y, self.position.z]

        return transform

    def from_matrix(self, matrix):
        """从4x4齐次变换矩阵恢复位姿"""
        self.position = Vector3(matrix[0, 3], matrix[1, 3], matrix[2, 3])

        # 从旋转矩阵提取欧拉角（ZYX顺序）
        sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(matrix[2, 1], matrix[2, 2])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            roll = math.atan2(-matrix[1, 2], matrix[1, 1])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = 0

        self.orientation = Vector3(roll, pitch, yaw)

        return self


# 卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=1.0):
        # 状态向量: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)

        # 状态转移矩阵 (恒定加速度模型)
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 测量矩阵 (只测量位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # 过程噪声协方差
        self.Q = np.eye(6) * process_noise

        # 测量噪声协方差
        self.R = np.eye(2) * measurement_noise

        # 误差协方差矩阵
        self.P = np.eye(6)

        # 时间步长
        self.dt = dt

    def predict(self):
        # 预测状态
        self.state = np.dot(self.F, self.state)

        # 预测误差协方差
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.state[:2]  # 返回预测的位置

    def update(self, measurement):
        # 计算卡尔曼增益
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态估计
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)

        # 更新误差协方差
        I = np.eye(self.state.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return self.state[:2]  # 返回更新后的位置


# 轨迹预测器类
class TrajectoryPredictor:
    def __init__(self, prediction_horizon=3.0, time_step=0.1):
        self.prediction_horizon = prediction_horizon  # 预测时间范围（秒）
        self.time_step = time_step  # 预测时间步长（秒）
        self.num_steps = int(prediction_horizon / time_step)

    def predict_constant_velocity(self, position, velocity, num_steps=None):
        """恒定速度模型预测"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        for i in range(1, num_steps + 1):
            t = i * self.time_step
            predicted_pos = position + velocity * t
            trajectory.append(predicted_pos)

        return trajectory

    def predict_constant_acceleration(self, position, velocity, acceleration, num_steps=None):
        """恒定加速度模型预测"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        for i in range(1, num_steps + 1):
            t = i * self.time_step
            predicted_pos = position + velocity * t + acceleration * (0.5 * t * t)
            trajectory.append(predicted_pos)

        return trajectory

    def predict_maneuvering(self, position, velocity, acceleration, maneuver_probability=0.1, num_steps=None):
        """考虑机动性的预测模型"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        current_velocity = velocity
        current_position = position

        for i in range(1, num_steps + 1):
            t = i * self.time_step

            # 随机引入机动性（转向或加减速）
            if random.random() < maneuver_probability:
                # 随机改变加速度
                acceleration = acceleration + Vector2(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                )

            # 更新速度和位置
            current_velocity = current_velocity + acceleration * self.time_step
            current_position = current_position + current_velocity * self.time_step

            trajectory.append(current_position)

        return trajectory

    def predict_trajectory(self, position, velocity, acceleration=None,
                           prediction_type=PredictionType.CONSTANT_VELOCITY):
        """根据指定预测类型预测轨迹"""
        if prediction_type == PredictionType.CONSTANT_VELOCITY:
            return self.predict_constant_velocity(position, velocity)
        elif prediction_type == PredictionType.CONSTANT_ACCELERATION and acceleration is not None:
            return self.predict_constant_acceleration(position, velocity, acceleration)
        elif prediction_type == PredictionType.MANEUVERING and acceleration is not None:
            return self.predict_maneuvering(position, velocity, acceleration)
        else:
            # 默认使用恒定速度模型
            return self.predict_constant_velocity(position, velocity)


# 传感器融合类
class SensorFusion:
    def __init__(self, association_threshold=50.0, max_age=1.0):
        self.tracked_objects = {}  # 跟踪的对象字典 {id: TrackedObject}
        self.next_id = 0
        self.association_threshold = association_threshold  # 关联阈值（像素）
        self.max_age = max_age  # 最大未更新时长（秒）
        self.kalman_filters = {}  # 卡尔曼滤波器字典 {id: KalmanFilter}
        self.trajectory_predictor = TrajectoryPredictor()

    def update(self, detections, current_time):
        """更新传感器融合状态"""
        # 移除过期的跟踪对象
        self.remove_old_objects(current_time)

        # 如果没有检测到任何物体，直接返回
        if not detections:
            return list(self.tracked_objects.values())

        # 关联检测结果与现有跟踪对象
        matched_detections = set()
        matched_tracks = set()

        # 计算所有检测结果与跟踪对象之间的距离
        distance_matrix = []
        for detection in detections:
            row = []
            for obj_id, tracked_obj in self.tracked_objects.items():
                distance = detection.position.distance_to(tracked_obj.position)
                row.append(distance)
            distance_matrix.append(row)

        # 简单的最近邻关联
        for i, detection in enumerate(detections):
            if not distance_matrix[i]:
                continue

            min_distance = min(distance_matrix[i])
            if min_distance < self.association_threshold:
                min_index = distance_matrix[i].index(min_distance)
                obj_id = list(self.tracked_objects.keys())[min_index]

                # 更新跟踪对象
                self.update_tracked_object(obj_id, detection, current_time)
                matched_detections.add(i)
                matched_tracks.add(obj_id)

        # 处理未匹配的检测结果（创建新的跟踪对象）
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self.create_new_track(detection, current_time)

        # 处理未匹配的跟踪对象（预测其状态）
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_tracks:
                self.predict_tracked_object(obj_id, current_time)

        return list(self.tracked_objects.values())

    def create_new_track(self, detection, current_time):
        """创建新的跟踪对象"""
        obj_id = self.next_id
        self.next_id += 1

        # 初始化卡尔曼滤波器
        kf = KalmanFilter()
        kf.state[:2] = np.array([detection.position.x, detection.position.y])

        # 创建跟踪对象
        tracked_obj = TrackedObject(
            id=obj_id,
            position=detection.position,
            velocity=Vector2(0, 0),  # 初始速度为0
            acceleration=Vector2(0, 0),  # 初始加速度为0
            object_type=detection.object_type,
            size=detection.size,
            confidence=detection.confidence,
            last_update=current_time
        )

        # 预测轨迹
        tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
            detection.position,
            Vector2(0, 0),  # 初始速度为0
            prediction_type=PredictionType.CONSTANT_VELOCITY
        )

        self.tracked_objects[obj_id] = tracked_obj
        self.kalman_filters[obj_id] = kf

    def update_tracked_object(self, obj_id, detection, current_time):
        """更新跟踪对象"""
        tracked_obj = self.tracked_objects[obj_id]
        kf = self.kalman_filters[obj_id]

        # 使用卡尔曼滤波器更新状态
        measurement = np.array([detection.position.x, detection.position.y])
        kf.update(measurement)

        # 计算速度和加速度（简单差分）
        dt = current_time - tracked_obj.last_update
        if dt > 0:
            new_velocity = (detection.position - tracked_obj.position) / dt
            new_acceleration = (new_velocity - tracked_obj.velocity) / dt

            # 更新跟踪对象
            tracked_obj.position = detection.position
            tracked_obj.velocity = new_velocity
            tracked_obj.acceleration = new_acceleration
            tracked_obj.confidence = detection.confidence
            tracked_obj.last_update = current_time

            # 预测轨迹
            tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
                detection.position,
                new_velocity,
                new_acceleration,
                prediction_type=PredictionType.CONSTANT_ACCELERATION
            )

    def predict_tracked_object(self, obj_id, current_time):
        """预测跟踪对象的状态"""
        tracked_obj = self.tracked_objects[obj_id]
        kf = self.kalman_filters[obj_id]

        # 使用卡尔曼滤波器预测状态
        predicted_position = kf.predict()

        # 更新跟踪对象的位置（仅预测，不更新速度和加速度）
        tracked_obj.position = Vector2(predicted_position[0], predicted_position[1])

        # 预测轨迹（基于最后已知的速度和加速度）
        tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
            tracked_obj.position,
            tracked_obj.velocity,
            Vector2(0, 0),
            prediction_type=PredictionType.CONSTANT_ACCELERATION
        )

    def remove_old_objects(self, current_time):
        """移除长时间未更新的跟踪对象"""
        objects_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if current_time - tracked_obj.last_update > self.max_age:
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]

    def get_collision_risk(self, ego_position, ego_velocity, time_horizon=3.0, safety_distance=20.0):
        """计算与所有跟踪对象的碰撞风险"""
        collision_risks = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            # 计算自我车辆的未来轨迹
            ego_trajectory = self.trajectory_predictor.predict_trajectory(
                ego_position, ego_velocity, Vector2(0, 0),
                prediction_type=PredictionType.CONSTANT_VELOCITY
            )

            # 计算最小距离
            min_distance = float('inf')
            collision_time = float('inf')

            for i, (ego_pos, obj_pos) in enumerate(zip(ego_trajectory, tracked_obj.prediction_horizon)):
                distance = ego_pos.distance_to(obj_pos)
                if distance < min_distance:
                    min_distance = distance
                    collision_time = i * self.trajectory_predictor.time_step

                # 如果距离小于安全距离，标记为碰撞风险
                if distance<safety_distance:
                    collision_risks.append({
                        'object_id': obj_id,
                        'object_type': tracked_obj.object_type,
                        'min_distance': min_distance,
                        'collision_time': collision_time,
                        'risk_level': self.calculate_risk_level(min_distance, collision_time)
                    })
                    break

        return collision_risks

    def calculate_risk_level(self, distance, time_to_collision):
        """计算风险等级"""
        if time_to_collision < 1.0 and distance < 10.0:
            return "高危"
        elif time_to_collision < 2.0 and distance < 20.0:
            return "中危"
        elif time_to_collision < 3.0 and distance < 30.0:
            return "低危"
        else:
            return "无风险"


# 检测结果类
class Detection:
    def __init__(self, position, object_type, size, confidence=1.0):
        self.position = position
        self.object_type = object_type
        self.size = size
        self.confidence = confidence


# SLAM系统类
class SLAMSystem:
    def __init__(self, map_size=(800, 800), grid_size=0.5):
        self.map_size = map_size  # 地图尺寸 (宽度, 高度)
        self.grid_size = grid_size  # 网格大小（米）

        # 初始化地图
        self.occupancy_grid = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
        self.elevation_map = np.zeros((map_size[1], map_size[0]), dtype=np.float32)

        # 初始化位姿
        self.current_pose = Pose(Vector3(map_size[0] // 2, map_size[1] // 2, 0), Vector3(0, 0, 0))
        self.estimated_pose = self.current_pose

        # 地标和特征点
        self.landmarks = []  # 存储地标位置
        self.landmark_descriptors = []  # 存储地标特征描述符
        self.landmark_covariances = []  # 存储地标协方差

        # 粒子滤波器用于定位
        self.num_particles = 100
        self.particles = []  # 粒子列表，每个粒子是一个位姿
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

        # 状态
        self.state = SLAMState.INITIALIZING
        self.initialization_counter = 0

        # 建图参数
        self.map_update_rate = 0.1  # 地图更新率
        self.observation_range = 50  # 观测范围（像素）

        # 创建KD树用于快速特征匹配
        self.landmark_tree = None
        self.update_landmark_tree()

    def update_landmark_tree(self):
        """更新地标KD树"""
        if self.landmarks:
            points = np.array([(lm.x, lm.y) for lm in self.landmarks])
            self.landmark_tree = KDTree(points)
        else:
            self.landmark_tree = None

    def predict(self, linear_velocity, angular_velocity, dt):
        """预测步骤：根据运动模型更新位姿估计"""
        # 更新当前位姿（真实位姿，仅用于仿真）
        self.current_pose.position.x += linear_velocity * math.cos(self.current_pose.orientation.z) * dt
        self.current_pose.position.y += linear_velocity * math.sin(self.current_pose.orientation.z) * dt
        self.current_pose.orientation.z += angular_velocity * dt

        # 更新估计位姿（带有噪声）
        noise_scale = 0.1
        estimated_linear_velocity = linear_velocity + random.gauss(0, noise_scale)
        estimated_angular_velocity = angular_velocity + random.gauss(0, noise_scale * 0.1)

        self.estimated_pose.position.x += estimated_linear_velocity * math.cos(self.estimated_pose.orientation.z) * dt
        self.estimated_pose.position.y += estimated_linear_velocity * math.sin(self.estimated_pose.orientation.z) * dt
        self.estimated_pose.orientation.z += estimated_angular_velocity * dt

        # 更新粒子滤波器
        self.update_particles(linear_velocity, angular_velocity, dt)

    def update_particles(self, linear_velocity, angular_velocity, dt):
        """更新粒子滤波器"""
        for i in range(len(self.particles)):
            # 添加运动噪声
            noisy_linear = linear_velocity + random.gauss(0, 0.1)
            noisy_angular = angular_velocity + random.gauss(0, 0.01)
            #print(self.particles)
            # 更新粒子位姿
            self.particles[i].position.x += noisy_linear * math.cos(self.particles[i].orientation.z) * dt
            self.particles[i].position.y += noisy_linear * math.sin(self.particles[i].orientation.z) * dt
            self.particles[i].orientation.z += noisy_angular * dt

    def update(self, observations):
        """更新步骤：根据观测更新地图和位姿估计"""
        if self.state == SLAMState.INITIALIZING:
            self.initialization_counter += 1
            if self.initialization_counter > 10:  # 初始化完成
                self.state = SLAMState.TRACKING
                # 初始化粒子滤波器
                self.initialize_particles()
            return

        # 提取观测中的特征点
        observed_features = self.extract_features(observations)

        # 数据关联：将观测特征与地图中的地标匹配
        matches = self.data_association(observed_features)

        # 更新地标
        self.update_landmarks(observed_features, matches)

        # 更新位姿估计
        self.update_pose_estimation(observed_features, matches)

        # 更新地图
        self.update_map(observations)

        # 重采样粒子
        self.resample_particles()

    def initialize_particles(self):
        """初始化粒子滤波器"""
        self.particles = []
        for _ in range(self.num_particles):
            # 在初始位姿周围随机分布粒子
            noise_pos = random.gauss(0, 10)  # 位置噪声
            noise_angle = random.gauss(0, 0.1)  # 角度噪声

            particle = Pose(
                Vector3(
                    self.estimated_pose.position.x + noise_pos,
                    self.estimated_pose.position.y + noise_pos,
                    self.estimated_pose.position.z
                ),
                Vector3(
                    self.estimated_pose.orientation.x,
                    self.estimated_pose.orientation.y,
                    self.estimated_pose.orientation.z + noise_angle
                )
            )
            self.particles.append(particle)

    def extract_features(self, observations):
        """从观测中提取特征点（简化版）"""
        features = []
        for obs in observations:
            # 简化特征提取：直接将观测位置作为特征
            features.append({
                'position': obs.position,
                'descriptor': np.random.rand(10)  # 随机特征描述符
            })
        return features

    def data_association(self, observed_features):
        """数据关联：将观测特征与地图中的地标匹配"""
        matches = []

        if not self.landmark_tree or not observed_features:
            return matches

        # 获取观测特征的位置
        observed_positions = np.array([(f['position'].x, f['position'].y) for f in observed_features])

        # 使用KD树查找最近邻
        distances, indices = self.landmark_tree.query(observed_positions, k=1)

        # 创建匹配对
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < 20.0:  # 匹配阈值
                matches.append({
                    'observed_idx': i,
                    'landmark_idx': idx,
                    'distance': dist
                })

        return matches

    def update_landmarks(self, observed_features, matches):
        """更新地标"""
        # 更新已匹配的地标
        for match in matches:
            obs_idx = match['observed_idx']
            lm_idx = match['landmark_idx']

            # 简化更新：移动地标位置向观测位置靠近
            alpha = 0.1  # 学习率
            observed_pos = observed_features[obs_idx]['position']
            self.landmarks[lm_idx] = self.landmarks[lm_idx] * (1 - alpha) + observed_pos * alpha

        # 添加新的地标
        matched_obs_indices = [m['observed_idx'] for m in matches]
        for i, feature in enumerate(observed_features):
            if i not in matched_obs_indices:
                self.landmarks.append(feature['position'])
                self.landmark_descriptors.append(feature['descriptor'])
                self.landmark_covariances.append(np.eye(2) * 10.0)  # 初始协方差

        # 更新KD树
        self.update_landmark_tree()

    def update_pose_estimation(self, observed_features, matches):
        """更新位姿估计"""
        if not matches:
            return

        # 计算位姿更新（简化版）
        # 在实际SLAM中，这里会使用扩展卡尔曼滤波器或图优化

        # 更新粒子权重
        for i in range(self.num_particles):
            weight = 1.0
            for match in matches:
                obs_idx = match['observed_idx']
                lm_idx = match['landmark_idx']

                # 计算观测预期位置（基于粒子位姿）
                expected_obs = self.landmarks[lm_idx] - Vector2(self.particles[i].position.x,
                                                                self.particles[i].position.y)
                expected_obs = expected_obs.rotate(-math.degrees(self.particles[i].orientation.z))

                # 计算实际观测位置
                actual_obs = observed_features[obs_idx]['position']

                # 计算误差
                error = expected_obs.distance_to(actual_obs)

                # 更新权重（误差越小权重越大）
                weight *= math.exp(-error * error / (2 * 10.0))  # 高斯分布

            self.particle_weights[i] = weight

        # 归一化权重
        self.particle_weights /= np.sum(self.particle_weights)

        # 选择最佳粒子作为位姿估计
        best_particle_idx = np.argmax(self.particle_weights)
        self.estimated_pose = self.particles[best_particle_idx]

    def resample_particles(self):
        """重采样粒子"""
        # 计算有效粒子数
        effective_particles = 1.0 / np.sum(self.particle_weights ** 2)

        # 如果有效粒子数太少，进行重采样
        if effective_particles < self.num_particles / 2:
            # 系统重采样
            indices = np.random.choice(
                range(self.num_particles),
                size=self.num_particles,
                p=self.particle_weights
            )

            new_particles = []
            for idx in indices:
                # 添加少量噪声以避免粒子退化
                noise_pos = random.gauss(0, 0.1)
                noise_angle = random.gauss(0, 0.01)

                new_particle = Pose(
                    Vector3(
                        self.particles[idx].position.x + noise_pos,
                        self.particles[idx].position.y + noise_pos,
                        self.particles[idx].position.z
                    ),
                    Vector3(
                        self.particles[idx].orientation.x,
                        self.particles[idx].orientation.y,
                        self.particles[idx].orientation.z + noise_angle
                    )
                )
                new_particles.append(new_particle)

            self.particles = new_particles
            self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def update_map(self, observations):
        """更新占据栅格地图"""
        # 获取当前位姿
        pose_x = int(self.estimated_pose.position.x)
        pose_y = int(self.estimated_pose.position.y)
        pose_angle = self.estimated_pose.orientation.z

        # 更新观测范围内的网格
        for obs in observations:
            # 转换观测到全局坐标系
            global_obs = Vector2(
                pose_x + obs.position.x * math.cos(pose_angle) - obs.position.y * math.sin(pose_angle),
                pose_y + obs.position.x * math.sin(pose_angle) + obs.position.y * math.cos(pose_angle)
            )

            # 确保坐标在地图范围内
            map_x = int(global_obs.x)
            map_y = int(global_obs.y)

            if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                # 更新占据概率（log odds形式）
                self.occupancy_grid[map_y, map_x] += math.log(9)  # 占据概率增加

        # 更新车辆当前位置周围的空闲区域
        for angle in np.linspace(0, 2 * math.pi, 36):
            for r in range(1, self.observation_range):
                map_x = int(pose_x + r * math.cos(pose_angle + angle))
                map_y = int(pose_y + r * math.sin(pose_angle + angle))

                if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                    # 如果这个位置没有被占据，减少占据概率
                    if self.occupancy_grid[map_y, map_x] < math.log(9):
                        self.occupancy_grid[map_y, map_x] -= math.log(9) * 0.1  # 空闲概率增加

                # 如果遇到占据网格，停止更新（假设光线被阻挡）
                if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                    if self.occupancy_grid[map_y, map_x] > 0:
                        break

    def get_occupancy_probability(self, x, y):
        """获取指定位置的占据概率"""
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            odds = math.exp(self.occupancy_grid[y, x])
            return odds / (1 + odds)
        return 0.5  # 未知区域

    def draw(self, screen, offset_x=0, offset_y=0, scale=1.0):
        """绘制SLAM地图"""
        # 绘制占据栅格地图
        for y in range(0, self.map_size[1], 5):
            for x in range(0, self.map_size[0], 5):
                prob = self.get_occupancy_probability(x, y)
                color_value = int(255 * (1 - prob))  # 占据越大概率颜色越深
                color = (color_value, color_value, color_value)

                rect = pygame.Rect(
                    offset_x + x * scale,
                    offset_y + y * scale,
                    5 * scale,
                    5 * scale
                )
                pygame.draw.rect(screen, color, rect)

        # 绘制地标
        for landmark in self.landmarks:
            pygame.draw.circle(
                screen,
                GREEN,
                (offset_x + int(landmark.x * scale), offset_y + int(landmark.y * scale)),
                3
            )

        # 绘制估计位姿
        pygame.draw.circle(
            screen,
            BLUE,
            (offset_x + int(self.estimated_pose.position.x * scale),
             offset_y + int(self.estimated_pose.position.y * scale)),
            5
        )

        # 绘制方向指示
        end_x = offset_x + int(
            self.estimated_pose.position.x * scale + 15 * math.cos(self.estimated_pose.orientation.z) * scale)
        end_y = offset_y + int(
            self.estimated_pose.position.y * scale + 15 * math.sin(self.estimated_pose.orientation.z) * scale)
        pygame.draw.line(
            screen,
            BLUE,
            (offset_x + int(self.estimated_pose.position.x * scale),
             offset_y + int(self.estimated_pose.position.y * scale)),
            (end_x, end_y),
            2
        )

        # 绘制粒子
        for particle in self.particles:
            pygame.draw.circle(
                screen,
                (255, 100, 100, 100),  # 半透明红色
                (offset_x + int(particle.position.x * scale), offset_y + int(particle.position.y * scale)),
                2
            )

# BEV系统类
class BEVSystem:
    def __init__(self, width=400, height=400, scale=0.5):
        self.width = width
        self.height = height
        self.scale = scale  # 像素/米
        self.view_mode = BEVViewMode.NORMAL
        self.surface = pygame.Surface((width, height))
        self.ego_vehicle = None
        self.visible_objects = []

    def update(self, ego_vehicle, objects, roads, obstacles, traffic_lights):
        """更新BEV视图"""
        self.ego_vehicle = ego_vehicle
        self.visible_objects = objects

        # 清空表面
        self.surface.fill(LIGHT_GRAY)

        # 绘制道路
        for road in roads:
            self.draw_road(road)

        # 绘制障碍物
        for obstacle in obstacles:
            self.draw_obstacle(obstacle)

        # 绘制交通灯
        for light in traffic_lights:
            self.draw_traffic_light(light)

        # 绘制其他车辆和行人
        for obj in objects:
            if hasattr(obj, 'type') and obj.type in ['vehicle', 'pedestrian']:
                self.draw_object(obj)

        # 绘制自我车辆
        self.draw_ego_vehicle(ego_vehicle)

        # 绘制传感器范围
        self.draw_sensor_range(ego_vehicle)

    def draw_road(self, road):
        """绘制道路"""
        # 转换道路坐标到BEV坐标系
        start_x = int(road.start.x * self.scale)
        start_y = int(road.start.y * self.scale)
        end_x = int(road.end.x * self.scale)
        end_y = int(road.end.y * self.scale)
        width = int(road.width * self.scale)

        # 计算道路方向向量和法向量
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            return

        # 归一化方向向量
        dx /= length
        dy /= length

        # 计算法向量
        nx = -dy
        ny = dx

        # 计算道路边界点
        half_width = width / 2
        points = [
            (start_x + nx * half_width, start_y + ny * half_width),
            (end_x + nx * half_width, end_y + ny * half_width),
            (end_x - nx * half_width, end_y - ny * half_width),
            (start_x - nx * half_width, start_y - ny * half_width)
        ]

        # 绘制道路
        pygame.draw.polygon(self.surface, GRAY, points)

        # 绘制车道极地
        lane_width = width / road.lanes
        for i in range(1, road.lanes):
            offset = half_width - i * lane_width
            line_start = (start_x + nx * offset, start_y + ny * offset)
            line_end = (end_x + nx * offset, end_y + ny * offset)
            pygame.draw.line(self.surface, WHITE, line_start, line_end, 1)

        # 绘制道路中心线
        center_start = (start_x, start_y)
        center_end = (end_x, end_y)
        pygame.draw.line(self.surface, YELLOW, center_start, center_end, 2)

    def draw_obstacle(self, obstacle):
        """绘制障碍物"""
        x = int(obstacle.position.x * self.scale)
        y = int(obstacle.position.y * self.scale)
        width = int(obstacle.size.x * self.scale)
        height = int(obstacle.size.y * self.scale)

        if obstacle.type == "cone":
            # 绘制锥形障碍物
            points = [
                (x, y - height // 2),
                (x - width // 2, y + height // 2),
                (x + width // 2, y + height // 2)
            ]
            pygame.draw.polygon(self.surface, ORANGE, points)
        else:
            # 绘制矩形障碍物
            pygame.draw.rect(self.surface, RED, (x - width // 2, y - height // 2, width, height))

    def draw_traffic_light(self, light):
        """绘制交通灯"""
        x = int(light.position.x * self.scale)
        y = int(light.position.y * self.scale)

        # 根据交通灯状态选择颜色
        if light.state == TrafficLightState.RED:
            color = RED
        elif light.state == TrafficLightState.YELLOW:
            color = YELLOW
        elif light.state == TrafficLightState.GREEN:
            color = GREEN
        else:  # LEFT_GREEN
            color = GREEN

        pygame.draw.circle(self.surface, color, (x, y), 5)

    def draw_object(self, obj):
        """绘制其他车辆和行人"""
        x = int(obj.position.x * self.scale)
        y = int(obj.position.y * self.scale)

        if obj.object_type == "vehicle":
            # 绘制车辆
            width = int(obj.size.x * self.scale)
            height = int(obj.size.y * self.scale)

            # 创建车辆矩形
            vehicle_rect = pygame.Rect(0, 0, width, height)
            vehicle_rect.center = (x, y)

            # 绘制车辆
            pygame.draw.rect(self.surface, BLUE, vehicle_rect)

            # 绘制方向指示
            angle_rad = math.radians(getattr(obj, 'angle', 0))
            end_x = x + int(15 * math.cos(angle_rad))
            end_y = y + int(15 * math.sin(angle_rad))
            pygame.draw.line(self.surface, BLACK, (x, y), (end_x, end_y), 2)

        elif obj.object_type == "pedestrian":
            # 绘制行人
            pygame.draw.circle(self.surface, MAGENTA, (x, y), 5)

    def draw_ego_vehicle(self, vehicle):
        """绘制自我车辆"""
        x = int(vehicle.position.x * self.scale)
        y = int(vehicle.position.y * self.scale)
        width = int(vehicle.size.x * self.scale)
        height = int(vehicle.size.y * self.scale)

        # 创建车辆矩形
        vehicle_rect = pygame.Rect(0, 0, width, height)
        vehicle_rect.center = (x, y)

        # 绘制车辆
        pygame.draw.rect(self.surface, GREEN, vehicle_rect)

        # 绘制方向指示
        angle_rad = math.radians(vehicle.angle)
        end_x = x + int(20 * math.cos(angle_rad))
        end_y = y + int(20 * math.sin(angle_rad))
        pygame.draw.line(self.surface, BLACK, (x, y), (end_x, end_y), 2)

    def draw_sensor_range(self, vehicle):
        """绘制传感器范围"""
        x = int(vehicle.position.x * self.scale)
        y = int(vehicle.position.y * self.scale)

        # 绘制传感器检测范围
        for sensor in vehicle.sensors:
            # 计算传感器位置
            sensor_pos = vehicle.position + sensor.position.rotate(vehicle.angle)
            sensor_x = int(sensor_pos.x * self.scale)
            sensor_y = int(sensor_pos.y * self.scale)

            # 计算传感器方向
            sensor_angle = math.radians(vehicle.angle + sensor.angle)

            # 计算传感器范围弧线
            start_angle = sensor_angle - math.radians(sensor.fov / 2)
            end_angle = sensor_angle + math.radians(sensor.fov / 2)

            # 绘制传感器范围
            pygame.draw.arc(
                self.surface,
                CYAN,
                (sensor_x - sensor.range * self.scale,
                 sensor_y - sensor.range * self.scale,
                 sensor.range * 2 * self.scale,
                 sensor.range * 2 * self.scale),
                start_angle,
                end_angle,
                1
            )

            # 绘制传感器位置
            pygame.draw.circle(self.surface, CYAN, (sensor_x, sensor_y), 3)

    def render(self, screen, x, y):
        """将BEV视图渲染到主屏幕"""
        # 绘制BEV视图边框
        pygame.draw.rect(screen, BLACK, (x - 2, y - 2, self.width + 4, self.height + 4), 2)

        # 绘制BEV视图
        screen.blit(self.surface, (x, y))

        # 绘制标题
        title_text = "鸟瞰图 (BEV)"
        title_surface = font.render(title_text, True, BLACK)
        screen.blit(title_surface, (x + 10, y + 10))

        # 绘制视图模式
        mode_text = f"模式: {self.view_mode.name}"
        mode_surface = font.render(mode_text, True, BLACK)
        screen.blit(mode_surface, (x + 10, y + 30))

        # 绘制比例尺
        scale_text = f"比例: 1px = {1 / self.scale:.2f}m"
        scale_surface = font.render(scale_text, True, BLACK)
        screen.blit(scale_surface, (x + 10, y + 50))

# 道路类
class Road:
    def __init__(self, start, end, width=80, lanes=2):
        self.start = start
        self.end = end
        self.width = width
        self.lanes = lanes
        self.length = (end - start).length()
        self.direction = (end - start).normalized()
        self.normal = Vector2(-self.direction.y, self.direction.x)

        # 计算道路边界
        half_width = width / 2
        self.left_boundary = [
            start + self.normal * half_width,
            end + self.normal * half_width
        ]
        self.right_boundary = [
            start - self.normal * half_width,
            end - self.normal * half_width
        ]

        # 计算车道中心线
        self.lane_centers = []
        lane_width = width / lanes
        for i in range(lanes):
            offset = half_width - (i + 0.5) * lane_width
            self.lane_centers.append([
                start + self.normal * offset,
                end + self.normal * offset
            ])

    def get_lane_center(self, lane_index, t):
        """获取指定车道上t位置的点 (0 <= t <= 1)"""
        if lane_index < 0 or lane_index >= self.lanes:
            lane_index = 0

        start_point = self.lane_centers[lane_index][0]
        end_point = self.lane_centers[lane_index][1]
        return start_point + (end_point - start_point) * t

    def get_lane_direction(self):
        """获取车道方向"""
        return self.direction

    def draw(self, screen):
        # 绘制道路底色
        road_points = [
            self.left_boundary[0].to_tuple(),
            self.left_boundary[1].to_tuple(),
            self.right_boundary[1].to_tuple(),
            self.right_boundary[0].to_tuple()
        ]
        pygame.draw.polygon(screen, GRAY, road_points)

        # 绘制道路中心线
        center_start = self.start
        center_end = self.end
        pygame.draw.line(screen, YELLOW, center_start.to_tuple(), center_end.to_tuple(), 2)

        # 绘制车道线
        lane_width = self.width / self.lanes
        for i in range(1, self.lanes):
            offset = self.normal * (self.width / 2 - i * lane_width)
        lane_start = self.start + offset
        lane_end = self.end + offset
        pygame.draw.line(screen, WHITE, lane_start.to_tuple(), lane_end.to_tuple(), 2)

        # 绘制道路边缘
        pygame.draw.line(screen, WHITE, self.left_boundary[0].to_tuple(), self.left_boundary[1].to_tuple(), 2)
        pygame.draw.line(screen, WHITE, self.right_boundary[0].to_tuple(), self.right_boundary[1].to_tuple(), 2)

        # 交叉口类


# 交叉口类
class Intersection:
    def __init__(self, position, radius=50):
        self.position = position
        self.radius = radius
        self.connected_roads = []

    def add_road(self, road):
        self.connected_roads.append(road)

    def draw(self, screen):
        pygame.draw.circle(screen, LIGHT_GRAY, self.position.to_tuple(), self.radius)
        pygame.draw.circle(screen, BLACK, self.position.to_tuple(), self.radius, 2)


class TrafficLight:
    def __init__(self, position, road, direction="horizontal"):
        self.position = position
        self.road = road
        self.state = TrafficLightState.RED
        self.timer = 0
        self.cycle_times = {
            TrafficLightState.RED: 10.0,
            TrafficLightState.YELLOW: 3.0,
            TrafficLightState.GREEN: 15.0,
            TrafficLightState.LEFT_GREEN: 5.0
        }
        self.direction = direction  # "horizontal" or "vertical"
        self.left_turn_state = False

    def update(self, dt):
        self.timer += dt
        current_cycle_time = self.cycle_times[self.state]

        if self.timer >= current_cycle_time:
            self.timer = 0
            if self.state == TrafficLightState.RED:
                self.state = TrafficLightState.GREEN
            elif self.state == TrafficLightState.GREEN:
                self.state = TrafficLightState.YELLOW
            elif self.state == TrafficLightState.YELLOW:
                self.state = TrafficLightState.RED
            elif self.state == TrafficLightState.LEFT_GREEN:
                self.state = TrafficLightState.GREEN
                self.left_turn_state = False

    def set_left_turn(self):
        if self.state == TrafficLightState.GREEN and not self.left_turn_state:
            self.state = TrafficLightState.LEFT_GREEN
            self.timer = 0
            self.left_turn_state = True

    def draw(self, screen):
        # 绘制交通灯支柱
        pygame.draw.rect(screen, GRAY, (self.position.x - 5, self.position.y - 100, 10, 100))

        # 绘制灯箱
        pygame.draw.rect(screen, BLACK, (self.position.x - 15, self.position.y - 120, 30, 60))

        # 绘制信号灯
        red_color = RED if self.state == TrafficLightState.RED else DARK_GRAY
        yellow_color = YELLOW if self.state == TrafficLightState.YELLOW else DARK_GRAY
        green_color = GREEN if self.state == TrafficLightState.GREEN or self.state == TrafficLightState.LEFT_GREEN else DARK_GRAY

        pygame.draw.circle(screen, red_color, (self.position.x, self.position.y - 110), 8)
        pygame.draw.circle(screen, yellow_color, (self.position.x, self.position.y - 100), 8)
        pygame.draw.circle(screen, green_color, (self.position.x, self.position.y - 90), 8)

        # 绘制左转箭头
        if self.state == TrafficLightState.LEFT_GREEN:
            points = [
                (self.position.x, self.position.y - 85),
                (self.position.x - 5, self.position.y - 80),
                (self.position.x + 5, self.position.y - 80)
            ]
            pygame.draw.polygon(screen, GREEN, points)


# 障碍物类
class Obstacle:
    def __init__(self, position, size, obstacle_type="cone"):
        self.position = position
        self.size = size
        self.type = obstacle_type

    def draw(self, screen):
        if self.type == "cone":
            pygame.draw.polygon(screen, ORANGE, [
                (self.position.x, self.position.y - self.size.y / 2),
                (self.position.x - self.size.x / 2, self.position.y + self.size.y / 2),
                (self.position.x + self.size.x / 2, self.position.y + self.size.y / 2)
            ])
        else:  # 默认绘制矩形障碍物
            pygame.draw.rect(screen, RED, (
                self.position.x - self.size.x / 2,
                self.position.y - self.size.y / 2,
                self.size.x,
                self.size.y
            ))


# 行人类
class Pedestrian:
    def __init__(self, position, speed=1.0):
        self.position = position
        self.speed = speed
        self.direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalized()
        self.size = Vector2(10, 20)
        self.target = None
        self.waiting = False
        self.wait_time = 0
        self.velocity = Vector2(0, 0)
        self.acceleration = Vector2(0, 0)
        self.object_type = "pedestrian"

    def update(self, dt, traffic_lights):
        # 保存旧位置用于计算速度
        old_position = self.position

        # 检查是否需要等待红绿灯
        self.waiting = False
        for light in traffic_lights:
            dist = (light.position - self.position).length()
            if dist < 50 and light.state != TrafficLightState.GREEN:
                self.waiting = True
                break

        if not self.waiting:
            # 随机改变极地或设置新目标
            if self.target is None or random.random() < 0.005:
                self.target = Vector2(
                    random.randint(50, SCREEN_WIDTH - 50),
                    random.randint(50, SCREEN_HEIGHT - 50)
                )
                self.direction = (self.target - self.position).normalized()

            # 移动
            self.position += self.direction * self.speed * dt * 30

            # 边界检查
            if self.position.x < 20 or self.position.x > SCREEN_WIDTH - 20:
                self.direction.x *= -1
            if self.position.y < 20 or self.position.y > SCREEN_HEIGHT - 20:
                self.direction.y *= -1

        # 计算速度和加速度
        if dt > 0:
            self.velocity = (self.position - old_position) / dt
            # 简单假设加速度为0（实际应用中可以使用更复杂的模型）
            self.acceleration = Vector2(0, 0)

    def draw(self, screen):
        # 绘制身体
        pygame.draw.ellipse(screen, BLUE, (
            self.position.x - self.size.x / 2,
            self.position.y - self.size.y / 2,
            self.size.x,
            self.size.y
        ))

        # 绘制头部
        pygame.draw.circle(screen, BROWN, (self.position.x, self.position.y - self.size.y / 3), 5)


# 传感器基类
class Sensor:
    def __init__(self, vehicle, sensor_type, position, angle, range=100, fov=60):
        self.vehicle = vehicle
        self.type = sensor_type
        self.position = position
        self.angle = angle
        self.range = range
        self.fov = fov
        self.detected_objects = []

    def update(self, objects):
        self.detected_objects = []
        sensor_pos = self.vehicle.position + self.position.rotate(self.vehicle.angle)
        sensor_dir = Vector2(1, 0).rotate(self.vehicle.angle + self.angle)

        for obj in objects:
            if isinstance(obj, Vehicle):
                obj_type = "vehicle"
                obj_size = obj.size
                confidence = 0.9  # 车辆检测置信度较高
            elif isinstance(obj, Pedestrian):
                obj_type = "pedestrian"
                obj_size = obj.size
                confidence = 0.7  # 行人检测置信度中等
            elif isinstance(obj, Obstacle):
                obj_type = "obstacle"
                obj_size = obj.size
                confidence = 0.8  # 障碍物检测置信度较高
            else:
                continue

            obj_pos = obj.position
            to_obj = obj_pos - sensor_pos
            dist = to_obj.length()

            if dist < self.range:
                angle_to_obj = math.degrees(math.atan2(to_obj.y, to_obj.x) - math.atan2(sensor_dir.y, sensor_dir.x))
                if angle_to_obj < -180:
                    angle_to_obj += 360
                if angle_to_obj > 180:
                    angle_to_obj -= 360

                if abs(angle_to_obj) < self.fov / 2:
                    # 添加检测结果
                    detection = Detection(obj_pos, obj_type, obj_size, confidence)
                    self.detected_objects.append(detection)

    def draw(self, screen):
        sensor_pos = self.vehicle.position + self.position.rotate(self.vehicle.angle)
        sensor_dir = Vector2(1, 0).rotate(self.vehicle.angle + self.angle)

        # 绘制传感器范围
        left_angle = self.angle - self.fov / 2
        right_angle = self.angle + self.fov / 2

        left_dir = Vector2(1, 0).rotate(self.vehicle.angle + left_angle) * self.range
        right_dir = Vector2(1, 0).rotate(self.vehicle.angle + right_angle) * self.range

        pygame.draw.line(screen, CYAN, sensor_pos.to_tuple(), (sensor_pos + left_dir).to_tuple(), 1)
        pygame.draw.line(screen, CYAN, sensor_pos.to_tuple(), (sensor_pos + right_dir).to_tuple(), 1)

        # 绘制检测到的物体
        for detection in self.detected_objects:
            pygame.draw.line(screen, YELLOW, sensor_pos.to_tuple(), detection.position.to_tuple(), 1)


# 车辆动力学模型
class VehicleDynamics:
    def __init__(self, mass=1500, max_engine_force=3000, max_brake_force=5000, drag_coefficient=0.3):
        self.mass = mass
        self.max_engine_force = max_engine_force
        self.max_brake_force = max_brake_force
        self.drag_coefficient = drag_coefficient
        self.velocity = 0
        self.acceleration = 0
        self.engine_force = 0
        self.brake_force = 0
        self.steering_angle = 0
        self.max_steering_angle = 30

    def update(self, dt, throttle_input, brake_input, steering_input):
        # 计算引擎力
        self.engine_force = throttle_input * self.max_engine_force

        # 计算制动力
        self.brake_force = brake_input * self.max_brake_force

        # 计算转向角
        self.steering_angle = steering_input * self.max_steering_angle

        # 计算阻力
        drag_force = self.drag_coefficient * self.velocity * abs(self.velocity)

        # 计算总力
        # 计算总力
        total_force = self.engine_force - self.brake_force - drag_force

        # 计算加速度
        self.acceleration = total_force / self.mass

        # 更新速度
        self.velocity += self.acceleration * dt

        # 确保速度不为负
        if self.velocity < 0:
            self.velocity = 0


# 车辆类
class Vehicle:
    def __init__(self, position, angle=0, vehicle_type=VehicleType.SEDAN):
        self.position = position
        self.angle = angle
        self.type = vehicle_type
        self.dynamics = VehicleDynamics()

        # 根据车辆类型设置尺寸和颜色
        if vehicle_type == VehicleType.SEDAN:
            self.size = Vector2(40, 20)
            self.color = BLUE
        elif vehicle_type == VehicleType.SUV:
            self.size = Vector2(45, 22)
            self.color = GREEN
        else:  # TRUCK
            self.size = Vector2(60, 25)
            self.color = RED

        # 传感器
        self.sensors = [
            Sensor(self, SensorType.CAMERA, Vector2(20, 0), 0, 150, 60),  # 前摄像头
            Sensor(self, SensorType.LIDAR, Vector2(10, 8), 30, 200, 30),  # 左前激光雷达
            Sensor(self, SensorType.LIDAR, Vector2(10, -8), -30, 200, 30),  # 右前激光雷达
            Sensor(self, SensorType.RADAR, Vector2(-15, 0), 180, 100, 80)  # 后雷达
        ]

        # 传感器融合
        self.sensor_fusion = SensorFusion()
        self.tracked_objects = []
        self.collision_risks = []

        # SLAM系统
        self.slam = SLAMSystem()

        # PID控制器
        self.speed_pid = PIDController(kp=0.5, ki=0.1, kd=0.2, output_limit=1.0,
                                       controller_type=PIDControllerType.SPEED)
        self.steering_pid = PIDController(kp=0.8, ki=0.05, kd=0.3, output_limit=1.0,
                                          controller_type=PIDControllerType.STEERING)
        self.brake_pid = PIDController(kp=1.0, ki=0.2, kd=0.1, output_limit=1.0,
                                       controller_type=PIDControllerType.BRAKE)

        # 控制参数
        self.target_speed = 0
        self.target_angle = 0
        self.throttle = 0
        self.brake = 0
        self.steering = 0

        # 路径跟踪
        self.path = []
        self.current_waypoint = 0
        self.current_road = None
        self.current_lane = 0
        self.road_progress = 0  # 0到1之间，表示在道路上的位置

        # 导航
        self.destination = None
        self.route = []  # 路线上的道路列表
        self.current_route_index = 0

        # 预测和避障
        self.prediction_horizon = []
        self.avoidance_maneuver = None
        self.avoidance_timer = 0

        # 对象类型（用于传感器检测）
        self.object_type = "vehicle"

        # PID调试信息
        self.pid_debug_info = {
            "speed": {"error": 0, "p": 0, "i": 0, "d": 0},
            "steering": {"error": 0, "p": 0, "i": 0, "d": 0},
            "brake": {"error": 0, "p": 0, "i": 0, "d": 0}
        }

    def assign_to_road(self, road, lane=0):
        """将车辆分配到指定道路的车道上"""
        self.current_road = road
        self.current_lane = lane
        self.road_progress = 0
        self.position = road.get_lane_center(lane, 0)
        self.angle = math.degrees(math.atan2(road.direction.y, road.direction.x))

    def update(self, dt, objects, traffic_lights, roads):
        # 更新所有传感器
        all_detections = []
        for sensor in self.sensors:
            sensor.update(objects)
            all_detections.extend(sensor.detected_objects)

        # 传感器融合
        current_time = time.time()
        self.tracked_objects = self.sensor_fusion.update(all_detections, current_time)

        # 计算碰撞风险
        #ego_velocity = Vector2(self.dynamics.velocity, 0).rotate(self.angle)
        ego_velocity = Vector2(self.dynamics.velocity, 0).rotate(self.angle)
        self.collision_risks = self.sensor_fusion.get_collision_risk(
            self.position, ego_velocity
        )

        # 更新SLAM系统
        linear_velocity = self.dynamics.velocity
        angular_velocity = math.radians(self.dynamics.steering_angle) * linear_velocity / self.size.x
        self.slam.predict(linear_velocity, angular_velocity, dt)
        self.slam.update(all_detections)

        # 根据传感器数据和预测结果做出决策
        self.make_decision(objects, traffic_lights, roads)

        # 使用PID控制器更新控制输入
        self.update_pid_controllers(dt)

        # 更新动力学
        self.dynamics.update(dt, self.throttle, self.brake, self.steering)

        # 如果车辆在道路上行驶
        if self.current_road:
            # 计算前进距离
            distance = self.dynamics.velocity * dt

            # 更新道路进度
            road_length = self.current_road.length
            self.road_progress += distance / road_length

            # 如果到达道路尽头
            if self.road_progress >= 1:
                # 寻找下一段道路
                self.find_next_road(roads)
                self.road_progress = 0

            # 更新位置和角度
            self.position = self.current_road.get_lane_center(self.current_lane, self.road_progress)
            road_direction = self.current_road.get_lane_direction()
            self.angle = math.degrees(math.atan2(road_direction.y, road_direction.x))
        else:
            # 自由行驶模式（如果没有分配到道路）
            velocity_vec = Vector2(self.dynamics.velocity, 0).rotate(self.angle)
            self.position += velocity_vec * dt

            # 更新角度
            if abs(self.dynamics.velocity) > 0.1:
                turn_radius = self.size.x / math.tan(math.radians(self.dynamics.steering_angle))
                angular_velocity = self.dynamics.velocity / turn_radius if turn_radius != 0 else 0
                self.angle += math.degrees(angular_velocity) * dt

        # 更新避障计时器
        if self.avoidance_maneuver:
            self.avoidance_timer += dt
            if self.avoidance_timer > 5.0:  # 5秒后重置避障行为
                self.avoidance_maneuver = None
                self.avoidance_timer = 0

    def update_pid_controllers(self, dt):
        """使用PID控制器更新控制输入"""
        # 速度PID控制器
        throttle_output = self.speed_pid.update(self.target_speed, self.dynamics.velocity, dt)
        if throttle_output > 0:
            self.throttle = throttle_output
            self.brake = 0
        else:
            self.throttle = 0
            # 使用刹车PID控制器
            self.brake = self.brake_pid.update(0, -throttle_output, dt)

        # 转向PID控制器
        if self.current_road:
            # 在道路上行驶时，保持车道中心
            lane_center = self.current_road.get_lane_center(self.current_lane, self.road_progress)
            road_direction = self.current_road.get_lane_direction()
            target_angle = math.degrees(math.atan2(road_direction.y, road_direction.x))

            # 计算横向误差（车辆当前位置到车道中心的距离）
            lateral_error = (lane_center - self.position).dot(self.current_road.normal)

            # 使用横向误差作为转向PID的输入
            steering_output = self.steering_pid.update(0, lateral_error, dt)
            self.steering = steering_output
        else:
            # 自由行驶模式下的转向控制
            if self.path and self.current_waypoint < len(self.path):
                target_pos = self.path[self.current_waypoint]
                to_target = target_pos - self.position

                # 计算目标角度
                target_angle = math.degrees(math.atan2(to_target.y, to_target.x))

                # 计算角度误差
                angle_error = (target_angle - self.angle) % 360
                if angle_error > 180:
                    angle_error -= 360

                # 使用角度误差作为转向PID的输入
                steering_output = self.steering_pid.update(0, angle_error, dt)
                self.steering = steering_output

        # 保存PID调试信息
        self.pid_debug_info["speed"] = self.speed_pid.get_debug_info()
        self.pid_debug_info["steering"] = self.steering_pid.get_debug_info()
        self.pid_debug_info["brake"] = self.brake_pid.get_debug_info()

    def find_next_road(self, roads):
        """寻找下一段道路"""
        if not self.current_road:
            return

        # 简单实现：随机选择一条与当前道路方向相近的道路
        current_end = self.current_road.end
        possible_roads = []

        for road in roads:
            if road.start.distance_to(current_end) < 50:  # 如果道路起点接近当前道路终点
                # 计算方向相似度
                dot_product = self.current_road.direction.dot(road.direction)
                if dot_product > 0.7:  # 方向相似度阈值
                    possible_roads.append(road)

        if possible_roads:
            # 随机选择一条道路
            self.current_road = random.choice(possible_roads)
            self.road_progress = 0

    def make_decision(self, objects, traffic_lights, roads):
        # 决策逻辑：考虑传感器融合结果和预测信息

        # 检查碰撞风险
        emergency_brake = False
        avoidance_needed = False

        for risk in self.collision_risks:
            if risk['risk_level'] == "高危":
                emergency_brake = True
                break
            elif risk['risk_level'] == "中危":
                avoidance_needed = True

        # 检查交通灯
        traffic_light_detected = False
        for light in traffic_lights:
            dist = (light.position - self.position).length()
            if dist < 100 and abs(self.angle - math.atan2(light.position.y - self.position.y,
                                                          light.position.x - self.position.x)) < 30:
                if light.state != TrafficLightState.GREEN and light.state != TrafficLightState.LEFT_GREEN:
                    traffic_light_detected = True
                    if dist < 80:
                        self.target_speed = 0

        # 决策逻辑
        if emergency_brake:
            # 紧急刹车
            self.target_speed = 0
            self.avoidance_maneuver = "emergency_brake"
        elif avoidance_needed and not self.avoidance_maneuver:
            # 执行避障动作
            self.perform_avoidance_maneuver()
        elif traffic_light_detected:
            # 减速停车
            self.target_speed = 0
        else:
            # 正常行驶
            self.target_speed = 50  # 目标速度50像素/秒

            # 重置避障状态
            self.avoidance_maneuver = None

    def perform_avoidance_maneuver(self):
        """执行避障动作"""
        # 简单避障策略：变道或减速
        if self.current_road and self.current_road.lanes > 1:
            # 变道避障
            self.current_lane = (self.current_lane + 1) % self.current_road.lanes
            self.avoidance_maneuver = "lane_change"
        else:
            # 减速避障
            self.target_speed *= 0.7  # 减速30%
            self.avoidance_maneuver = "slow_down"

    def draw(self, screen):
        # 绘制车身
        car_rect = pygame.Rect(0, 0, self.size.x, self.size.y)
        car_rect.center = self.position.to_tuple()

        # 创建旋转后的车辆表面
        car_surface = pygame.Surface((self.size.x, self.size.y), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, self.color, (0, 0, self.size.x, self.size.y))

        # 绘制车窗
        pygame.draw.rect(car_surface, CYAN,
                         (self.size.x * 0.6, self.size.y * 0.2, self.size.x * 0.3, self.size.y * 0.6))

        # 绘制车轮
        pygame.draw.rect(car_surface, BLACK, (0, 0, self.size.x * 0.2, self.size.y))
        pygame.draw.rect(car_surface, BLACK, (self.size.x * 0.8, 0, self.size.x, self.size.y))

        # 旋转车辆表面
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=self.position.to_tuple())

        # 绘制到屏幕
        screen.blit(rotated_surface, rotated_rect)

        # 绘制传感器
        for sensor in self.sensors:
            sensor.draw(screen)

        # 绘制跟踪对象的预测轨迹
        for obj in self.tracked_objects:
            if obj.prediction_horizon:
                # 绘制预测轨迹
                for i in range(len(obj.prediction_horizon) - 1):
                    start_pos = obj.prediction_horizon[i].to_tuple()
                    end_pos = obj.prediction_horizon[i + 1].to_tuple()
                    pygame.draw.line(screen, PURPLE, start_pos, end_pos, 2)

                # 绘制预测终点
                pygame.draw.circle(screen, PURPLE, obj.prediction_horizon[-1].to_tuple(), 5)

        # 绘制碰撞风险
        for risk in self.collision_risks:
            if risk['risk_level'] == "高危":
                color = RED
            elif risk['risk_level'] == "中危":
                color = ORANGE
            elif risk['risk_level'] == "低危":
                color = YELLOW
            else:
                continue

            # 找到对应的跟踪对象
            for obj in self.tracked_objects:
                if obj.id == risk['object_id']:
                    # 绘制风险区域
                    pygame.draw.circle(screen, color, obj.position.to_tuple(), 20, 2)

                    # 绘制风险文本
                    risk_text = f"{risk['risk_level']}: {risk['min_distance']:.1f}m"
                    text_surface = font.render(risk_text, True, color)
                    screen.blit(text_surface, (obj.position.x + 15, obj.position.y - 15))
                    break

        # 绘制路径
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                pygame.draw.line(screen, MAGENTA, self.path[i].to_tuple(), self.path[i + 1].to_tuple(), 2)
            for i, point in enumerate(self.path):
                color = GREEN if i == self.current_waypoint else MAGENTA
                pygame.draw.circle(screen, color, point.to_tuple(), 5)

        # 绘制PID调试信息
        if hasattr(self, 'pid_debug_info'):
            pid_text = f"PID: S({self.pid_debug_info['speed']['error']:.1f}) ST({self.pid_debug_info['steering']['error']:.1f}) B({self.pid_debug_info['brake']['error']:.1f})"
            pid_surface = font.render(pid_text, True, PURPLE)
            screen.blit(pid_surface, (self.position.x + 20, self.position.y - 30))


# 仿真环境类
class Simulation:
    def __init__(self):
        self.vehicles = []
        self.traffic_lights = []
        self.obstacles = []
        self.pedestrians = []
        self.roads = []
        self.intersections = []
        self.running = True
        self.clock = pygame.time.Clock()
        self.dt = 0
        self.selected_vehicle = None
        self.edit_mode = "road"  # road, vehicle, light, obstacle, pedestrian
        self.show_sensors = True
        self.show_predictions = True
        self.show_slam = True
        self.show_bev = True
        self.show_pid = True
        self.paused = False
        self.simulation_time = 0

        # BEV系统
        self.bev_system = BEVSystem()

        # 创建道路网络
        self.create_road_network()

        # 创建一些初始车辆
        for i in range(5):
            # 随机选择一条道路和车道
            road = random.choice(self.roads)
            lane = random.randint(0, road.lanes - 1)

            vehicle_type = random.choice(list(VehicleType))
            vehicle = Vehicle(Vector2(0, 0), 0, vehicle_type)
            vehicle.assign_to_road(road, lane)

            self.vehicles.append(vehicle)

        # 创建一些交通灯
        for road in self.roads:
            if road.start.x in [300, 500, 700, 900] or road.start.y in [250, 400, 550]:
                light_pos = road.start + road.direction * 40
                self.traffic_lights.append(TrafficLight(light_pos, road))

        # 创建一些障碍物
        for i in range(3):
            # 将障碍物放在道路旁边
            road = random.choice(self.roads)
            side = random.choice([-1, 1])
            pos = road.get_lane_center(0, random.random()) + road.normal * (road.width / 2 + 20) * side
            size = Vector2(20, 20)
            self.obstacles.append(Obstacle(pos, size))

        for i in range(10):
            # 将行人放在人行道上
            road = random.choice(self.roads)
            side = random.choice([-1, 1])
            pos = road.get_lane_center(0, random.random()) + road.normal * (road.width / 2 + 30) * side
            self.pedestrians.append(Pedestrian(pos))


    def create_road_network(self):
        # 创建水平道路
        horizontal_roads = [
            Road(Vector2(0, 250), Vector2(SCREEN_WIDTH, 250), 80, 2),
            Road(Vector2(0, 400), Vector2(SCREEN_WIDTH, 400), 80, 2),
            Road(Vector2(0, 550), Vector2(SCREEN_WIDTH, 550), 80, 2),
        ]

        # 创建垂直道路
        vertical_roads = [
            Road(Vector2(300, 0), Vector2(300, SCREEN_HEIGHT), 80, 2),
            Road(Vector2(500, 0), Vector2(500, SCREEN_HEIGHT), 80, 2),
            Road(Vector2(700, 0), Vector2(700, SCREEN_HEIGHT), 80, 2),
            Road(Vector2(900, 0), Vector2(900, SCREEN_HEIGHT), 80, 2),
        ]

        self.roads = horizontal_roads + vertical_roads

        # 创建交叉口
        for h_road in horizontal_roads:
            for v_road in vertical_roads:
                intersection_pos = Vector2(v_road.start.x, h_road.start.y)
                intersection = Intersection(intersection_pos)
                intersection.add_road(h_road)
                intersection.add_road(v_road)
                self.intersections.append(intersection)


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_p:
                    self.show_predictions = not self.show_predictions
                elif event.key == pygame.K_m:
                    self.show_slam = not self.show_slam
                elif event.key == pygame.K_b:
                    self.show_bev = not self.show_bev
                elif event.key == pygame.K_1:
                    self.edit_mode = "road"
                elif event.key == pygame.K_2:
                    self.edit_mode = "vehicle"
                elif event.key == pygame.K_3:
                    self.edit_mode = "light"
                elif event.key == pygame.K_4:
                    self.edit_mode = "obstacle"
                elif event.key == pygame.K_5:
                    self.edit_mode = "pedestrian"
                elif event.key == pygame.K_l:
                    for light in self.traffic_lights:
                        light.set_left_turn()
                elif event.key == pygame.K_v:
                    # 切换BEV视图模式
                    if self.bev_system.view_mode == BEVViewMode.NORMAL:
                        self.bev_system.view_mode = BEVViewMode.OCCUPANCY_GRID
                    elif self.bev_system.view_mode == BEVViewMode.OCCUPANCY_GRID:
                        self.bev_system.view_mode = BEVViewMode.ELEVATION
                    else:
                        self.bev_system.view_mode = BEVViewMode.NORMAL

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = Vector2(*pygame.mouse.get_pos())

                if event.button == 1:  # 左键
                    if self.edit_mode == "vehicle":
                        # 选择或创建车辆
                        self.selected_vehicle = None
                        for vehicle in self.vehicles:
                            if (vehicle.position - mouse_pos).length() < 20:
                                self.selected_vehicle = vehicle
                                break

                        if not self.selected_vehicle:
                            # 找到最近的道路
                            closest_road = None
                            min_dist = float('inf')
                            for road in self.roads:
                                # 计算点到道路的距离
                                road_vec = road.end - road.start
                                road_len = road_vec.length()
                                road_unit = road_vec / road_len

                                # 计算投影长度
                                t = max(0, min(road_len, (mouse_pos - road.start).dot(road_unit)))

                                # 计算投影点
                                projection = road.start + road_unit * t

                                # 计算距离
                                dist = (mouse_pos - projection).length()

                                if dist < min_dist:
                                    min_dist = dist
                                    closest_road = road

                            if closest_road:
                                # 确定车道
                                road_normal = closest_road.normal
                                side_dist = (mouse_pos - closest_road.start).dot(road_normal)
                                lane_width = closest_road.width / closest_road.lanes
                                lane = int(abs(side_dist) / lane_width)
                                lane = min(lane, closest_road.lanes - 1)

                                # 创建车辆
                                vehicle_type = random.choice(list(VehicleType))
                                new_vehicle = Vehicle(mouse_pos, 0, vehicle_type)
                                new_vehicle.assign_to_road(closest_road, lane)
                                self.vehicles.append(new_vehicle)
                                self.selected_vehicle = new_vehicle

                    elif self.edit_mode == "light":
                        # 找到最近的道路
                        closest_road = None
                        min_dist = float('inf')
                        for road in self.roads:
                            dist = mouse_pos.distance_to(road.start)
                            if dist < min_dist:
                                min_dist = dist
                                closest_road = road

                        if closest_road:
                            # 创建交通灯
                            light_pos = closest_road.start + closest_road.direction * 40
                            new_light = TrafficLight(light_pos, closest_road)
                            self.traffic_lights.append(new_light)

                    elif self.edit_mode == "obstacle":
                        # 创建障碍物
                        new_obstacle = Obstacle(mouse_pos, Vector2(20, 20))
                        self.obstacles.append(new_obstacle)

                    elif self.edit_mode == "pedestrian":
                        # 创建行人
                        new_pedestrian = Pedestrian(mouse_pos)
                        self.pedestrians.append(new_pedestrian)

                elif event.button == 3:  # 右键
                    if self.selected_vehicle:
                        # 为选中的车辆添加路径点
                        self.selected_vehicle.path.append(mouse_pos)


    def update(self):
        if self.paused:
            return

        # 更新仿真时间
        self.simulation_time += self.dt

        # 更新交通灯
        for light in self.traffic_lights:
            light.update(self.dt)

        # 更新行人
        for pedestrian in self.pedestrians:
            pedestrian.update(self.dt, self.traffic_lights)

        # 更新车辆
        all_objects = self.vehicles + self.obstacles + self.pedestrians
        for vehicle in self.vehicles:
            vehicle.update(self.dt, all_objects, self.traffic_lights, self.roads)

        # 更新BEV系统
        if self.selected_vehicle and self.show_bev:
            self.bev_system.update(
                self.selected_vehicle,
                all_objects,
                self.roads,
                self.obstacles,
                self.traffic_lights
            )


    def draw(self):
        # 清屏
        screen.fill(WHITE)

        # 绘制道路
        for road in self.roads:
            road.draw(screen)

        # 绘制交叉口
        for intersection in self.intersections:
            intersection.draw(screen)

        # 绘制障碍物
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        # 绘制行人
        for pedestrian in self.pedestrians:
            pedestrian.draw(screen)

        # 绘制交通灯
        for light in self.traffic_lights:
            light.draw(screen)

        # 绘制车辆
        for vehicle in self.vehicles:
            vehicle.draw(screen)

        # 绘制SLAM地图
        if self.selected_vehicle and self.show_slam:
            self.selected_vehicle.slam.draw(screen, 10, 10, 0.5)

        # 绘制BEV视图
        if self.show_bev:
            self.bev_system.render(screen, SCREEN_WIDTH - 410, 10)

        # 绘制UI
        self.draw_ui()

        # 更新显示
        pygame.display.flip()


    def draw_ui(self):
        # 绘制模式指示器
        global info_text
        mode_text = f"模式: {self.edit_mode} (1-5切换)"
        mode_surface = font.render(mode_text, True, BLACK)
        screen.blit(mode_surface, (10, 10))

        # 绘制控制提示
        controls_text = "空格: 暂停/继续 | S: 传感器显示 | P: 预测显示 | M: SLAM显示 | B: BEV显示 | D: PID显示 | V: BEV模式 | R: 重置PID | L: 左转信号 | 右键: 添加路径点"
        controls_surface = font.render(controls_text, True, BLACK)
        screen.blit(controls_surface, (10, 40))

        # 绘制状态信息
        status_text = f"车辆数: {len(self.vehicles)} | 行人: {len(self.pedestrians)} | 障碍物: {len(self.obstacles)} | 交通灯: {len(self.traffic_lights)}"
        status_surface = font.render(status_text, True, BLACK)
        screen.blit(status_surface, (10, 70))

        # 绘制选中的车辆信息
        info_text = ""
        if self.selected_vehicle:
            info_text = f"选中车辆 | 速度: {self.selected_vehicle.dynamics.velocity:.1f} | 角度: {self.selected_vehicle.angle:.1f}"
            info_surface = font.render(info_text, True, BLUE)
            screen.blit(info_surface, (SCREEN_WIDTH - 250, 10))

            # 显示跟踪对象数量
            tracking_text = f"跟踪对象: {len(self.selected_vehicle.tracked_objects)}"
            tracking_surface = font.render(tracking_text, True, BLUE)
            screen.blit(tracking_surface, (SCREEN_WIDTH - 250, 40))

            # 显示碰撞风险
            if self.selected_vehicle.collision_risks:
                risk_text = f"碰撞风险: {len(self.selected_vehicle.collision_risks)}"
                risk_surface = font.render(risk_text, True, RED)
                screen.blit(risk_surface, (SCREEN_WIDTH - 250, 70))

            # 显示SLAM状态
            slam_text = f"SLAM状态: {self.selected_vehicle.slam.state.name}"
            slam_surface = font.render(slam_text, True, PURPLE)
            screen.blit(slam_surface, (SCREEN_WIDTH - 250, 100))

            # 显示PID信息
            if self.show_pid:
                pid_info = self.selected_vehicle.pid_debug_info
                speed_text = f"速度PID: P={pid_info['speed']['p_term']:.1f} I={pid_info['speed']['i_term']:.1f} D={pid_info['speed']['d_term']:.1f}"
                steering_text = f"转向PID: P={pid_info['steering']['p_term']:.1f} I={pid_info['steering']['i_term']:.1f} D={pid_info['steering']['d_term']:.1f}"
                brake_text = f"刹车PID: P={pid_info['brake']['p_term']:.1f} I={pid_info['brake']['i_term']:.1f} D={pid_info['brake']['d_term']:.1f}"

                speed_surface = font.render(speed_text, True, GREEN)
                steering_surface = font.render(steering_text, True, BLUE)
                brake_surface = font.render(brake_text, True, RED)

                screen.blit(speed_surface, (SCREEN_WIDTH - 250, 130))
                screen.blit(steering_surface, (SCREEN_WIDTH - 250, 150))
                screen.blit(brake_surface, (SCREEN_WIDTH - 250, 170))

        # 绘制暂停指示
        if self.paused:
            pause_surface = large_font.render("已暂停", True, RED)
            screen.blit(pause_surface, (SCREEN_WIDTH // 2 - 50, 10))

        # 绘制预测显示状态
        prediction_text = f"预测显示: {'开启' if self.show_predictions else '关闭'}"
        prediction_surface = font.render(prediction_text, True, PURPLE)
        screen.blit(prediction_surface, (10, 100))

        # 绘制传感器显示状态
        sensor_text = f"传感器显示: {'开启' if self.show_sensors else '关闭'}"
        sensor_surface = font.render(sensor_text, True, CYAN)
        screen.blit(sensor_surface, (10, 130))

        # 绘制SLAM显示状态
        slam_text = f"SLAM显示: {'开启' if self.show_slam else '关闭'}"
        slam_surface = font.render(slam_text, True, MAGENTA)
        screen.blit(slam_surface, (10, 160))

        # 绘制BEV显示状态
        # bev_text = f"BEV显示: {'开启' if self.show_bev else '关闭'}"
        # bev_surface = font.render(bev_text, True, GREEN)
        # screen.blit(bev_surface, (10, 190))

        # 绘制PID显示状态
        pid_text = f"PID: {'开启' if self.show_pid else '关闭'}"
        pid_surface = font.render(pid_text, True, ORANGE)
        screen.blit(pid_surface, (10, 190))


    def run(self):
        while self.running:
            self.dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()


# 运行仿真
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
