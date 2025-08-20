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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 初始化pygame
pygame.init()
pygame.font.init()

# 屏幕设置
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("自动驾驶仿真系统")

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


class PredictionType(Enum):
    CONSTANT_VELOCITY = 0
    CONSTANT_ACCELERATION = 1
    MANEUVERING = 2


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

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other):
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))


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
            tracked_obj.acceleration,
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
                if distance < safety_distance:
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


# 交通灯类
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
            # 随机改变方向或设置新目标
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
        ego_velocity = Vector2(self.dynamics.velocity, 0).rotate(self.angle)
        self.collision_risks = self.sensor_fusion.get_collision_risk(
            self.position, ego_velocity
        )

        # 根据传感器数据和预测结果做出决策
        self.make_decision(objects, traffic_lights, roads)

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

            # 如果在道路上行驶，保持车道
            if self.current_road:
                self.steering = 0
            else:
                # 自由行驶模式下的路径跟踪
                if self.path and self.current_waypoint < len(self.path):
                    target_pos = self.path[self.current_waypoint]
                    to_target = target_pos - self.position
                    dist_to_target = to_target.length()

                    if dist_to_target < 20:
                        self.current_waypoint += 1
                        if self.current_waypoint >= len(self.path):
                            self.current_waypoint = 0

                    # 计算转向角度
                    target_angle = math.degrees(math.atan2(to_target.y, to_target.x))
                    angle_diff = (target_angle - self.angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360

                    self.steering = max(-1, min(1, angle_diff / 30))

        # 控制油门和刹车
        speed_error = self.target_speed - self.dynamics.velocity
        if speed_error > 0:
            self.throttle = min(1.0, speed_error / 10)
            self.brake = 0
        else:
            self.throttle = 0
            self.brake = min(1.0, -speed_error / 10)

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
        pygame.draw.rect(car_surface, BLACK, (self.size.x * 0.8, 0, self.size.x * 0.2, self.size.y))

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
        self.paused = False
        self.simulation_time = 0

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

        # 创建一些行人
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

        # 绘制UI
        self.draw_ui()

        # 更新显示
        pygame.display.flip()

    def draw_ui(self):
        # 绘制模式指示器
        mode_text = f"模式: {self.edit_mode} (1-5切换)"
        mode_surface = font.render(mode_text, True, BLACK)
        screen.blit(mode_surface, (10, 10))

        # 绘制控制提示
        controls_text = "空格: 暂停/继续 | S: 切换传感器显示 | P: 切换预测显示 | L: 左转信号 | 右键: 添加路径点"
        controls_surface = font.render(controls_text, True, BLACK)
        screen.blit(controls_surface, (10, 40))

        # 绘制状态信息
        status_text = f"车辆数: {len(self.vehicles)} | 行人: {len(self.pedestrians)} | 障碍物: {len(self.obstacles)} | 交通灯: {len(self.traffic_lights)}"
        status_surface = font.render(status_text, True, BLACK)
        screen.blit(status_surface, (10, 70))

        # 绘制选中的车辆信息
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

    def run(self):
        while self.running:
            self.dt = self.clock.tick(60) / 1000.0  # 转换为秒

            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()


# 运行仿真
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
