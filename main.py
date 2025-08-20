#coding=utf-8
import pygame
import math
import random
import numpy as np
import time
from enum import Enum
from collections import deque
import json
import threading

# 初始化pygame
pygame.init()
pygame.font.init()

# 屏幕设置
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("auto simulation system")

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

# 字体
font = pygame.font.SysFont('Arial', 16)
large_font = pygame.font.SysFont('Arial', 24)


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

    def update(self, dt, traffic_lights):
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
            if isinstance(obj, (Vehicle, Obstacle, Pedestrian)):
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
                        self.detected_objects.append((obj, dist))

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
        for obj, dist in self.detected_objects:
            pygame.draw.line(screen, YELLOW, sensor_pos.to_tuple(), obj.position.to_tuple(), 1)


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

    def assign_to_road(self, road, lane=0):
        """将车辆分配到指定道路的车道上"""
        self.current_road = road
        self.current_lane = lane
        self.road_progress = 0
        self.position = road.get_lane_center(lane, 0)
        self.angle = math.degrees(math.atan2(road.direction.y, road.direction.x))

    def update(self, dt, objects, traffic_lights, roads):
        # 更新传感器
        for sensor in self.sensors:
            sensor.update(objects)

        # 根据传感器数据做出决策
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
        # 简单决策逻辑：跟随道路，避开障碍物，遵守交通规则

        # 检查前方是否有障碍物
        front_sensor = self.sensors[0]
        obstacle_detected = False
        min_obstacle_dist = float('inf')

        for obj, dist in front_sensor.detected_objects:
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
            obstacle_detected = True

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
        if obstacle_detected and min_obstacle_dist < 50:
            # 紧急刹车
            self.target_speed = 0
            # 尝试变道避开
            if self.current_road and self.current_road.lanes > 1:
                self.current_lane = (self.current_lane + 1) % self.current_road.lanes
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
        self.paused = False

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
        mode_text = f"mode: {self.edit_mode} (1-5)"
        mode_surface = font.render(mode_text, True, BLACK)
        screen.blit(mode_surface, (10, 10))

        # 绘制控制提示
        controls_text = "info: "
        controls_surface = font.render(controls_text, True, BLACK)
        screen.blit(controls_surface, (10, 40))

        # 绘制状态信息
        status_text = f"cars: {len(self.vehicles)} | Pedestrian: {len(self.pedestrians)} | obstacle: {len(self.obstacles)} | trafficlight: {len(self.traffic_lights)}"
        status_surface = font.render(status_text, True, BLACK)
        screen.blit(status_surface, (10, 70))

        # 绘制选中的车辆信息
        if self.selected_vehicle:
            info_text = f"selected car | speed: {self.selected_vehicle.dynamics.velocity:.1f} | veo: {self.selected_vehicle.angle:.1f}"
            info_surface = font.render(info_text, True, BLUE)
            screen.blit(info_surface, (SCREEN_WIDTH - 250, 10))

        # 绘制暂停指示
        if self.paused:
            pause_surface = large_font.render("已暂停", True, RED)
            screen.blit(pause_surface, (SCREEN_WIDTH // 2 - 50, 10))

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
