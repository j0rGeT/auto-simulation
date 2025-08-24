import math
import random
import pygame
import time
from ..core.enums import VehicleType, SensorType
from ..core.constants import *
from ..utils.vectors import Vector2
from ..entities.vehicle_dynamics import VehicleDynamics
from ..entities.sensor import Sensor
from ..utils.sensor_fusion import SensorFusion
from ..utils.slam_system import SLAMSystem
from ..utils.bev_system import BEVSystem
from ..utils.pid_controller import PIDController, PIDControllerType
from ..algorithms.interfaces import DecisionAlgorithm, ObstacleAvoidanceAlgorithm, TrafficLightRecognitionAlgorithm, ObstacleRecognitionAlgorithm, ControlAlgorithm
from ..algorithms.decision_algorithms import BasicDecisionAlgorithm
from ..algorithms.avoidance_algorithms import LaneChangeAvoidanceAlgorithm, PedestrianAvoidanceAlgorithm
from ..algorithms.control_algorithms import PIDControlAlgorithm

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

        # Sensors
        self.sensors = [
            Sensor(self, SensorType.CAMERA, Vector2(20, 0), 0, 150, 60),  # Front camera
            Sensor(self, SensorType.LIDAR, Vector2(10, 8), 30, 200, 30),  # Front-left lidar
            Sensor(self, SensorType.LIDAR, Vector2(10, -8), -30, 200, 30),  # Front-right lidar
            Sensor(self, SensorType.RADAR, Vector2(-15, 0), 180, 100, 80)  # Rear radar
        ]

        # 传感器融合
        self.sensor_fusion = SensorFusion()
        self.tracked_objects = []
        self.collision_risks = []

        # SLAM系统
        self.slam = SLAMSystem()

        # BEV系统
        self.bev = BEVSystem()

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

        # Algorithm configuration
        self.decision_algorithm = BasicDecisionAlgorithm()
        self.obstacle_avoidance_algorithm = PedestrianAvoidanceAlgorithm()
        self.control_algorithm = PIDControlAlgorithm()
        self.traffic_light_recognition_algorithm = None
        self.obstacle_recognition_algorithm = None

        # 对象类型（用于传感器检测）
        self.object_type = "vehicle"

        # PID调试信息
        self.pid_debug_info = {
            "speed": {"error": 0, "p": 0, "i": 0, "d": 0},
            "steering": {"error": 0, "p": 0, "i": 0, "d": 0},
            "brake": {"error": 0, "p": 0, "i": 0, "d": 0}
        }

    def set_decision_algorithm(self, algorithm):
        """设置决策算法"""
        if isinstance(algorithm, DecisionAlgorithm):
            self.decision_algorithm = algorithm

    def set_obstacle_avoidance_algorithm(self, algorithm):
        """设置避障算法"""
        if isinstance(algorithm, ObstacleAvoidanceAlgorithm):
            self.obstacle_avoidance_algorithm = algorithm

    def set_traffic_light_recognition_algorithm(self, algorithm):
        """设置交通灯识别算法"""
        if isinstance(algorithm, TrafficLightRecognitionAlgorithm):
            self.traffic_light_recognition_algorithm = algorithm

    def set_obstacle_recognition_algorithm(self, algorithm):
        """设置障碍物识别算法"""
        if isinstance(algorithm, ObstacleRecognitionAlgorithm):
            self.obstacle_recognition_algorithm = algorithm

    def set_control_algorithm(self, algorithm):
        """设置控制算法"""
        if isinstance(algorithm, ControlAlgorithm):
            self.control_algorithm = algorithm

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

        # Sensor fusion
        current_time = time.time()
        self.tracked_objects = self.sensor_fusion.update(all_detections, current_time)

        # Use recognition algorithms for traffic lights and obstacles
        if self.traffic_light_recognition_algorithm:
            traffic_light_detections = [d for d in all_detections if d.object_type == "traffic_light"]
            recognized_traffic_lights = self.traffic_light_recognition_algorithm.recognize_traffic_lights(
                traffic_light_detections, self.position
            )
            # Can process recognized traffic lights here

        if self.obstacle_recognition_algorithm:
            obstacle_detections = [d for d in all_detections if d.object_type == "obstacle"]
            recognized_obstacles = self.obstacle_recognition_algorithm.recognize_obstacles(
                obstacle_detections, self.position
            )
            # Can process recognized obstacles here

        # 计算碰撞风险
        ego_velocity = Vector2(self.dynamics.velocity, 0).rotate(self.angle)
        self.collision_risks = self.sensor_fusion.get_collision_risk(
            self.position, ego_velocity
        )

        # 更新SLAM系统
        linear_velocity = self.dynamics.velocity
        angular_velocity = math.radians(self.dynamics.steering_angle) * linear_velocity / self.size.x
        self.slam.predict(linear_velocity, angular_velocity, dt)
        self.slam.update(all_detections)

        # Update BEV system
        obstacles = [obj for obj in objects if hasattr(obj, 'object_type') and obj.object_type == 'obstacle']
        self.bev.update(self, objects, roads, obstacles, traffic_lights)

        # 根据传感器数据和预测结果做出决策
        self.make_decision(objects, traffic_lights, roads)

        # 使用控制算法更新控制输入
        self.update_control_inputs(dt)

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

    def update_control_inputs(self, dt):
        """使用控制算法更新控制输入"""
        # 准备目标状态
        target_state = self._prepare_target_state()
        
        # 准备障碍物信息
        obstacles = self._prepare_obstacle_info()
        
        # 使用控制算法计算控制输入
        control_output = self.control_algorithm.compute_control(self, target_state, obstacles, dt)
        
        # 应用控制输入
        self.throttle = control_output.get('throttle', 0)
        self.brake = control_output.get('brake', 0)
        self.steering = control_output.get('steering', 0)
        print(f"Vehicle: throttle={self.throttle:.2f}, brake={self.brake:.2f}, steering={self.steering:.2f}, target_speed={self.target_speed}")
    
    def _prepare_target_state(self):
        """准备目标状态"""
        target_state = {'target_speed': self.target_speed}
        
        if self.current_road:
            # When driving on road, target position is lane center
            lane_center = self.current_road.get_lane_center(self.current_lane, self.road_progress)
            road_direction = self.current_road.get_lane_direction()
            target_angle = math.degrees(math.atan2(road_direction.y, road_direction.x))
            
            target_state['target_position'] = lane_center
            target_state['target_angle'] = target_angle
        elif self.path and self.current_waypoint < len(self.path):
            # 自由行驶时，目标位置是下一个路径点
            target_state['target_position'] = self.path[self.current_waypoint]
        
        return target_state
    
    def _prepare_obstacle_info(self):
        """准备障碍物信息"""
        obstacles = []
        
        # 添加跟踪的障碍物
        for obj in self.tracked_objects:
            obstacles.append({
                'position': obj.position,
                'velocity': obj.velocity,
                'size': obj.size if hasattr(obj, 'size') else Vector2(10, 10),
                'type': obj.object_type if hasattr(obj, 'object_type') else 'unknown'
            })
        
        # 添加碰撞风险信息
        for risk in self.collision_risks:
            for obj in self.tracked_objects:
                if obj.id == risk['object_id']:
                    obstacles.append({
                        'position': obj.position,
                        'velocity': obj.velocity,
                        'size': obj.size if hasattr(obj, 'size') else Vector2(10, 10),
                        'type': obj.object_type if hasattr(obj, 'object_type') else 'unknown',
                        'risk_level': risk['risk_level']
                    })
                    break
        
        return obstacles

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
        # 使用决策算法进行决策
        print("Vehicle: Making decision...")
        self.decision_algorithm.make_decision(self, objects, traffic_lights, roads)

    def perform_avoidance_maneuver(self):
        """执行避障动作"""
        # 使用避障算法进行避障
        self.obstacle_avoidance_algorithm.perform_avoidance(self, self.collision_risks)

    def draw(self, screen):
        # 绘制BEV视图
        self.bev.render(screen, SCREEN_WIDTH - self.bev.width - 20, 20)
        
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

        # Draw collision risks
        for risk in self.collision_risks:
            if risk['risk_level'] == "high_risk":
                color = RED
            elif risk['risk_level'] == "medium_risk":
                color = ORANGE
            elif risk['risk_level'] == "low_risk":
                color = YELLOW
            else:
                continue

            # 找到对应的跟踪对象
            for obj in self.tracked_objects:
                if obj.id == risk['object_id']:
                    # 绘制风险区域
                    pygame.draw.circle(screen, color, obj.position.to_tuple(), 20, 2)
                    break

        # 绘制路径
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                pygame.draw.line(screen, MAGENTA, self.path[i].to_tuple(), self.path[i + 1].to_tuple(), 2)
            for i, point in enumerate(self.path):
                color = GREEN if i == self.current_waypoint else MAGENTA
                pygame.draw.circle(screen, color, point.to_tuple(), 5)