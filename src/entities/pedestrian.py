import random
import pygame
from ..utils.vectors import Vector2
from ..core.constants import *
from ..core.enums import TrafficLightState
from ..algorithms.interfaces import TrajectoryPredictionAlgorithm
from ..algorithms.trajectory_prediction_algorithms import BasicTrajectoryPredictionAlgorithm

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
        
        # 轨迹预测算法
        self.trajectory_prediction_algorithm = BasicTrajectoryPredictionAlgorithm()
        self.predicted_trajectory = []

    def set_trajectory_prediction_algorithm(self, algorithm):
        """设置轨迹预测算法"""
        if isinstance(algorithm, TrajectoryPredictionAlgorithm):
            self.trajectory_prediction_algorithm = algorithm

    def update(self, dt, traffic_lights, obstacles=None):
        if obstacles is None:
            obstacles = []
        
        # 保存旧位置用于计算速度
        old_position = self.position

        # 预测轨迹
        self.predicted_trajectory = self.trajectory_prediction_algorithm.predict_trajectory(
            self, obstacles, traffic_lights, dt
        )

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

    def draw_predicted_trajectory(self, screen):
        """绘制预测轨迹"""
        if len(self.predicted_trajectory) > 1:
            # 绘制轨迹线
            for i in range(len(self.predicted_trajectory) - 1):
                start_pos = self.predicted_trajectory[i].to_tuple()
                end_pos = self.predicted_trajectory[i + 1].to_tuple()
                pygame.draw.line(screen, PURPLE, start_pos, end_pos, 2)
            
            # 绘制轨迹点
            for i, point in enumerate(self.predicted_trajectory):
                alpha = 255 * (i + 1) // len(self.predicted_trajectory)
                color = (128, 0, 128, alpha)  # 渐变的紫色
                pygame.draw.circle(screen, color, point.to_tuple(), 3)

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