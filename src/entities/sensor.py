import math
import pygame
from ..utils.vectors import Vector2
from ..core.constants import *
from ..entities.datatypes import Detection

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
            if hasattr(obj, 'object_type'):
                if obj.object_type == "vehicle":
                    obj_type = "vehicle"
                    obj_size = obj.size
                    confidence = 0.9  # 车辆检测置信度较高
                elif obj.object_type == "pedestrian":
                    obj_type = "pedestrian"
                    obj_size = obj.size
                    confidence = 0.7  # 行人检测置信度中等
                elif obj.object_type == "obstacle":
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