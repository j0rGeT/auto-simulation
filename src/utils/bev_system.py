import math
import pygame
from ..core.enums import BEVViewMode, TrafficLightState
from ..core.constants import *

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
            if hasattr(obj, 'object_type') and obj.object_type in ['vehicle', 'pedestrian']:
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

        # 绘制道路中心极地
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
        title_text = "BEV"
        title_surface = pygame.font.Font(None, 24).render(title_text, True, BLACK)
        screen.blit(title_surface, (x + 10, y + 10))

        # 绘制视图模式
        mode_text = f"Mode: {self.view_mode.name}"
        mode_surface = pygame.font.Font(None, 20).render(mode_text, True, BLACK)
        screen.blit(mode_surface, (x + 10, y + 30))

        # 绘制比例尺
        scale_text = f"check: 1px = {1 / self.scale:.2f}m"
        scale_surface = pygame.font.Font(None, 20).render(scale_text, True, BLACK)
        screen.blit(scale_surface, (x + 10, y + 50))