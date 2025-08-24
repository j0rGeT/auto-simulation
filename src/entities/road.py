import pygame
from ..utils.vectors import Vector2
from ..core.constants import *

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