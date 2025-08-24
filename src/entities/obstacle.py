import pygame
from ..core.constants import *

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