import pygame
from ..utils.vectors import Vector2
from ..core.constants import *

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