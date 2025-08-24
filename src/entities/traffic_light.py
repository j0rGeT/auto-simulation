import pygame
from ..core.enums import TrafficLightState
from ..core.constants import *
from ..algorithms.interfaces import TrafficLightAlgorithm
from ..algorithms.traffic_light_algorithms import FixedCycleTrafficLightAlgorithm

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
        
        # 交通灯控制算法
        self.traffic_light_algorithm = FixedCycleTrafficLightAlgorithm()

    def set_traffic_light_algorithm(self, algorithm):
        """设置交通灯控制算法"""
        if isinstance(algorithm, TrafficLightAlgorithm):
            self.traffic_light_algorithm = algorithm

    def update(self, dt, vehicles=None):
        """更新交通灯状态"""
        if vehicles is None:
            vehicles = []
        self.traffic_light_algorithm.update_traffic_light(self, dt, vehicles)

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