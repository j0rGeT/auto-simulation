import pygame
import random
import time
from ..core.constants import *
from ..core.enums import VehicleType
from ..utils.vectors import Vector2
from ..entities.road import Road
from ..entities.intersection import Intersection
from ..entities.traffic_light import TrafficLight
from ..entities.obstacle import Obstacle
from ..entities.pedestrian import Pedestrian
from ..entities.vehicle import Vehicle

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
                        # 为选中的车辆添加路径极地
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
            pedestrian.update(self.dt, self.traffic_lights, self.obstacles)

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
            pedestrian.draw_predicted_trajectory(screen)

        # 绘制交通灯
        for light in self.traffic_lights:
            light.draw(screen)

        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(screen)

        # Draw UI
        self.draw_ui()

        # Update display
        pygame.display.flip()

    def draw_ui(self):
        # Draw mode indicator
        mode_text = f"Mode: {self.edit_mode} (1-5 to switch)"
        mode_surface = pygame.font.Font(None, 24).render(mode_text, True, BLACK)
        screen.blit(mode_surface, (10, 10))

        # Draw control hints
        controls_text = "Space: Pause/Resume | S: Toggle sensors | P: Toggle predictions | L: Left turn | Right click: Add waypoint"
        controls_surface = pygame.font.Font(None, 20).render(controls_text, True, BLACK)
        screen.blit(controls_surface, (10, 40))

        # Draw status information
        status_text = f"Vehicles: {len(self.vehicles)} | Pedestrians: {len(self.pedestrians)} | Obstacles: {len(self.obstacles)} | Traffic lights: {len(self.traffic_lights)}"
        status_surface = pygame.font.Font(None, 20).render(status_text, True, BLACK)
        screen.blit(status_surface, (10, 70))

        # Draw selected vehicle information
        if self.selected_vehicle:
            info_text = f"Selected vehicle | Speed: {self.selected_vehicle.dynamics.velocity:.1f} | Angle: {self.selected_vehicle.angle:.1f}"
            info_surface = pygame.font.Font(None, 20).render(info_text, True, BLUE)
            screen.blit(info_surface, (SCREEN_WIDTH - 250, 10))

            # Show tracked objects count
            tracking_text = f"Tracked objects: {len(self.selected_vehicle.tracked_objects)}"
            tracking_surface = pygame.font.Font(None, 20).render(tracking_text, True, BLUE)
            screen.blit(tracking_surface, (SCREEN_WIDTH - 250, 40))

            # Show collision risks
            if self.selected_vehicle.collision_risks:
                risk_text = f"Collision risks: {len(self.selected_vehicle.collision_risks)}"
                risk_surface = pygame.font.Font(None, 20).render(risk_text, True, RED)
                screen.blit(risk_surface, (SCREEN_WIDTH - 250, 70))

        # Draw pause indicator
        if self.paused:
            pause_surface = pygame.font.Font(None, 36).render("PAUSED", True, RED)
            screen.blit(pause_surface, (SCREEN_WIDTH // 2 - 50, 10))

        # Draw prediction display status
        prediction_text = f"predict: {'open' if self.show_predictions else 'close'}"
        prediction_surface = pygame.font.Font(None, 20).render(prediction_text, True, PURPLE)
        screen.blit(prediction_surface, (10, 100))

        # 绘制传感器显示状态
        sensor_text = f"sensor: {'open' if self.show_sensors else 'close'}"
        sensor_surface = pygame.font.Font(None, 20).render(sensor_text, True, CYAN)
        screen.blit(sensor_surface, (10, 130))

    def run(self):
        # 初始化屏幕
        global screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("auto simulation - PID")
        
        while self.running:
            self.dt = self.clock.tick(60) / 1000.0  # 转换为秒

            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()