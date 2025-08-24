import numpy as np
import math
import random
from scipy.spatial import KDTree
from .pose import Pose
from ..utils.vectors import Vector2, Vector3
from ..core.enums import SLAMState
import pygame
from ..core.constants import *

class SLAMSystem:
    def __init__(self, map_size=(800, 800), grid_size=0.5):
        self.map_size = map_size  # 地图尺寸 (宽度, 高度)
        self.grid_size = grid_size  # 网格大小（米）

        # 初始化地图
        self.occupancy_grid = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
        self.elevation_map = np.zeros((map_size[1], map_size[0]), dtype=np.float32)

        # 初始化位姿
        self.current_pose = Pose(Vector3(map_size[0] // 2, map_size[1] // 2, 0), Vector3(0, 0, 0))
        self.estimated_pose = self.current_pose

        # 地标和特征点
        self.landmarks = []  # 存储地标位置
        self.landmark_descriptors = []  # 存储地标特征描述符
        self.landmark_covariances = []  # 存储地标协方差

        # 粒子滤波器用于定位
        self.num_particles = 100
        self.particles = []  # 粒子列表，每个粒子是一个位姿
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

        # 状态
        self.state = SLAMState.INITIALIZING
        self.initialization_counter = 0

        # 建图参数
        self.map_update_rate = 0.1  # 地图更新率
        self.observation_range = 50  # 观测范围（像素）

        # 创建KD树用于快速特征匹配
        self.landmark_tree = None
        self.update_landmark_tree()

    def update_landmark_tree(self):
        """更新地标KD树"""
        if self.landmarks:
            points = np.array([(lm.x, lm.y) for lm in self.landmarks])
            self.landmark_tree = KDTree(points)
        else:
            self.landmark_tree = None

    def predict(self, linear_velocity, angular_velocity, dt):
        """预测步骤：根据运动模型更新位姿估计"""
        # 更新当前位姿（真实位姿，仅用于仿真）
        self.current_pose.position.x += linear_velocity * math.cos(self.current_pose.orientation.z) * dt
        self.current_pose.position.y += linear_velocity * math.sin(self.current_pose.orientation.z) * dt
        self.current_pose.orientation.z += angular_velocity * dt

        # 更新估计位姿（带有噪声）
        noise_scale = 0.1
        estimated_linear_velocity = linear_velocity + random.gauss(0, noise_scale)
        estimated_angular_velocity = angular_velocity + random.gauss(0, noise_scale * 0.1)

        self.estimated_pose.position.x += estimated_linear_velocity * math.cos(self.estimated_pose.orientation.z) * dt
        self.estimated_pose.position.y += estimated_linear_velocity * math.sin(self.estimated_pose.orientation.z) * dt
        self.estimated_pose.orientation.z += estimated_angular_velocity * dt

        # 更新粒子滤波器
        self.update_particles(linear_velocity, angular_velocity, dt)

    def update_particles(self, linear_velocity, angular_velocity, dt):
        """更新粒子滤波器"""
        for i in range(len(self.particles)):
            # 添加运动噪声
            noisy_linear = linear_velocity + random.gauss(0, 0.1)
            noisy_angular = angular_velocity + random.gauss(0, 0.01)
            #print(self.particles)
            # 更新粒子位姿
            self.particles[i].position.x += noisy_linear * math.cos(self.particles[i].orientation.z) * dt
            self.particles[i].position.y += noisy_linear * math.sin(self.particles[i].orientation.z) * dt
            self.particles[i].orientation.z += noisy_angular * dt

    def update(self, observations):
        """更新步骤：根据观测更新地图和位姿估计"""
        if self.state == SLAMState.INITIALIZING:
            self.initialization_counter += 1
            if self.initialization_counter > 10:  # 初始化完成
                self.state = SLAMState.TRACKING
                # 初始化粒子滤波器
                self.initialize_particles()
            return

        # 提取观测中的特征点
        observed_features = self.extract_features(observations)

        # 数据关联：将观测特征与地图中的地标匹配
        matches = self.data_association(observed_features)

        # 更新地标
        self.update_landmarks(observed_features, matches)

        # 更新位姿估计
        self.update_pose_estimation(observed_features, matches)

        # 更新地图
        self.update_map(observations)

        # 重采样粒子
        self.resample_particles()

    def initialize_particles(self):
        """初始化粒子滤波器"""
        self.particles = []
        for _ in range(self.num_particles):
            # 在初始位姿周围随机分布粒子
            noise_pos = random.gauss(0, 10)  # 位置噪声
            noise_angle = random.gauss(0, 0.1)  # 角度噪声

            particle = Pose(
                Vector3(
                    self.estimated_pose.position.x + noise_pos,
                    self.estimated_pose.position.y + noise_pos,
                    self.estimated_pose.position.z
                ),
                Vector3(
                    self.estimated_pose.orientation.x,
                    self.estimated_pose.orientation.y,
                    self.estimated_pose.orientation.z + noise_angle
                )
            )
            self.particles.append(particle)

    def extract_features(self, observations):
        """从观测中提取特征点（简化版）"""
        features = []
        for obs in observations:
            # 简化特征提取：直接将观测位置作为特征
            features.append({
                'position': obs.position,
                'descriptor': np.random.rand(10)  # 随机特征描述符
            })
        return features

    def data_association(self, observed_features):
        """数据关联：将观测特征与地图中的地标匹配"""
        matches = []

        if not self.landmark_tree or not observed_features:
            return matches

        # 获取观测特征的位置
        observed_positions = np.array([(f['position'].x, f['position'].y) for f in observed_features])

        # 使用KD树查找最近邻
        distances, indices = self.landmark_tree.query(observed_positions, k=1)

        # 创建匹配对
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < 20.0:  # 匹配阈值
                matches.append({
                    'observed_idx': i,
                    'landmark_idx': idx,
                    'distance': dist
                })

        return matches

    def update_landmarks(self, observed_features, matches):
        """更新地标"""
        # 更新已匹配的地标
        for match in matches:
            obs_idx = match['observed_idx']
            lm_idx = match['landmark_idx']

            # 简化更新：移动地标位置向观测位置靠近
            alpha = 0.1  # 学习率
            observed_pos = observed_features[obs_idx]['position']
            self.landmarks[lm_idx] = self.landmarks[lm_idx] * (1 - alpha) + observed_pos * alpha

        # 添加新的地标
        matched_obs_indices = [m['observed_idx'] for m in matches]
        for i, feature in enumerate(observed_features):
            if i not in matched_obs_indices:
                self.landmarks.append(feature['position'])
                self.landmark_descriptors.append(feature['descriptor'])
                self.landmark_covariances.append(np.eye(2) * 10.0)  # 初始协方差

        # 更新KD树
        self.update_landmark_tree()

    def update_pose_estimation(self, observed_features, matches):
        """更新位姿估计"""
        if not matches:
            return

        # 计算位姿更新（简化版）
        # 在实际SLAM中，这里会使用扩展卡尔曼滤波器或图优化

        # 更新粒子权重
        for i in range(self.num_particles):
            weight = 1.0
            for match in matches:
                obs_idx = match['observed_idx']
                lm_idx = match['landmark_idx']

                # 计算观测预期位置（基于粒子位姿）
                expected_obs = self.landmarks[lm_idx] - Vector2(self.particles[i].position.x,
                                                                self.particles[i].position.y)
                expected_obs = expected_obs.rotate(-math.degrees(self.particles[i].orientation.z))

                # 计算实际观测位置
                actual_obs = observed_features[obs_idx]['position']

                # 计算误差
                error = expected_obs.distance_to(actual_obs)

                # 更新权重（误差越小权重越大）
                weight *= math.exp(-error * error / (2 * 10.0))  # 高斯分布

            self.particle_weights[i] = weight

        # 归一化权重
        self.particle_weights /= np.sum(self.particle_weights)

        # 选择最佳粒子作为位姿估计
        best_particle_idx = np.argmax(self.particle_weights)
        self.estimated_pose = self.particles[best_particle_idx]

    def resample_particles(self):
        """重采样粒子"""
        # 计算有效粒子数
        effective_particles = 1.0 / np.sum(self.particle_weights ** 2)

        # 如果有效粒子数太少，进行重采样
        if effective_particles < self.num_particles / 2:
            # 系统重采样
            indices = np.random.choice(
                range(self.num_particles),
                size=self.num_particles,
                p=self.particle_weights
            )

            new_particles = []
            for idx in indices:
                # 添加少量噪声以避免粒子退化
                noise_pos = random.gauss(0, 0.1)
                noise_angle = random.gauss(0, 0.01)

                new_particle = Pose(
                    Vector3(
                        self.particles[idx].position.x + noise_pos,
                        self.particles[idx].position.y + noise_pos,
                        self.particles[idx].position.z
                    ),
                    Vector3(
                        self.particles[idx].orientation.x,
                        self.particles[idx].orientation.y,
                        self.particles[idx].orientation.z + noise_angle
                    )
                )
                new_particles.append(new_particle)

            self.particles = new_particles
            self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def update_map(self, observations):
        """更新占据栅格地图"""
        # 获取当前位姿
        pose_x = int(self.estimated_pose.position.x)
        pose_y = int(self.estimated_pose.position.y)
        pose_angle = self.estimated_pose.orientation.z

        # 更新观测范围内的网格
        for obs in observations:
            # 转换观测到全局坐标系
            global_obs = Vector2(
                pose_x + obs.position.x * math.cos(pose_angle) - obs.position.y * math.sin(pose_angle),
                pose_y + obs.position.x * math.sin(pose_angle) + obs.position.y * math.cos(pose_angle)
            )

            # 确保坐标在地图范围内
            map_x = int(global_obs.x)
            map_y = int(global_obs.y)

            if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                # 更新占据概率（log odds形式）
                self.occupancy_grid[map_y, map_x] += math.log(9)  # 占据概率增加

        # 更新车辆当前位置周围的空闲区域
        for angle in np.linspace(0, 2 * math.pi, 36):
            for r in range(1, self.observation_range):
                map_x = int(pose_x + r * math.cos(pose_angle + angle))
                map_y = int(pose_y + r * math.sin(pose_angle + angle))

                if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                    # 如果这个位置没有被占据，减少占据概率
                    if self.occupancy_grid[map_y, map_x] < math.log(9):
                        self.occupancy_grid[map_y, map_x] -= math.log(9) * 0.1  # 空闲概率增加

                # 如果遇到占据网格，停止更新（假设光线被阻挡）
                if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                    if self.occupancy_grid[map_y, map_x] > 0:
                        break

    def get_occupancy_probability(self, x, y):
        """获取指定位置的占据概率"""
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            odds = math.exp(self.occupancy_grid[y, x])
            return odds / (1 + odds)
        return 0.5  # 未知区域

    def draw(self, screen, offset_x=0, offset_y=0, scale=1.0):
        """绘制SLAM地图"""
        # 绘制占据栅格地图
        for y in range(0, self.map_size[1], 5):
            for x in range(0, self.map_size[0], 5):
                prob = self.get_occupancy_probability(x, y)
                color_value = int(255 * (1 - prob))  # 占据越大概率颜色越深
                color = (color_value, color_value, color_value)

                rect = pygame.Rect(
                    offset_x + x * scale,
                    offset_y + y * scale,
                    5 * scale,
                    5 * scale
                )
                pygame.draw.rect(screen, color, rect)

        # 绘制地标
        for landmark in self.landmarks:
            pygame.draw.circle(
                screen,
                GREEN,
                (offset_x + int(landmark.x * scale), offset_y + int(landmark.y * scale)),
                3
            )

        # 绘制估计位姿
        pygame.draw.circle(
            screen,
            BLUE,
            (offset_x + int(self.estimated_pose.position.x * scale),
             offset_y + int(self.estimated_pose.position.y * scale)),
            5
        )

        # 绘制方向指示
        end_x = offset_x + int(
            self.estimated_pose.position.x * scale + 15 * math.cos(self.estimated_pose.orientation.z) * scale)
        end_y = offset_y + int(
            self.estimated_pose.position.y * scale + 15 * math.sin(self.estimated_pose.orientation.z) * scale)
        pygame.draw.line(
            screen,
            BLUE,
            (offset_x + int(self.estimated_pose.position.x * scale),
             offset_y + int(self.estimated_pose.position.y * scale)),
            (end_x, end_y),
            2
        )

        # 绘制粒子
        for particle in self.particles:
            pygame.draw.circle(
                screen,
                (255, 100, 100, 100),  # 半透明红色
                (offset_x + int(particle.position.x * scale), offset_y + int(particle.position.y * scale)),
                2
            )