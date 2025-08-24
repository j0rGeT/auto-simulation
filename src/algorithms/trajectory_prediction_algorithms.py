# 行人轨迹预测算法实现
import numpy as np
from .interfaces import TrajectoryPredictionAlgorithm
from ..utils.vectors import Vector2

class BasicTrajectoryPredictionAlgorithm(TrajectoryPredictionAlgorithm):
    def predict_trajectory(self, pedestrian, obstacles, traffic_lights, dt, horizon=5):
        """基础行人轨迹预测算法 - 线性外推"""
        predicted_trajectory = []
        current_pos = pedestrian.position
        current_vel = getattr(pedestrian, 'velocity', Vector2(0, 0))
        
        for step in range(horizon):
            # 简单线性外推
            predicted_pos = current_pos + current_vel * dt * (step + 1)
            
            # 简单的边界检查
            predicted_pos = self._apply_boundary_constraints(predicted_pos)
            
            # 简单的障碍物避让
            predicted_pos = self._avoid_obstacles(predicted_pos, obstacles, current_vel)
            
            predicted_trajectory.append(predicted_pos)
        
        return predicted_trajectory
    
    def _apply_boundary_constraints(self, position):
        """应用边界约束"""
        from ..core.constants import SCREEN_WIDTH, SCREEN_HEIGHT
        
        x = max(20, min(position.x, SCREEN_WIDTH - 20))
        y = max(20, min(position.y, SCREEN_HEIGHT - 20))
        
        return Vector2(x, y)
    
    def _avoid_obstacles(self, position, obstacles, velocity):
        """简单的障碍物避让"""
        if not obstacles:
            return position
        
        # 检查是否与障碍物碰撞
        for obstacle in obstacles:
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                obstacle_pos = obstacle.position
                obstacle_size = obstacle.size
                
                # 计算距离
                distance = (position - obstacle_pos).length()
                min_distance = max(obstacle_size.x, obstacle_size.y) + 10  # 安全距离
                
                if distance < min_distance:
                    # 简单的避让：沿着速度方向偏移
                    if velocity.length() > 0:
                        avoidance_dir = velocity.normalized().rotate(90)  # 垂直方向
                        position += avoidance_dir * (min_distance - distance)
        
        return position

class AdvancedTrajectoryPredictionAlgorithm(TrajectoryPredictionAlgorithm):
    def __init__(self):
        self.history = {}  # 存储行人历史轨迹
        self.prediction_horizon = 5
        
    def predict_trajectory(self, pedestrian, obstacles, traffic_lights, dt, horizon=5):
        """高级行人轨迹预测算法 - 基于社会力和意图"""
        pedestrian_id = id(pedestrian)
        
        # 更新历史轨迹
        self._update_history(pedestrian_id, pedestrian.position, pedestrian.velocity)
        
        # 预测轨迹
        predicted_trajectory = []
        current_pos = pedestrian.position
        current_vel = getattr(pedestrian, 'velocity', Vector2(0, 0))
        
        for step in range(horizon):
            # 计算社会力（障碍物排斥力、目标吸引力等）
            social_force = self._calculate_social_force(current_pos, current_vel, obstacles, traffic_lights)
            
            # 计算意图力（基于历史行为）
            intention_force = self._calculate_intention_force(pedestrian_id, current_pos)
            
            # 计算总力
            total_force = social_force + intention_force
            
            # 更新速度和位置
            acceleration = total_force * 0.1  # 简化假设质量=10
            current_vel += acceleration * dt
            current_vel = current_vel * 0.95  # 速度衰减
            
            current_pos += current_vel * dt
            
            # 应用约束
            current_pos = self._apply_constraints(current_pos, current_vel)
            
            predicted_trajectory.append(current_pos.copy())
        
        return predicted_trajectory
    
    def _update_history(self, pedestrian_id, position, velocity):
        """更新行人历史轨迹"""
        if pedestrian_id not in self.history:
            self.history[pedestrian_id] = {
                'positions': [],
                'velocities': [],
                'timestamps': []
            }
        
        # 保留最近10个位置
        self.history[pedestrian_id]['positions'].append(position)
        self.history[pedestrian_id]['velocities'].append(velocity)
        
        if len(self.history[pedestrian_id]['positions']) > 10:
            self.history[pedestrian_id]['positions'].pop(0)
            self.history[pedestrian_id]['velocities'].pop(0)
    
    def _calculate_social_force(self, position, velocity, obstacles, traffic_lights):
        """计算社会力"""
        social_force = Vector2(0, 0)
        
        # 障碍物排斥力
        for obstacle in obstacles:
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                obstacle_pos = obstacle.position
                obstacle_size = obstacle.size
                
                # 计算到障碍物的距离
                to_obstacle = obstacle_pos - position
                distance = to_obstacle.length()
                
                if distance > 0:
                    # 排斥力与距离的平方成反比
                    repulsion_force = -to_obstacle.normalized() * (100 / (distance ** 2 + 1))
                    social_force += repulsion_force
        
        # 交通灯影响力
        for light in traffic_lights:
            if hasattr(light, 'position') and hasattr(light, 'state'):
                light_pos = light.position
                to_light = light_pos - position
                distance = to_light.length()
                
                if distance < 100:  # 只在近距离有影响
                    if light.state.name in ['RED', 'YELLOW']:
                        # 红灯和黄灯产生排斥力
                        repulsion_force = -to_light.normalized() * (50 / (distance + 1))
                        social_force += repulsion_force
        
        return social_force
    
    def _calculate_intention_force(self, pedestrian_id, current_pos):
        """计算意图力（基于历史行为）"""
        if pedestrian_id not in self.history or len(self.history[pedestrian_id]['positions']) < 2:
            return Vector2(0, 0)
        
        positions = self.history[pedestrian_id]['positions']
        velocities = self.history[pedestrian_id]['velocities']
        
        # 计算平均移动方向
        if len(positions) >= 2:
            recent_movement = positions[-1] - positions[-2]
            if recent_movement.length() > 0:
                intention_direction = recent_movement.normalized()
                # 意图力与历史速度成正比
                avg_speed = sum([v.length() for v in velocities[-3:]]) / min(3, len(velocities))
                intention_force = intention_direction * avg_speed * 0.5
                return intention_force
        
        return Vector2(0, 0)
    
    def _apply_constraints(self, position, velocity):
        """应用各种约束"""
        from ..core.constants import SCREEN_WIDTH, SCREEN_HEIGHT
        
        # 边界约束
        x = max(20, min(position.x, SCREEN_WIDTH - 20))
        y = max(20, min(position.y, SCREEN_HEIGHT - 20))
        
        # 速度方向约束（防止突然转向）
        if velocity.length() > 0:
            new_pos = Vector2(x, y)
            # 保持一定的运动惯性
            new_pos = new_pos + velocity * 0.1
            return new_pos
        
        return Vector2(x, y)

class MLTrajectoryPredictionAlgorithm(TrajectoryPredictionAlgorithm):
    def __init__(self):
        self.trajectory_patterns = self._initialize_patterns()
        
    def predict_trajectory(self, pedestrian, obstacles, traffic_lights, dt, horizon=5):
        """机器学习轨迹预测算法（简化版）"""
        current_pos = pedestrian.position
        current_vel = getattr(pedestrian, 'velocity', Vector2(0, 0))
        
        # 识别运动模式
        movement_pattern = self._identify_movement_pattern(current_vel)
        
        # 基于模式预测
        if movement_pattern == "straight":
            return self._predict_straight_trajectory(current_pos, current_vel, dt, horizon)
        elif movement_pattern == "turning":
            return self._predict_turning_trajectory(current_pos, current_vel, dt, horizon)
        elif movement_pattern == "stopping":
            return self._predict_stopping_trajectory(current_pos, current_vel, dt, horizon)
        else:
            return self._predict_random_trajectory(current_pos, current_vel, dt, horizon)
    
    def _initialize_patterns(self):
        """初始化运动模式"""
        return {
            "straight": {"probability": 0.6, "speed_factor": 1.0},
            "turning": {"probability": 0.3, "speed_factor": 0.8},
            "stopping": {"probability": 0.1, "speed_factor": 0.5}
        }
    
    def _identify_movement_pattern(self, velocity):
        """识别运动模式"""
        speed = velocity.length()
        
        if speed < 0.5:
            return "stopping"
        
        # 简单的基于速度方向变化识别转向
        if speed > 2.0 and abs(velocity.angle_to(Vector2(1, 0))) > 30:
            return "turning"
        
        return "straight"
    
    def _predict_straight_trajectory(self, current_pos, current_vel, dt, horizon):
        """预测直线运动轨迹"""
        predicted_trajectory = []
        
        for step in range(horizon):
            predicted_pos = current_pos + current_vel * dt * (step + 1)
            predicted_trajectory.append(predicted_pos)
        
        return predicted_trajectory
    
    def _predict_turning_trajectory(self, current_pos, current_vel, dt, horizon):
        """预测转向运动轨迹"""
        predicted_trajectory = []
        
        # 模拟转向（简单的圆弧运动）
        turn_angle = 15  # 度/秒
        
        for step in range(horizon):
            # 计算转向后的速度方向
            turned_vel = current_vel.rotate(turn_angle * step * dt)
            predicted_pos = current_pos + turned_vel * dt * (step + 1)
            predicted_trajectory.append(predicted_pos)
        
        return predicted_trajectory
    
    def _predict_stopping_trajectory(self, current_pos, current_vel, dt, horizon):
        """预测停止运动轨迹"""
        predicted_trajectory = []
        
        # 模拟减速停止
        deceleration = current_vel.normalized() * -0.5  # 减速度
        
        for step in range(horizon):
            # 计算减速后的速度
            step_vel = current_vel + deceleration * dt * step
            if step_vel.length() < 0:
                step_vel = Vector2(0, 0)
            
            predicted_pos = current_pos + step_vel * dt * (step + 1)
            predicted_trajectory.append(predicted_pos)
        
        return predicted_trajectory
    
    def _predict_random_trajectory(self, current_pos, current_vel, dt, horizon):
        """预测随机运动轨迹"""
        predicted_trajectory = []
        
        # 添加一些随机性
        random_factor = Vector2(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        
        for step in range(horizon):
            perturbed_vel = current_vel + random_factor * (step + 1)
            predicted_pos = current_pos + perturbed_vel * dt * (step + 1)
            predicted_trajectory.append(predicted_pos)
        
        return predicted_trajectory