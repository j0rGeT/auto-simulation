from ..core.enums import PredictionType
import random
from ..utils.vectors import Vector2

class TrajectoryPredictor:
    def __init__(self, prediction_horizon=3.0, time_step=0.1):
        self.prediction_horizon = prediction_horizon  # 预测时间范围（秒）
        self.time_step = time_step  # 预测时间步长（秒）
        self.num_steps = int(prediction_horizon / time_step)

    def predict_constant_velocity(self, position, velocity, num_steps=None):
        """恒定速度模型预测"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        for i in range(1, num_steps + 1):
            t = i * self.time_step
            predicted_pos = position + velocity * t
            trajectory.append(predicted_pos)

        return trajectory

    def predict_constant_acceleration(self, position, velocity, acceleration, num_steps=None):
        """恒定加速度模型预测"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        for i in range(1, num_steps + 1):
            t = i * self.time_step
            predicted_pos = position + velocity * t + acceleration * (0.5 * t * t)
            trajectory.append(predicted_pos)

        return trajectory

    def predict_maneuvering(self, position, velocity, acceleration, maneuver_probability=0.1, num_steps=None):
        """考虑机动性的预测模型"""
        if num_steps is None:
            num_steps = self.num_steps

        trajectory = []
        current_velocity = velocity
        current_position = position

        for i in range(1, num_steps + 1):
            t = i * self.time_step

            # 随机引入机动性（转向或加减速）
            if random.random() < maneuver_probability:
                # 随机改变加速度
                acceleration = acceleration + Vector2(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                )

            # 更新速度和位置
            current_velocity = current_velocity + acceleration * self.time_step
            current_position = current_position + current_velocity * self.time_step

            trajectory.append(current_position)

        return trajectory

    def predict_trajectory(self, position, velocity, acceleration=None,
                           prediction_type=PredictionType.CONSTANT_VELOCITY):
        """根据指定预测类型预测轨迹"""
        if prediction_type == PredictionType.CONSTANT_VELOCITY:
            return self.predict_constant_velocity(position, velocity)
        elif prediction_type == PredictionType.CONSTANT_ACCELERATION and acceleration is not None:
            return self.predict_constant_acceleration(position, velocity, acceleration)
        elif prediction_type == PredictionType.MANEUVERING and acceleration is not None:
            return self.predict_maneuvering(position, velocity, acceleration)
        else:
            # 默认使用恒定速度模型
            return self.predict_constant_velocity(position, velocity)