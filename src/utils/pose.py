import math
import numpy as np
from .vectors import Vector3

class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position if position else Vector3()
        self.orientation = orientation if orientation else Vector3()  # 欧拉角（roll, pitch, yaw）

    def __repr__(self):
        return f"Pose(pos={self.position}, orient={self.orientation})"

    def to_matrix(self):
        """将位姿转换为4x4齐次变换矩阵"""
        # 计算旋转矩阵（使用欧拉角）
        roll, pitch, yaw = self.orientation.x, self.orientation.y, self.orientation.z

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # ZYX旋转顺序
        rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        # 构建齐次变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [self.position.x, self.position.y, self.position.z]

        return transform

    def from_matrix(self, matrix):
        """从4x4齐次变换矩阵恢复位姿"""
        self.position = Vector3(matrix[0, 3], matrix[1, 3], matrix[2, 3])

        # 从旋转矩阵提取欧拉角（ZYX顺序）
        sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(matrix[2, 1], matrix[2, 2])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            roll = math.atan2(-matrix[1, 2], matrix[1, 1])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = 0

        self.orientation = Vector3(roll, pitch, yaw)

        return self