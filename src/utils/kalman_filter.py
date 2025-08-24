import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=1.0):
        # 状态向量: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)

        # 状态转移矩阵 (恒定加速度模型)
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 测量矩阵 (只测量位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # 过程噪声协方差
        self.Q = np.eye(6) * process_noise

        # 测量噪声协方差
        self.R = np.eye(2) * measurement_noise

        # 误差协方差矩阵
        self.P = np.eye(6)

        # 时间步长
        self.dt = dt

    def predict(self):
        # 预测状态
        self.state = np.dot(self.F, self.state)

        # 预测误差协方差
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.state[:2]  # 返回预测的位置

    def update(self, measurement):
        # 计算卡尔曼增益
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态估计
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)

        # 更新误差协方差
        I = np.eye(self.state.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return self.state[:2]  # 返回更新后的位置