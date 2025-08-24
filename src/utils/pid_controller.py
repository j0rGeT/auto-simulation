import time
from ..core.enums import PIDControllerType

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=10.0, output_limit=1.0,
                 controller_type=PIDControllerType.SPEED):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.integral_limit = integral_limit  # 积分限幅
        self.output_limit = output_limit  # 输出限幅
        self.controller_type = controller_type  # 控制器类型

        # 状态变量
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()
        self.output = 0.0

        # 调试信息
        self.debug_info = {
            "error": 0.0,
            "p_term": 0.0,
            "i_term": 0.0,
            "d_term": 0.0
        }

    def reset(self):
        """重置控制器状态"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()
        self.output = 0.0

        # 重置调试信息
        self.debug_info = {
            "error": 0.0,
            "p_term": 0.0,
            "i_term": 0.0,
            "d_term": 0.0
        }

    def update(self, setpoint, process_variable, dt=None):
        """更新PID控制器

        Args:
            setpoint: 设定值
            process_variable: 过程变量（当前值）
            dt: 时间步长（秒），如果为None则自动计算

        Returns:
            output: 控制器输出
        """
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.01  # 默认时间步长
        self.previous_time = current_time

        # 计算误差
        error = setpoint - process_variable

        # 比例项
        p_term = self.kp * error

        # 积分项（带限幅）
        self.integral += error * dt
        # 积分抗饱和
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral

        # 微分项
        d_error = (error - self.previous_error) / dt if dt > 0 else 0
        d_term = self.kd * d_error

        # 保存当前误差用于下一次计算
        self.previous_error = error

        # 计算输出
        self.output = p_term + i_term + d_term

        # 输出限幅
        if self.output > self.output_limit:
            self.output = self.output_limit
        elif self.output < -self.output_limit:
            self.output = -self.output_limit

        # 保存调试信息
        self.debug_info = {
            "error": error,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term
        }

        return self.output

    def get_debug_info(self):
        """获取调试信息"""
        return self.debug_info