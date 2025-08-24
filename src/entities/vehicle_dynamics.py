class VehicleDynamics:
    def __init__(self, mass=1500, max_engine_force=3000, max_brake_force=5000, drag_coefficient=0.3):
        self.mass = mass
        self.max_engine_force = max_engine_force
        self.max_brake_force = max_brake_force
        self.drag_coefficient = drag_coefficient
        self.velocity = 0
        self.acceleration = 0
        self.engine_force = 0
        self.brake_force = 0
        self.steering_angle = 0
        self.max_steering_angle = 30

    def update(self, dt, throttle_input, brake_input, steering_input):
        # 计算引擎力
        self.engine_force = throttle_input * self.max_engine_force

        # 计算制动力
        self.brake_force = brake_input * self.max_brake_force

        # 计算转向角
        self.steering_angle = steering_input * self.max_steering_angle

        # 计算阻力
        drag_force = self.drag_coefficient * self.velocity * abs(self.velocity)

        # 计算总力
        total_force = self.engine_force - self.brake_force - drag_force

        # 计算加速度
        self.acceleration = total_force / self.mass

        # 更新速度
        self.velocity += self.acceleration * dt

        # 确保速度不为负
        if self.velocity < 0:
            self.velocity = 0