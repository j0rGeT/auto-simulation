from enum import Enum

class TrafficLightState(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    LEFT_GREEN = 3

class VehicleType(Enum):
    SEDAN = 0
    SUV = 1
    TRUCK = 2

class SensorType(Enum):
    CAMERA = 0
    LIDAR = 1
    RADAR = 2
    GPS = 3
    IMU = 4

class PredictionType(Enum):
    CONSTANT_VELOCITY = 0
    CONSTANT_ACCELERATION = 1
    MANEUVERING = 2

class SLAMState(Enum):
    INITIALIZING = 0
    TRACKING = 1
    LOST = 2

class BEVViewMode(Enum):
    NORMAL = 0
    OCCUPANCY_GRID = 1
    ELEVATION = 2

class PIDControllerType(Enum):
    SPEED = 0
    STEERING = 1
    BRAKE = 2

class AlgorithmType(Enum):
    DECISION_MAKING = 0
    TRAFFIC_LIGHT = 1
    OBSTACLE_AVOIDANCE = 2