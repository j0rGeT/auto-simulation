from dataclasses import dataclass
from ..utils.vectors import Vector2
from typing import List

@dataclass
class TrackedObject:
    id: int
    position: Vector2
    velocity: Vector2
    acceleration: Vector2
    object_type: str  # "vehicle", "pedestrian", "obstacle"
    size: Vector2
    confidence: float
    last_update: float
    prediction_horizon: List[Vector2] = None

    def __post_init__(self):
        if self.prediction_horizon is None:
            self.prediction_horizon = []

class Detection:
    def __init__(self, position, object_type, size, confidence=1.0):
        self.position = position
        self.object_type = object_type
        self.size = size
        self.confidence = confidence