import time
import numpy as np
from .kalman_filter import KalmanFilter
from .trajectory_predictor import TrajectoryPredictor
from ..entities.datatypes import TrackedObject
from ..core.enums import PredictionType
from ..utils.vectors import Vector2

class SensorFusion:
    def __init__(self, association_threshold=50.0, max_age=1.0):
        self.tracked_objects = {}  # 跟踪的对象字典 {id: TrackedObject}
        self.next_id = 0
        self.association_threshold = association_threshold  # 关联阈值（像素）
        self.max_age = max_age  # 最大未更新时长（秒）
        self.kalman_filters = {}  # 卡尔曼滤波器字典 {id: KalmanFilter}
        self.trajectory_predictor = TrajectoryPredictor()

    def update(self, detections, current_time):
        """更新传感器融合状态"""
        # 移除过期的跟踪对象
        self.remove_old_objects(current_time)

        # 如果没有检测到任何物体，直接返回
        if not detections:
            return list(self.tracked_objects.values())

        # 关联检测结果与现有跟踪对象
        matched_detections = set()
        matched_tracks = set()

        # 计算所有检测结果与跟踪对象之间的距离
        distance_matrix = []
        for detection in detections:
            row = []
            for obj_id, tracked_obj in self.tracked_objects.items():
                distance = detection.position.distance_to(tracked_obj.position)
                row.append(distance)
            distance_matrix.append(row)

        # 简单的最近邻关联
        for i, detection in enumerate(detections):
            if not distance_matrix[i]:
                continue

            min_distance = min(distance_matrix[i])
            if min_distance < self.association_threshold:
                min_index = distance_matrix[i].index(min_distance)
                obj_id = list(self.tracked_objects.keys())[min_index]

                # 更新跟踪对象
                self.update_tracked_object(obj_id, detection, current_time)
                matched_detections.add(i)
                matched_tracks.add(obj_id)

        # 处理未匹配的检测结果（创建新的跟踪对象）
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self.create_new_track(detection, current_time)

        # 处理未匹配的跟踪对象（预测其状态）
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_tracks:
                self.predict_tracked_object(obj_id, current_time)

        return list(self.tracked_objects.values())

    def create_new_track(self, detection, current_time):
        """创建新的跟踪对象"""
        obj_id = self.next_id
        self.next_id += 1

        # 初始化卡尔曼滤波器
        kf = KalmanFilter()
        kf.state[:2] = np.array([detection.position.x, detection.position.y])

        # 创建跟踪对象
        tracked_obj = TrackedObject(
            id=obj_id,
            position=detection.position,
            velocity=Vector2(0, 0),  # 初始速度为0
            acceleration=Vector2(0, 0),  # 初始加速度为0
            object_type=detection.object_type,
            size=detection.size,
            confidence=detection.confidence,
            last_update=current_time
        )

        # 预测轨迹
        tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
            detection.position,
            Vector2(0, 0),  # 初始速度为0
            prediction_type=PredictionType.CONSTANT_VELOCITY
        )

        self.tracked_objects[obj_id] = tracked_obj
        self.kalman_filters[obj_id] = kf

    def update_tracked_object(self, obj_id, detection, current_time):
        """更新跟踪对象"""
        tracked_obj = self.tracked_objects[obj_id]
        kf = self.kalman_filters[obj_id]

        # 使用卡尔曼滤波器更新状态
        measurement = np.array([detection.position.x, detection.position.y])
        kf.update(measurement)

        # 计算速度和加速度（简单差分）
        dt = current_time - tracked_obj.last_update
        if dt > 0:
            new_velocity = (detection.position - tracked_obj.position) / dt
            new_acceleration = (new_velocity - tracked_obj.velocity) / dt

            # 更新跟踪对象
            tracked_obj.position = detection.position
            tracked_obj.velocity = new_velocity
            tracked_obj.acceleration = new_acceleration
            tracked_obj.confidence = detection.confidence
            tracked_obj.last_update = current_time

            # 预测轨迹
            tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
                detection.position,
                new_velocity,
                new_acceleration,
                prediction_type=PredictionType.CONSTANT_ACCELERATION
            )

    def predict_tracked_object(self, obj_id, current_time):
        """预测跟踪对象的状态"""
        tracked_obj = self.tracked_objects[obj_id]
        kf = self.kalman_filters[obj_id]

        # 使用卡尔曼滤波器预测状态
        predicted_position = kf.predict()

        # 更新跟踪对象的位置（仅预测，不更新速度和加速度）
        tracked_obj.position = Vector2(predicted_position[0], predicted_position[1])

        # 预测轨迹（基于最后已知的速度和加速度）
        tracked_obj.prediction_horizon = self.trajectory_predictor.predict_trajectory(
            tracked_obj.position,
            tracked_obj.velocity,
            Vector2(0, 0),
            prediction_type=PredictionType.CONSTANT_ACCELERATION
        )

    def remove_old_objects(self, current_time):
        """移除长时间未更新的跟踪对象"""
        objects_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            # Remove objects that haven't been updated for a while
            if current_time - tracked_obj.last_update > self.max_age:
                objects_to_remove.append(obj_id)
            # Also remove objects that are very far from ego vehicle (assuming ego at 0,0 for simplicity)
            # This helps prevent stale collision risks from distant objects
            elif tracked_obj.position.length() > 500:  # 500 pixels away
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]

    def get_collision_risk(self, ego_position, ego_velocity, time_horizon=3.0, safety_distance=20.0):
        """计算与所有跟踪对象的碰撞风险"""
        collision_risks = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            # 计算自我车辆的未来轨迹
            ego_trajectory = self.trajectory_predictor.predict_trajectory(
                ego_position, ego_velocity, Vector2(0, 0),
                prediction_type=PredictionType.CONSTANT_VELOCITY
            )

            # 计算最小距离
            min_distance = float('inf')
            collision_time = float('inf')

            for i, (ego_pos, obj_pos) in enumerate(zip(ego_trajectory, tracked_obj.prediction_horizon)):
                distance = ego_pos.distance_to(obj_pos)
                if distance < min_distance:
                    min_distance = distance
                    collision_time = i * self.trajectory_predictor.time_step

                # 如果距离小于安全距离，标记为碰撞风险
                if distance < safety_distance:
                    collision_risks.append({
                        'object_id': obj_id,
                        'object_type': tracked_obj.object_type,
                        'min_distance': min_distance,
                        'collision_time': collision_time,
                        'risk_level': self.calculate_risk_level(min_distance, collision_time)
                    })
                    print(f"Collision risk detected: {tracked_obj.object_type} at {distance:.1f} pixels")
                    break
                # 如果物体距离很远，不再考虑碰撞风险
                elif distance > 200:  # 200像素以外的物体不考虑碰撞风险
                    break

        return collision_risks

    def calculate_risk_level(self, distance, time_to_collision):
        """Calculate risk level"""
        if time_to_collision < 1.0 and distance < 10.0:
            return "high_risk"
        elif time_to_collision < 2.0 and distance < 20.0:
            return "medium_risk"
        elif time_to_collision < 3.0 and distance < 30.0:
            return "low_risk"
        else:
            return "no_risk"