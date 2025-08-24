# 识别算法实现
from .interfaces import TrafficLightRecognitionAlgorithm, ObstacleRecognitionAlgorithm
from ..entities.datatypes import Detection

class BasicTrafficLightRecognitionAlgorithm(TrafficLightRecognitionAlgorithm):
    def recognize_traffic_lights(self, sensor_data, vehicle_position):
        """基础交通灯识别算法"""
        recognized_lights = []
        
        for detection in sensor_data:
            if detection.object_type == "traffic_light":
                # 简单实现：根据距离和置信度判断交通灯状态
                distance = (detection.position - vehicle_position).length()
                
                # 距离越近，识别越准确
                if distance < 50:  # 50像素范围内
                    confidence = detection.confidence * (1 - distance / 100)
                    if confidence > 0.6:
                        recognized_lights.append({
                            'position': detection.position,
                            'confidence': confidence,
                            'distance': distance
                        })
        
        return recognized_lights

class AdvancedTrafficLightRecognitionAlgorithm(TrafficLightRecognitionAlgorithm):
    def __init__(self):
        self.previous_detections = []
        
    def recognize_traffic_lights(self, sensor_data, vehicle_position):
        """高级交通灯识别算法"""
        recognized_lights = []
        
        # 使用多帧数据进行跟踪和验证
        current_detections = []
        for detection in sensor_data:
            if detection.object_type == "traffic_light":
                distance = (detection.position - vehicle_position).length()
                
                # 基于距离和置信度的加权评分
                score = detection.confidence * max(0, 1 - distance / 200)
                
                if score > 0.4:
                    current_detections.append({
                        'position': detection.position,
                        'confidence': detection.confidence,
                        'distance': distance,
                        'score': score
                    })
        
        # 与前一帧数据进行匹配和跟踪
        matched_detections = self._match_detections(current_detections)
        
        # 更新历史数据
        self.previous_detections = current_detections
        
        return matched_detections
    
    def _match_detections(self, current_detections):
        """匹配当前帧和前一帧的检测结果"""
        matched = []
        
        for current in current_detections:
            # 寻找前一帧中最近的检测
            best_match = None
            min_distance = float('inf')
            
            for prev in self.previous_detections:
                distance = (current['position'] - prev['position']).length()
                if distance < min_distance and distance < 20:  # 20像素匹配阈值
                    min_distance = distance
                    best_match = prev
            
            if best_match:
                # 如果找到匹配，使用加权平均提高置信度
                current['confidence'] = (current['confidence'] + best_match['confidence']) / 2
                current['score'] = (current['score'] + best_match['score']) / 2
            
            matched.append(current)
        
        return matched

class BasicObstacleRecognitionAlgorithm(ObstacleRecognitionAlgorithm):
    def recognize_obstacles(self, sensor_data, vehicle_position):
        """基础障碍物识别算法"""
        recognized_obstacles = []
        
        for detection in sensor_data:
            if detection.object_type == "obstacle":
                distance = (detection.position - vehicle_position).length()
                
                # 根据距离和置信度判断障碍物
                if distance < 100:  # 100像素范围内
                    confidence = detection.confidence * (1 - distance / 150)
                    if confidence > 0.5:
                        recognized_obstacles.append({
                            'position': detection.position,
                            'size': detection.size,
                            'confidence': confidence,
                            'distance': distance,
                            'type': 'static'  # 默认静态障碍物
                        })
        
        return recognized_obstacles

class AdvancedObstacleRecognitionAlgorithm(ObstacleRecognitionAlgorithm):
    def __init__(self):
        self.obstacle_history = {}
        self.next_id = 1
        
    def recognize_obstacles(self, sensor_data, vehicle_position):
        """高级障碍物识别算法"""
        current_obstacles = []
        
        for detection in sensor_data:
            if detection.object_type == "obstacle":
                distance = (detection.position - vehicle_position).length()
                
                # 计算障碍物威胁等级
                threat_level = self._calculate_threat_level(distance, detection.confidence)
                
                if threat_level > 0.3:
                    current_obstacles.append({
                        'position': detection.position,
                        'size': detection.size,
                        'confidence': detection.confidence,
                        'distance': distance,
                        'threat_level': threat_level,
                        'type': self._classify_obstacle(detection.size)
                    })
        
        # 跟踪和更新障碍物
        tracked_obstacles = self._track_obstacles(current_obstacles)
        
        return tracked_obstacles
    
    def _calculate_threat_level(self, distance, confidence):
        """计算障碍物威胁等级"""
        # 距离越近，威胁越大
        distance_factor = max(0, 1 - distance / 200)
        # 置信度越高，威胁评估越准确
        return confidence * distance_factor
    
    def _classify_obstacle(self, size):
        """根据尺寸分类障碍物"""
        area = size.x * size.y
        if area < 100:
            return "small"
        elif area < 400:
            return "medium"
        else:
            return "large"
    
    def _track_obstacles(self, current_obstacles):
        """跟踪障碍物并分配ID"""
        tracked = []
        
        for obstacle in current_obstacles:
            # 寻找历史中最近的障碍物
            best_match_id = None
            min_distance = float('inf')
            
            for obs_id, history in self.obstacle_history.items():
                last_pos = history['positions'][-1] if history['positions'] else None
                if last_pos:
                    distance = (obstacle['position'] - last_pos).length()
                    if distance < min_distance and distance < 30:  # 30像素匹配阈值
                        min_distance = distance
                        best_match_id = obs_id
            
            if best_match_id:
                # 更新现有障碍物
                obstacle['id'] = best_match_id
                self.obstacle_history[best_match_id]['positions'].append(obstacle['position'])
                self.obstacle_history[best_match_id]['confidence'] = obstacle['confidence']
            else:
                # 新障碍物
                obstacle['id'] = self.next_id
                self.obstacle_history[self.next_id] = {
                    'positions': [obstacle['position']],
                    'confidence': obstacle['confidence']
                }
                self.next_id += 1
            
            tracked.append(obstacle)
        
        # 清理长时间未检测到的障碍物
        self._cleanup_old_obstacles()
        
        return tracked
    
    def _cleanup_old_obstacles(self):
        """清理长时间未检测到的障碍物"""
        ids_to_remove = []
        for obs_id in self.obstacle_history:
            if len(self.obstacle_history[obs_id]['positions']) > 10:  # 保留最近10个位置
                self.obstacle_history[obs_id]['positions'] = self.obstacle_history[obs_id]['positions'][-10:]
            # 这里可以添加更复杂的清理逻辑
        
        return ids_to_remove