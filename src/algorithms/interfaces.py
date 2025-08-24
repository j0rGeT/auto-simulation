# Algorithm interface base classes
class DecisionAlgorithm:
    def make_decision(self, vehicle, objects, traffic_lights, roads):
        """Decision algorithm interface"""
        raise NotImplementedError("DecisionAlgorithm.make_decision() must be implemented")

class TrafficLightAlgorithm:
    def update_traffic_light(self, traffic_light, dt, vehicles):
        """Traffic light control algorithm interface"""
        raise NotImplementedError("TrafficLightAlgorithm.update_traffic_light() must be implemented")

class ObstacleAvoidanceAlgorithm:
    def perform_avoidance(self, vehicle, collision_risks):
        """Obstacle avoidance algorithm interface"""
        raise NotImplementedError("ObstacleAvoidanceAlgorithm.perform_avoidance() must be implemented")

class TrafficLightRecognitionAlgorithm:
    def recognize_traffic_lights(self, sensor_data, vehicle_position):
        """Traffic light recognition algorithm interface"""
        raise NotImplementedError("TrafficLightRecognitionAlgorithm.recognize_traffic_lights() must be implemented")

class ObstacleRecognitionAlgorithm:
    def recognize_obstacles(self, sensor_data, vehicle_position):
        """Obstacle recognition algorithm interface"""
        raise NotImplementedError("ObstacleRecognitionAlgorithm.recognize_obstacles() must be implemented")

class ControlAlgorithm:
    def compute_control(self, vehicle, target_state, obstacles, dt):
        """Control algorithm interface
        
        Args:
            vehicle: Vehicle object
            target_state: Target state (position, velocity, angle, etc.)
            obstacles: List of obstacle information
            dt: Time step
            
        Returns:
            control_output: Control output dictionary containing throttle, brake, steering, etc.
        """
        raise NotImplementedError("ControlAlgorithm.compute_control() must be implemented")

class TrajectoryPredictionAlgorithm:
    def predict_trajectory(self, pedestrian, obstacles, traffic_lights, dt, horizon=5):
        """Pedestrian trajectory prediction algorithm interface
        
        Args:
            pedestrian: Pedestrian object
            obstacles: List of obstacles
            traffic_lights: List of traffic lights
            dt: Time step
            horizon: Prediction horizon (number of steps)
            
        Returns:
            predicted_trajectory: List of predicted trajectory points [position1, position2, ...]
        """
        raise NotImplementedError("TrajectoryPredictionAlgorithm.predict_trajectory() must be implemented")