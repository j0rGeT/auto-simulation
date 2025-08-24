import math
from .interfaces import ObstacleAvoidanceAlgorithm
from ..utils.vectors import Vector2

class LaneChangeAvoidanceAlgorithm(ObstacleAvoidanceAlgorithm):
    """Lane Change Avoidance Algorithm"""
    def perform_avoidance(self, vehicle, collision_risks):
        if vehicle.current_road and vehicle.current_road.lanes > 1:
            # Lane change avoidance
            vehicle.current_lane = (vehicle.current_lane + 1) % vehicle.current_road.lanes
            vehicle.avoidance_maneuver = "lane_change"
        else:
            # Slow down avoidance
            vehicle.target_speed *= 0.7
            vehicle.avoidance_maneuver = "slow_down"

class PredictiveAvoidanceAlgorithm(ObstacleAvoidanceAlgorithm):
    """Predictive Avoidance Algorithm"""
    def perform_avoidance(self, vehicle, collision_risks):
        # Analyze all collision risks and choose the best avoidance strategy
        if not collision_risks:
            return

        # Find the closest collision risk
        closest_risk = min(collision_risks, key=lambda x: x['collision_time'])
        
        if closest_risk['collision_time'] < 1.0:  # Emergency situation
            vehicle.target_speed = 0
            vehicle.avoidance_maneuver = "emergency_brake"
        elif closest_risk['collision_time'] < 2.0:  # Need quick response
            if vehicle.current_road and vehicle.current_road.lanes > 1:
                vehicle.current_lane = (vehicle.current_lane + 1) % vehicle.current_road.lanes
                vehicle.avoidance_maneuver = "lane_change"
            else:
                vehicle.target_speed *= 0.5
                vehicle.avoidance_maneuver = "slow_down"
        else:  # Have enough time to react
            # Choose avoidance strategy based on risk direction
            risk_object = next((obj for obj in vehicle.tracked_objects 
                              if obj.id == closest_risk['object_id']), None)
            if risk_object:
                relative_pos = risk_object.position - vehicle.position
                # Simple judgment: if on the left, avoid to the right, and vice versa
                if relative_pos.dot(vehicle.current_road.normal) > 0:
                    # Risk on the left, avoid to the right
                    if vehicle.current_lane > 0:
                        vehicle.current_lane -= 1
                    vehicle.avoidance_maneuver = "lane_change_right"
                else:
                    # Risk on the right, avoid to the left
                    if vehicle.current_lane < vehicle.current_road.lanes - 1:
                        vehicle.current_lane += 1
                    vehicle.avoidance_maneuver = "lane_change_left"

class PedestrianAvoidanceAlgorithm(ObstacleAvoidanceAlgorithm):
    """Pedestrian-Specific Avoidance Algorithm - Stop and Wait"""
    def perform_avoidance(self, vehicle, collision_risks):
        # Filter pedestrian collision risks
        pedestrian_risks = [risk for risk in collision_risks 
                           if self._is_pedestrian_risk(risk, vehicle.tracked_objects)]
        
        if not pedestrian_risks:
            # No pedestrian risks, use standard avoidance
            predictive_algorithm = PredictiveAvoidanceAlgorithm()
            predictive_algorithm.perform_avoidance(vehicle, collision_risks)
            return

        # Check if any pedestrian is in the vehicle's path and poses collision risk
        pedestrians_in_path = self._get_pedestrians_in_path(vehicle, pedestrian_risks)
        
        if pedestrians_in_path:
            # Stop completely and wait for pedestrians to clear
            vehicle.target_speed = 0
            vehicle.avoidance_maneuver = "wait_for_pedestrians"
            
            # Check if pedestrians have cleared the path
            if self._pedestrians_cleared(vehicle, pedestrians_in_path):
                # Resume normal operation
                vehicle.avoidance_maneuver = None
        else:
            # No pedestrians in immediate path, use standard avoidance
            predictive_algorithm = PredictiveAvoidanceAlgorithm()
            predictive_algorithm.perform_avoidance(vehicle, collision_risks)

    def _is_pedestrian_risk(self, risk, tracked_objects):
        """Check if risk is from a pedestrian"""
        for obj in tracked_objects:
            if obj.id == risk['object_id'] and hasattr(obj, 'object_type'):
                return obj.object_type == "pedestrian"
        return False

    def _get_pedestrians_in_path(self, vehicle, pedestrian_risks):
        """Get pedestrians that are in the vehicle's predicted path"""
        pedestrians_in_path = []
        
        for risk in pedestrian_risks:
            # Only consider medium and high risk pedestrians
            if risk['risk_level'] not in ["medium_risk", "high_risk"]:
                continue
                
            # Find the pedestrian object
            pedestrian = None
            for obj in vehicle.tracked_objects:
                if obj.id == risk['object_id'] and obj.object_type == "pedestrian":
                    pedestrian = obj
                    break
            
            if pedestrian and self._is_in_vehicle_path(vehicle, pedestrian):
                pedestrians_in_path.append(pedestrian)
        
        return pedestrians_in_path

    def _is_in_vehicle_path(self, vehicle, pedestrian):
        """Check if pedestrian is in the vehicle's predicted path"""
        # Calculate vehicle's forward direction
        vehicle_direction = Vector2(math.cos(math.radians(vehicle.angle)), 
                                  math.sin(math.radians(vehicle.angle)))
        
        # Calculate relative position of pedestrian
        relative_pos = pedestrian.position - vehicle.position
        
        if relative_pos.length() == 0:
            return False
            
        # Check if pedestrian is in front of vehicle (within 45 degrees)
        relative_dir = relative_pos.normalized()
        dot_product = relative_dir.dot(vehicle_direction)
        
        if dot_product > 0.7:  # ~45 degrees field of view
            # Check distance (within 30 meters)
            distance = relative_pos.length()
            if distance < 30:
                return True
        
        return False

    def _pedestrians_cleared(self, vehicle, pedestrians):
        """Check if pedestrians have cleared the vehicle's path"""
        for pedestrian in pedestrians:
            if self._is_in_vehicle_path(vehicle, pedestrian):
                return False
        return True