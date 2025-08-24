from ..core.enums import TrafficLightState
from .interfaces import TrafficLightAlgorithm

class FixedCycleTrafficLightAlgorithm(TrafficLightAlgorithm):
    """English docstring"""
    def update_traffic_light(self, traffic_light, dt, vehicles):
        # English comment
        traffic_light.timer += dt
        current_cycle_time = traffic_light.cycle_times[traffic_light.state]

        if traffic_light.timer >= current_cycle_time:
            traffic_light.timer = 0
            if traffic_light.state == TrafficLightState.RED:
                traffic_light.state = TrafficLightState.GREEN
            elif traffic_light.state == TrafficLightState.GREEN:
                traffic_light.state = TrafficLightState.YELLOW
            elif traffic_light.state == TrafficLightState.YELLOW:
                traffic_light.state = TrafficLightState.RED
            elif traffic_light.state == TrafficLightState.LEFT_GREEN:
                traffic_light.state = TrafficLightState.GREEN
                traffic_light.left_turn_state = False

class AdaptiveTrafficLightAlgorithm(TrafficLightAlgorithm):
    """English docstring"""
    def update_traffic_light(self, traffic_light, dt, vehicles):
        # English comment
        vehicles_in_direction = self.count_vehicles_in_direction(traffic_light, vehicles)
        
        # English comment
        min_green_time = 10.0
        max_green_time = 30.0
        vehicle_factor = min(1.0, vehicles_in_direction / 5.0)  # English comment
        
        green_time = min_green_time + (max_green_time - min_green_time) * vehicle_factor
        traffic_light.cycle_times[TrafficLightState.GREEN] = green_time

        # English comment
        traffic_light.timer += dt
        current_cycle_time = traffic_light.cycle_times[traffic_light.state]

        if traffic_light.timer >= current_cycle_time:
            traffic_light.timer = 0
            if traffic_light.state == TrafficLightState.RED:
                traffic_light.state = TrafficLightState.GREEN
            elif traffic_light.state == TrafficLightState.GREEN:
                traffic_light.state = TrafficLightState.YELLOW
            elif traffic_light.state == TrafficLightState.YELLOW:
                traffic_light.state = TrafficLightState.RED
            elif traffic_light.state == TrafficLightState.LEFT_GREEN:
                traffic_light.state = TrafficLightState.GREEN
                traffic_light.left_turn_state = False

    def count_vehicles_in_direction(self, traffic_light, vehicles):
        """English docstring"""
        count = 0
        road_direction = traffic_light.road.direction
        
        for vehicle in vehicles:
            if vehicle.current_road == traffic_light.road:
                # English comment
                to_light = traffic_light.position - vehicle.position
                if to_light.dot(road_direction) > 0:  # English comment
                    count += 1
        
        return count