import math
from ..core.enums import TrafficLightState
from .interfaces import DecisionAlgorithm

class BasicDecisionAlgorithm(DecisionAlgorithm):
    """基础决策算法"""
    def make_decision(self, vehicle, objects, traffic_lights, roads):
        # 检查碰撞风险
        emergency_brake = False
        avoidance_needed = False

        for risk in vehicle.collision_risks:
            print(f"Collision risk: {risk['risk_level']}, type: {risk['object_type']}, distance: {risk['min_distance']:.1f}")
            if risk['risk_level'] == "high_risk":
                emergency_brake = True
                print("HIGH RISK DETECTED - EMERGENCY BRAKE")
                break
            elif risk['risk_level'] == "medium_risk":
                avoidance_needed = True
                print("MEDIUM RISK DETECTED - AVOIDANCE NEEDED")

        # 检查交通灯
        traffic_light_detected = False
        for light in traffic_lights:
            dist = (light.position - vehicle.position).length()
            if dist < 100 and abs(vehicle.angle - math.atan2(light.position.y - vehicle.position.y,
                                                              light.position.x - vehicle.position.x)) < 30:
                if light.state != TrafficLightState.GREEN and light.state != TrafficLightState.LEFT_GREEN:
                    traffic_light_detected = True
                    if dist < 80:
                        vehicle.target_speed = 0

        # Decision logic
        if emergency_brake:
            vehicle.target_speed = 0
            vehicle.avoidance_maneuver = "emergency_brake"
            print(f"Emergency brake active, collision_risks: {len(vehicle.collision_risks)}")
        elif avoidance_needed and not vehicle.avoidance_maneuver:
            vehicle.perform_avoidance_maneuver()
        elif traffic_light_detected:
            vehicle.target_speed = 0
        else:
            # Resume normal speed if no emergencies
            # Check if control algorithm indicates we should resume
            if hasattr(vehicle.control_algorithm, 'lanes_blocked') and not vehicle.control_algorithm.lanes_blocked:
                vehicle.target_speed = 50
                vehicle.avoidance_maneuver = None
                print(f"Decision: Resuming normal speed to 50, lanes_blocked={vehicle.control_algorithm.lanes_blocked}")
            else:
                # Still blocked, maintain current state
                print(f"Decision: Still blocked, maintaining state, lanes_blocked={vehicle.control_algorithm.lanes_blocked}")
                pass

class AdvancedDecisionAlgorithm(DecisionAlgorithm):
    """高级决策算法（考虑更多因素）"""
    def make_decision(self, vehicle, objects, traffic_lights, roads):
        # 更复杂的决策逻辑，考虑车辆密度、道路类型等
        vehicle_density = len([v for v in objects if hasattr(v, 'object_type') and v.object_type == "vehicle"]) / 10.0
        
        # 根据车辆密度调整目标速度
        base_speed = 50
        density_factor = max(0.3, 1.0 - vehicle_density * 0.1)
        target_speed = base_speed * density_factor

        # 检查碰撞风险
        emergency_brake = False
        avoidance_needed = False

        for risk in vehicle.collision_risks:
            print(f"Collision risk: {risk['risk_level']}, type: {risk['object_type']}, distance: {risk['min_distance']:.1f}")
            if risk['risk_level'] == "high_risk":
                emergency_brake = True
                print("HIGH RISK DETECTED - EMERGENCY BRAKE")
                break
            elif risk['risk_level'] == "medium_risk":
                avoidance_needed = True
                print("MEDIUM RISK DETECTED - AVOIDANCE NEEDED")

        # 检查交通灯
        traffic_light_detected = False
        for light in traffic_lights:
            dist = (light.position - vehicle.position).length()
            if dist < 120:  # 更早检测交通灯
                angle_diff = abs(vehicle.angle - math.atan2(light.position.y - vehicle.position.y,
                                                           light.position.x - vehicle.position.x))
                if angle_diff < 45:  # 更大的检测角度
                    if light.state != TrafficLightState.GREEN and light.state != TrafficLightState.LEFT_GREEN:
                        traffic_light_detected = True
                        # 根据距离调整减速策略
                        if dist < 100:
                            target_speed = min(target_speed, 20)
                        if dist < 60:
                            target_speed = 0

        # Decision logic
        if emergency_brake:
            vehicle.target_speed = 0
            vehicle.avoidance_maneuver = "emergency_brake"
            print(f"Emergency brake active, collision_risks: {len(vehicle.collision_risks)}")
        elif avoidance_needed and not vehicle.avoidance_maneuver:
            vehicle.perform_avoidance_maneuver()
        elif traffic_light_detected:
            vehicle.target_speed = target_speed
        else:
            # Resume normal speed if no emergencies
            # Check if control algorithm indicates we should resume
            if hasattr(vehicle.control_algorithm, 'lanes_blocked') and not vehicle.control_algorithm.lanes_blocked:
                vehicle.target_speed = target_speed
                vehicle.avoidance_maneuver = None
                print(f"Advanced Decision: Resuming normal speed to {target_speed}, lanes_blocked={vehicle.control_algorithm.lanes_blocked}")
            else:
                # Still blocked, maintain current state
                print(f"Advanced Decision: Still blocked, maintaining state, lanes_blocked={vehicle.control_algorithm.lanes_blocked}")
                pass