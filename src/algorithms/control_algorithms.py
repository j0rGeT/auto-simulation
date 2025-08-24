# English comment
import numpy as np
import time
from .interfaces import ControlAlgorithm
from ..utils.vectors import Vector2
import math

class PIDControlAlgorithm(ControlAlgorithm):
    def __init__(self, kp_speed=0.5, ki_speed=0.1, kd_speed=0.2,
                 kp_steering=0.8, ki_steering=0.05, kd_steering=0.3,
                 kp_brake=1.0, ki_brake=0.2, kd_brake=0.1):
        # English comment
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.kp_steering = kp_steering
        self.ki_steering = ki_steering
        self.kd_steering = kd_steering
        self.kp_brake = kp_brake
        self.ki_brake = ki_brake
        self.kd_brake = kd_brake
        
        # English comment
        self.integral_speed = 0
        self.integral_steering = 0
        self.integral_brake = 0
        self.prev_error_speed = 0
        self.prev_error_steering = 0
        self.prev_error_brake = 0
        
        # State tracking for pedestrian blockage
        self.lanes_blocked = False
        self.blockage_start_time = 0
        self.blockage_duration = 0
    
    def compute_control(self, vehicle, target_state, obstacles, dt):
        """English docstring"""
        # Check pedestrian blockage state and handle resume logic
        current_blocked = self._all_lanes_blocked_by_pedestrians(vehicle, obstacles)
        
        # Update blockage state and timing
        if current_blocked and not self.lanes_blocked:
            # Just became blocked
            self.lanes_blocked = True
            self.blockage_start_time = time.time()
            self.blockage_duration = 0
            print(f"Blockage started at {self.blockage_start_time}")
        elif not current_blocked and self.lanes_blocked:
            # Just became unblocked - pedestrians cleared
            self.lanes_blocked = False
            self.blockage_start_time = 0
            self.blockage_duration = 0
            print(f"Blockage cleared - pedestrians gone, resetting state")
        elif self.lanes_blocked:
            # Still blocked, update duration
            self.blockage_duration = time.time() - self.blockage_start_time
            print(f"Still blocked for {self.blockage_duration:.2f} seconds")
        
        # English comment
        speed_error = target_state.get('target_speed', 0) - vehicle.dynamics.velocity
        self.integral_speed += speed_error * dt
        derivative_speed = (speed_error - self.prev_error_speed) / dt if dt > 0 else 0
        
        throttle_output = (self.kp_speed * speed_error + 
                          self.ki_speed * self.integral_speed + 
                          self.kd_speed * derivative_speed)
        
        # English comment
        if 'target_position' in target_state:
            target_pos = target_state['target_position']
            lateral_error = self._calculate_lateral_error(vehicle, target_pos)
            
            self.integral_steering += lateral_error * dt
            derivative_steering = (lateral_error - self.prev_error_steering) / dt if dt > 0 else 0
            
            steering_output = (self.kp_steering * lateral_error + 
                              self.ki_steering * self.integral_steering + 
                              self.kd_steering * derivative_steering)
        else:
            steering_output = 0
        
        # English comment
        brake_error = self._calculate_brake_error(vehicle, obstacles)
        self.integral_brake += brake_error * dt
        derivative_brake = (brake_error - self.prev_error_brake) / dt if dt > 0 else 0
        
        brake_output = (self.kp_brake * brake_error + 
                       self.ki_brake * self.integral_brake + 
                       self.kd_brake * derivative_brake)
        
        # English comment
        self.prev_error_speed = speed_error
        self.prev_error_steering = lateral_error if 'target_position' in target_state else 0
        self.prev_error_brake = brake_error
        
        # English comment
        throttle_output = np.clip(throttle_output, 0, 1)
        steering_output = np.clip(steering_output, -1, 1)
        brake_output = np.clip(brake_output, 0, 1)
        
        # Make throttle and brake mutually exclusive
        if brake_output > 0.1:  # If brake is being applied significantly
            throttle_output = 0  # Cut throttle completely
        
        return {
            'throttle': throttle_output,
            'brake': brake_output,
            'steering': steering_output
        }
    
    def _calculate_lateral_error(self, vehicle, target_position):
        """English docstring"""
        if vehicle.current_road:
            # English comment
            lane_center = vehicle.current_road.get_lane_center(vehicle.current_lane, vehicle.road_progress)
            lateral_error = (lane_center - vehicle.position).dot(vehicle.current_road.normal)
            return lateral_error
        else:
            # English comment
            to_target = target_position - vehicle.position
            target_angle = math.degrees(math.atan2(to_target.y, to_target.x))
            angle_error = target_angle - vehicle.angle
            # English comment
            while angle_error > 180:
                angle_error -= 360
            while angle_error < -180:
                angle_error += 360
            return angle_error / 180.0  # English comment
    
    def _calculate_brake_error(self, vehicle, obstacles):
        """Calculate brake error based on obstacles and pedestrian density"""
        min_distance = float('inf')
        
        # Use state tracking for pedestrian blockage
        if self.lanes_blocked:
            # Auto-resume after 3 seconds of blockage
            if self.blockage_duration > 3.0:
                self.lanes_blocked = False
                self.blockage_duration = 0
                print(f"Control: Auto-resuming after 3.0 seconds of blockage")
                return 0.0  # Resume driving
            print(f"Control: Blocked for {self.blockage_duration:.2f} seconds, still waiting")
            return 1.0  # Full brake - lanes are blocked
        else:
            print(f"Control: Not blocked, calculating normal brake")
        
        for obstacle in obstacles:
            distance = (obstacle['position'] - vehicle.position).length()
            if distance < min_distance:
                min_distance = distance
                print(f"Nearest obstacle: {obstacle.get('type', 'unknown')} at {distance:.1f} pixels")
        
        safe_distance = 50
        if min_distance < safe_distance:
            return (safe_distance - min_distance) / safe_distance
        return 0
    
    def _all_lanes_blocked_by_pedestrians(self, vehicle, obstacles):
        """Check if all available lanes are blocked by pedestrians"""
        if not vehicle.current_road:
            return False
            
        # Count pedestrians in each lane
        lanes_blocked = set()
        
        for obstacle in obstacles:
            if obstacle.get('type') == 'pedestrian':
                # Check which lane this pedestrian is in
                pedestrian_lane = self._get_lane_for_position(vehicle.current_road, obstacle['position'])
                if pedestrian_lane is not None:
                    lanes_blocked.add(pedestrian_lane)
        
        # If all lanes have pedestrians, return True
        return len(lanes_blocked) >= vehicle.current_road.lanes
    
    def _get_lane_for_position(self, road, position):
        """Determine which lane a position is in"""
        # Calculate projection onto road
        road_vec = road.end - road.start
        road_len = road_vec.length()
        if road_len == 0:
            return None
            
        road_unit = road_vec / road_len
        road_normal = Vector2(-road_unit.y, road_unit.x)
        
        # Calculate lateral offset
        lateral_offset = (position - road.start).dot(road_normal)
        
        # Determine lane
        lane_width = road.width / road.lanes
        lane_index = int((lateral_offset + road.width / 2) / lane_width)
        
        if 0 <= lane_index < road.lanes:
            return lane_index
        return None

class LQRControlAlgorithm(ControlAlgorithm):
    def __init__(self, Q=None, R=None):
        # English comment
        self.Q = Q if Q is not None else np.diag([1.0, 1.0, 0.5, 0.5])  # [x, y, v, theta]
        self.R = R if R is not None else np.diag([0.1, 0.1])  # [throttle, steering]
        
        # State tracking for pedestrian blockage
        self.lanes_blocked = False
        self.blockage_start_time = 0
        self.blockage_duration = 0
        
    def compute_control(self, vehicle, target_state, obstacles, dt):
        """English docstring"""
        # Check pedestrian blockage state and handle resume logic
        current_blocked = self._all_lanes_blocked_by_pedestrians(vehicle, obstacles)
        
        # Update blockage state and timing
        if current_blocked and not self.lanes_blocked:
            # Just became blocked
            self.lanes_blocked = True
            self.blockage_start_time = time.time()
            self.blockage_duration = 0
            print(f"Blockage started at {self.blockage_start_time}")
        elif not current_blocked and self.lanes_blocked:
            # Just became unblocked - pedestrians cleared
            self.lanes_blocked = False
            self.blockage_start_time = 0
            self.blockage_duration = 0
            print(f"Blockage cleared - pedestrians gone, resetting state")
        elif self.lanes_blocked:
            # Still blocked, update duration
            self.blockage_duration = time.time() - self.blockage_start_time
            print(f"Still blocked for {self.blockage_duration:.2f} seconds")
        
        # English comment
        x = vehicle.position.x
        y = vehicle.position.y
        v = vehicle.dynamics.velocity
        theta = math.radians(vehicle.angle)
        
        # English comment
        target_x = target_state.get('target_position', vehicle.position).x
        target_y = target_state.get('target_position', vehicle.position).y
        target_v = target_state.get('target_speed', 0)
        target_theta = math.radians(target_state.get('target_angle', vehicle.angle))
        
        # English comment
        state_error = np.array([
            x - target_x,
            y - target_y,
            v - target_v,
            theta - target_theta
        ])
        
        # English comment
        A = np.array([
            [1, 0, dt * np.cos(theta), -v * dt * np.sin(theta)],
            [0, 1, dt * np.sin(theta), v * dt * np.cos(theta)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        B = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],  # English comment
            [0, dt]   # English comment
        ])
        
        # English comment
        P = self._solve_riccati(A, B, self.Q, self.R)
        
        # English comment
        K = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
        
        # English comment
        control_input = -K @ state_error
        
        throttle = np.clip(control_input[0], 0, 1)
        steering = np.clip(control_input[1], -1, 1)
        
        # English comment
        brake = self._calculate_brake(vehicle, obstacles)
        
        # Make throttle and brake mutually exclusive
        if brake > 0.1:  # If brake is being applied significantly
            throttle = 0  # Cut throttle completely
        
        return {
            'throttle': throttle,
            'brake': brake,
            'steering': steering
        }
    
    def _solve_riccati(self, A, B, Q, R, max_iter=100, tol=1e-6):
        """English docstring"""
        P = Q.copy()
        
        for _ in range(max_iter):
            P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            
            if np.max(np.abs(P_next - P)) < tol:
                break
            P = P_next
        
        return P
    
    def _calculate_brake(self, vehicle, obstacles):
        """Calculate brake based on obstacles and pedestrian density"""
        # Use state tracking for pedestrian blockage
        if self.lanes_blocked:
            # Auto-resume after 3 seconds of blockage
            if self.blockage_duration > 3.0:
                self.lanes_blocked = False
                self.blockage_duration = 0
                print(f"Control: Auto-resuming after 3.0 seconds of blockage")
                return 0.0  # Resume driving
            print(f"Control: Blocked for {self.blockage_duration:.2f} seconds, still waiting")
            return 1.0  # Full brake - lanes are blocked
        else:
            print(f"Control: Not blocked, calculating normal brake")
        
        min_distance = float('inf')
        
        for obstacle in obstacles:
            distance = (obstacle['position'] - vehicle.position).length()
            if distance < min_distance:
                min_distance = distance
                print(f"Nearest obstacle: {obstacle.get('type', 'unknown')} at {distance:.1f} pixels")
        
        safe_distance = 50
        if min_distance < safe_distance:
            return min(1.0, (safe_distance - min_distance) / safe_distance)
        return 0
    
    def _all_lanes_blocked_by_pedestrians(self, vehicle, obstacles):
        """Check if all available lanes are blocked by pedestrians"""
        if not vehicle.current_road:
            return False
            
        # Count pedestrians in each lane
        lanes_blocked = set()
        
        for obstacle in obstacles:
            if obstacle.get('type') == 'pedestrian':
                # Check which lane this pedestrian is in
                pedestrian_lane = self._get_lane_for_position(vehicle.current_road, obstacle['position'])
                if pedestrian_lane is not None:
                    lanes_blocked.add(pedestrian_lane)
        
        # If all lanes have pedestrians, return True
        return len(lanes_blocked) >= vehicle.current_road.lanes
    
    def _get_lane_for_position(self, road, position):
        """Determine which lane a position is in"""
        # Calculate projection onto road
        road_vec = road.end - road.start
        road_len = road_vec.length()
        if road_len == 0:
            return None
            
        road_unit = road_vec / road_len
        road_normal = Vector2(-road_unit.y, road_unit.x)
        
        # Calculate lateral offset
        lateral_offset = (position - road.start).dot(road_normal)
        
        # Determine lane
        lane_width = road.width / road.lanes
        lane_index = int((lateral_offset + road.width / 2) / lane_width)
        
        if 0 <= lane_index < road.lanes:
            return lane_index
        return None

class MPCControlAlgorithm(ControlAlgorithm):
    def __init__(self, horizon=10, Q=None, R=None):
        self.horizon = horizon
        self.Q = Q if Q is not None else np.diag([1.0, 1.0, 0.5, 0.5])
        self.R = R if R is not None else np.diag([0.1, 0.1])
        
        # State tracking for pedestrian blockage
        self.lanes_blocked = False
        self.blockage_start_time = 0
        self.blockage_duration = 0
        
    def compute_control(self, vehicle, target_state, obstacles, dt):
        """English docstring"""
        # Check pedestrian blockage state and handle resume logic
        current_blocked = self._all_lanes_blocked_by_pedestrians(vehicle, obstacles)
        
        # Update blockage state and timing
        if current_blocked and not self.lanes_blocked:
            # Just became blocked
            self.lanes_blocked = True
            self.blockage_start_time = time.time()
            self.blockage_duration = 0
            print(f"Blockage started at {self.blockage_start_time}")
        elif not current_blocked and self.lanes_blocked:
            # Just became unblocked - pedestrians cleared
            self.lanes_blocked = False
            self.blockage_start_time = 0
            self.blockage_duration = 0
            print(f"Blockage cleared - pedestrians gone, resetting state")
        elif self.lanes_blocked:
            # Still blocked, update duration
            self.blockage_duration = time.time() - self.blockage_start_time
            print(f"Still blocked for {self.blockage_duration:.2f} seconds")
        
        # English comment
        current_state = np.array([
            vehicle.position.x,
            vehicle.position.y,
            vehicle.dynamics.velocity,
            math.radians(vehicle.angle)
        ])
        
        # English comment
        ref_trajectory = self._generate_reference_trajectory(current_state, target_state, self.horizon, dt)
        
        # English comment
        control_sequence = self._optimize_control(current_state, ref_trajectory, obstacles, dt)
        
        # English comment
        throttle = np.clip(control_sequence[0, 0], 0, 1)
        steering = np.clip(control_sequence[0, 1], -1, 1)
        
        # English comment
        brake = self._calculate_brake(vehicle, obstacles)
        
        # Make throttle and brake mutually exclusive
        if brake > 0.1:  # If brake is being applied significantly
            throttle = 0  # Cut throttle completely
        
        return {
            'throttle': throttle,
            'brake': brake,
            'steering': steering
        }
    
    def _generate_reference_trajectory(self, current_state, target_state, horizon, dt):
        """English docstring"""
        ref_traj = []
        
        target_x = target_state.get('target_position', Vector2(current_state[0], current_state[1])).x
        target_y = target_state.get('target_position', Vector2(current_state[0], current_state[1])).y
        target_v = target_state.get('target_speed', 0)
        target_theta = math.radians(target_state.get('target_angle', math.degrees(current_state[3])))
        
        for i in range(horizon):
            ref_traj.append(np.array([target_x, target_y, target_v, target_theta]))
        
        return np.array(ref_traj)
    
    def _optimize_control(self, current_state, ref_trajectory, obstacles, dt):
        """English docstring"""
        # English comment
        best_control = np.zeros((self.horizon, 2))
        best_cost = float('inf')
        
        # English comment
        for throttle in np.linspace(0, 1, 5):
            for steering in np.linspace(-1, 1, 5):
                control_seq = np.tile([throttle, steering], (self.horizon, 1))
                cost = self._calculate_cost(current_state, control_seq, ref_trajectory, obstacles, dt)
                
                if cost < best_cost:
                    best_cost = cost
                    best_control = control_seq
        
        return best_control
    
    def _calculate_cost(self, current_state, control_sequence, ref_trajectory, obstacles, dt):
        """English docstring"""
        state = current_state.copy()
        total_cost = 0
        
        for i in range(self.horizon):
            # English comment
            state = self._update_state(state, control_sequence[i], dt)
            
            # English comment
            error = state - ref_trajectory[i]
            tracking_cost = error.T @ self.Q @ error
            
            # English comment
            control_cost = control_sequence[i].T @ self.R @ control_sequence[i]
            
            # English comment
            obstacle_cost = self._calculate_obstacle_cost(state, obstacles)
            
            total_cost += tracking_cost + control_cost + obstacle_cost
        
        return total_cost
    
    def _update_state(self, state, control, dt):
        """English docstring"""
        x, y, v, theta = state
        throttle, steering = control
        
        # English comment
        acceleration = throttle * 5.0 - v * 0.1  # English comment
        angular_velocity = steering * 0.5  # English comment
        
        new_v = v + acceleration * dt
        new_theta = theta + angular_velocity * dt
        new_x = x + new_v * np.cos(new_theta) * dt
        new_y = y + new_v * np.sin(new_theta) * dt
        
        return np.array([new_x, new_y, new_v, new_theta])
    
    def _calculate_obstacle_cost(self, state, obstacles):
        """English docstring"""
        x, y, _, _ = state
        cost = 0
        
        for obstacle in obstacles:
            obs_x = obstacle['position'].x
            obs_y = obstacle['position'].y
            distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            
            if distance < 50:  # English comment
                cost += 1000 * (1 - distance / 50)**2
        
        return cost
    
    def _calculate_brake(self, vehicle, obstacles):
        """Calculate brake based on obstacles and pedestrian density"""
        # Use state tracking for pedestrian blockage
        if self.lanes_blocked:
            # Auto-resume after 3 seconds of blockage
            if self.blockage_duration > 3.0:
                self.lanes_blocked = False
                self.blockage_duration = 0
                print(f"Control: Auto-resuming after 3.0 seconds of blockage")
                return 0.0  # Resume driving
            print(f"Control: Blocked for {self.blockage_duration:.2f} seconds, still waiting")
            return 1.0  # Full brake - lanes are blocked
        else:
            print(f"Control: Not blocked, calculating normal brake")
        
        min_distance = float('inf')
        
        for obstacle in obstacles:
            distance = (obstacle['position'] - vehicle.position).length()
            if distance < min_distance:
                min_distance = distance
                print(f"Nearest obstacle: {obstacle.get('type', 'unknown')} at {distance:.1f} pixels")
        
        safe_distance = 50
        if min_distance < safe_distance:
            return min(1.0, (safe_distance - min_distance) / safe_distance)
        return 0
    
    def _all_lanes_blocked_by_pedestrians(self, vehicle, obstacles):
        """Check if all available lanes are blocked by pedestrians"""
        if not vehicle.current_road:
            return False
            
        # Count pedestrians in each lane
        lanes_blocked = set()
        
        for obstacle in obstacles:
            if obstacle.get('type') == 'pedestrian':
                # Check which lane this pedestrian is in
                pedestrian_lane = self._get_lane_for_position(vehicle.current_road, obstacle['position'])
                if pedestrian_lane is not None:
                    lanes_blocked.add(pedestrian_lane)
        
        # If all lanes have pedestrians, return True
        return len(lanes_blocked) >= vehicle.current_road.lanes
    
    def _get_lane_for_position(self, road, position):
        """Determine which lane a position is in"""
        # Calculate projection onto road
        road_vec = road.end - road.start
        road_len = road_vec.length()
        if road_len == 0:
            return None
            
        road_unit = road_vec / road_len
        road_normal = Vector2(-road_unit.y, road_unit.x)
        
        # Calculate lateral offset
        lateral_offset = (position - road.start).dot(road_normal)
        
        # Determine lane
        lane_width = road.width / road.lanes
        lane_index = int((lateral_offset + road.width / 2) / lane_width)
        
        if 0 <= lane_index < road.lanes:
            return lane_index
        return None