# Auto Simulation System

A comprehensive autonomous vehicle simulation system with advanced perception, decision-making, and control capabilities.

## Features

### Core Functionality
- **Traffic Light System**: Set and control traffic lights with left/right/straight signals
- **Obstacle Management**: Place various types of obstacles (cones, barriers, etc.)
- **Pedestrian Simulation**: Realistic pedestrian behavior with trajectory prediction
- **Vehicle Control**: Advanced control algorithms for autonomous driving

### Sensor Systems
- **Camera**: Visual perception for object detection
- **LiDAR**: 3D point cloud sensing for precise distance measurement
- **Radar**: Robust detection in adverse weather conditions
- **Multi-sensor Fusion**: Kalman filter-based sensor fusion for accurate object tracking

### Advanced Algorithms
- **SLAM**: Simultaneous Localization and Mapping with particle filters
- **BEV System**: Bird's Eye View visualization for situational awareness
- **Trajectory Prediction**: Advanced pedestrian and vehicle trajectory prediction
- **Collision Avoidance**: Multi-lane pedestrian and obstacle avoidance

### Control Algorithms
- **PID Control**: Traditional proportional-integral-derivative control
- **LQR Control**: Linear Quadratic Regulator for optimal control
- **MPC Control**: Model Predictive Control for predictive optimization

## System Architecture

### Key Components
1. **Sensor Fusion** (`src/utils/sensor_fusion.py`): Multi-sensor data integration
2. **SLAM System** (`src/utils/slam_system.py`): Real-time mapping and localization
3. **BEV System** (`src/utils/bev_system.py`): Top-down visualization
4. **Control Algorithms** (`src/algorithms/control_algorithms.py`): PID, LQR, MPC
5. **Decision Algorithms** (`src/algorithms/decision_algorithms.py`): Basic and advanced decision making
6. **Avoidance Algorithms** (`src/algorithms/avoidance_algorithms.py`): Lane change and pedestrian avoidance

### Pedestrian Avoidance Features
- **Multi-lane Detection**: Detects pedestrians across all lanes
- **Risk Assessment**: Classifies collision risk levels (high/medium/low)
- **Smart Stopping**: Stops completely when all lanes are blocked by pedestrians
- **Wait and Resume**: Automatically resumes when path is clear

## Installation & Deployment

### Prerequisites
- Python 3.8+
- Pygame 2.6.1+
- NumPy

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd auto-simulation

# Install dependencies
pip install pygame numpy

# Run the simulation
python main.py
```

### Controls
- **1-5**: Switch edit modes (Road, Vehicle, Traffic Light, Obstacle, Pedestrian)
- **Space**: Pause/Resume simulation
- **S**: Toggle sensor visualization
- **P**: Toggle prediction visualization
- **L**: Switch traffic light to left-turn mode
- **Right Click**: Add waypoints for selected vehicle

### Configuration
Modify `src/core/constants.py` to adjust simulation parameters:
- Screen dimensions
- Vehicle dynamics
- Sensor ranges
- Control parameters

## Algorithm Customization

The system uses interface-based architecture for easy algorithm swapping:

### Control Algorithms
```python
# Switch between PID, LQR, or MPC
vehicle.control_algorithm = PIDControlAlgorithm()
vehicle.control_algorithm = LQRControlAlgorithm()  
vehicle.control_algorithm = MPCControlAlgorithm()
```

### Avoidance Algorithms
```python
# Choose avoidance strategy
vehicle.obstacle_avoidance_algorithm = LaneChangeAvoidanceAlgorithm()
vehicle.obstacle_avoidance_algorithm = PredictiveAvoidanceAlgorithm()
vehicle.obstacle_avoidance_algorithm = PedestrianAvoidanceAlgorithm()  # Default
```

## Performance Features

- Real-time sensor fusion and object tracking
- Advanced pedestrian trajectory prediction
- Multi-lane collision risk assessment
- Adaptive control based on road conditions
- Extensible algorithm architecture

## Development

The codebase follows modular architecture with clear separation of concerns:

- **Entities**: Vehicle, Road, TrafficLight, Obstacle, Pedestrian classes
- **Algorithms**: Control, decision, and avoidance algorithms
- **Utils**: Sensor fusion, SLAM, BEV visualization
- **Core**: Constants, enums, and base types

## License

This project is for educational and research purposes.