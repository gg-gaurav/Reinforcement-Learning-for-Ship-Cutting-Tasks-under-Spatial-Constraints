# Reinforcement-Learning-for-Ship-Cutting-Tasks-under-Spatial-Constraints

### Project Overview
Research Project focused on using Reinforcement Learning to train a 7-DoF Franka Panda Emika Robot to perform the ship cutting task while adhering to position and orientation constraints. This work explores progressive complexity in constraint handling, from simple reach tasks to full 7-DoF constrained trajectory following.

### Motivation
The project investigates whether RL-based policies can learn to navigate complex geometric constraints while maintaining tool orientation.

### Robot Platform
- **Robot**: Franka Panda Emika Robot (7-DoF Manipulator)
- **Simulation Environment**: PyBullet with PandaGym
- **RL Framework**: Stable-Baselines3
- **RL Algorithm**: DDPG 

### Project Structure
The project follows an incremental development approach, with each test building upon previous capabilities:

#### Test0 - SB + Panda
**Basic reach task using StableBaseline3**  
Initial experiments with the robot in the default PandaGym environment. The robot learns to complete a simple reach task where it must move the end-effector to a specific target location.  
**Purpose**: Baseline RL setup and environment validation

#### Test1 - Panda + Path Following
**Version 1.0: Initial path following experiments**  
First attempt at teaching the robot to follow a path using RL. This version explored basic trajectory following concepts but was not developed further due to identified limitations.  
**Status**: Discontinued - superseded by Test2

#### Test2 - Panda + Path_v2
**Version 2.0: Straight line path with spatial constraints**  
Improved path following implementation where the robot must move the end-effector from a start location to an end location while staying within defined spatial bounds.  
**Key Features**:
- Straight line trajectory (no intermediate points)
- Spatial boundary constraints

#### Test3 - Panda + MultiSizedBoxes
**Variable spatial constraints**  
Experiments with dynamic spatial constraints where the boundary dimensions vary across different episodes/runs. This tested the robot's ability to generalize across different constraint geometries  
**Status**: Discontinued - concept integrated into later tests.

#### Test4 - Panda + Curve
**Multi-waypoint trajectory following**  
Advanced path following where the robot must traverse through multiple intermediate waypoints while maintaining spatial constraints.  
**Key Features**:
- Multiple intermediate waypoints
- Spatial boundary enforcement through multiple segments of the path

#### Test5 - Panda + Orient
**Full 7-DoF constrained control**  
Complete implementation combining position and orientation constraints. The robot must:
- Navigate through all intermediate waypoints
- Reach the final end location
- Maintain spatial constraints
- Satisfy orientation constraints throughout the trajectory
**Key Features**:
- 6 DoF constraint handling (3 position + 3 orientation)
- End-effector orientation control
- Most comprehensive setup

#### Test6 - Final Ablation Study
**Performance evaluation and analysis**  
Does the final ablation study for different kinds of reward formualation for various constraint setups.  
**Purpose**:
- Performance benchmarking
- Constraint satisfation analysis
- Comparison across different test configurations
- Comparison across different reward formulations
- Final results and metrics

### Usage


### Research Context
This project was developed as part of a research project at IGMR, RWTH Aachen. The goal is to advance RL-based control methods for constrained manipulation tasks in industrial robotics applications.

### Acknowledgements
- Panda-Gym for the simulation environment
- Stable-Baselines3 for RL implementations

