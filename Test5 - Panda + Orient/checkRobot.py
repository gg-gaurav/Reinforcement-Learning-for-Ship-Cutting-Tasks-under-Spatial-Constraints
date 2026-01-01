
from panda_gym.pybullet import PyBullet
from pandaOrientRobot import pandaOrientRobot
import numpy as np
import time

simulator = PyBullet(render_mode="human")
robot = pandaOrientRobot(sim = simulator)

simulator.physics_client.resetDebugVisualizerCamera(cameraDistance= 1.5,            # Distance to the target position
                                              cameraYaw= 0,                         # adjust viewing angle
                                              cameraPitch= -20,                     # adjust tilt
                                              cameraTargetPosition=[0, 0, 0.5])     # Focus point
                                                                                    # When [Roll, Pitch, Yaw] = [0,0,0]
                                                                                    # [x, y, z] = [right, into_the_screen, up] 
simulator.physics_client.setGravity(1, 0, -9.8)
simulator.physics_client.setRealTimeSimulation(0)  # Disable real-time simulation
time.sleep(4)

for _ in range(50):
    action = robot.action_space.sample()
    print(action)
    robot.set_action(action)
    print("HELP")
    simulator.step()
    time.sleep(1)

