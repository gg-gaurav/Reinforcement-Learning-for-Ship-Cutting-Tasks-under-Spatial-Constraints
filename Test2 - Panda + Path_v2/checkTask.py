from panda_gym.pybullet import PyBullet
from pandaFollowTask import pandaFollowTask
from pandaFollowRobot import pandaFollowRobot
import time



simulator = PyBullet(render_mode="human")
robot = pandaFollowRobot(sim= simulator)
task = pandaFollowTask(sim= simulator, get_ee_position= robot.get_ee_position, get_ee_velocity= robot.get_ee_velocity)
simulator.physics_client.resetDebugVisualizerCamera(cameraDistance= 1.5,            # Distance to the target position
                                              cameraYaw= 45,                       # adjust viewing angle
                                              cameraPitch= -50,                     # adjust tilt
                                              cameraTargetPosition=[0, 0, 0.5])     # Focus point

task.reset()
print(f"Task observation: {task.get_obs()}")
print(f"EE Position Goal: {task.get_achieved_goal()}")
print(f"Success: {task.is_success(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")
print(f"Reward: {task.compute_reward(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")
time.sleep(50)