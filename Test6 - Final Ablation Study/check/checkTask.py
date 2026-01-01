from panda_gym.pybullet import PyBullet
#from pandaOrientTask import pandaOrientTask
from templateTask import templateTask
from pandaOrientRobot import pandaOrientRobot
import time



simulator = PyBullet(render_mode="human")
robot = pandaOrientRobot(sim= simulator)
"""
task = pandaOrientTask(sim= simulator, 
                       body_id= robot.sim._bodies_idx[robot.body_name],
                       get_ee_position= robot.get_ee_position, 
                       get_ee_velocity= robot.get_ee_velocity,
                       get_ee_orientation= robot.get_ee_orientation,
                       get_joint_angle= robot.get_joint_angle,
                       get_joint_velocity= robot.get_joint_velocity)
"""
task = templateTask(sim= simulator, 
                    body_id= robot.sim._bodies_idx[robot.body_name],
                    get_ee_position=robot.get_ee_position, 
                    get_ee_velocity=robot.get_ee_velocity,
                    get_ee_orientation= robot.get_ee_orientation,
                    get_joint_angle= robot.get_joint_angle,
                    get_joint_velocity = robot.get_joint_velocity,
                    inverse_kinematics = robot.inverse_kinematics,
                    set_joint_angles= robot.set_joint_angles)

simulator.physics_client.resetDebugVisualizerCamera(cameraDistance= 1.5,            # Distance to the target position
                                              cameraYaw= 45,                        # adjust viewing angle
                                              cameraPitch= -50,                     # adjust tilt
                                              cameraTargetPosition=[0, 0, 0.5])     # Focus point

task.reset()
"""
print(f"Task observation: {task.get_obs()}")
print(f"EE Position Goal: {task.get_achieved_goal()}")
print(f"Success: {task.is_success(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")
print(f"Reward: {task.compute_reward(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")
"""
time.sleep(2000)