from panda_gym.pybullet import PyBullet
from pandaReachTask import pandaReachTask
from pandaRobot import pandaRobot



simulator = PyBullet(render_mode="human")
robot = pandaRobot(sim= simulator)
task = pandaReachTask(sim= simulator, get_ee_position= robot.get_ee_position)

task.reset()
print(f"Task observation: {task.get_obs()}")
print(f"EE Position Goal: {task.get_achieved_goal()}")
print(f"Success: {task.is_success(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")
print(f"Reward: {task.compute_reward(achieved_goal= task.get_achieved_goal(), desired_goal= task.get_goal())}")