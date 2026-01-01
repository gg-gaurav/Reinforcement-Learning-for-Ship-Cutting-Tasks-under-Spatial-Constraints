from pandaFollowRobot import pandaFollowRobot
from pandaFollowTask import pandaFollowTask
from pandaFollowTaskEnv import pandaFollowTaskEnv

from panda_gym.pybullet import PyBullet
import gymnasium as gym
import numpy as np
import time


env = pandaFollowTaskEnv(render_mode= "human")
env.reset()

episodes = 50

for i in range(1, episodes+1):

    observation, _ = env.reset()
    env.sim.create_sphere(
            body_name="target",
            radius=0.1,
            mass=0.0,
            ghost=True,
            position= observation["desired_goal"],
            rgba_color= np.array([0.1, 0.9, 0.1, 0.7]),
        )
    """
    for i in range(env.sim.physics_client.getNumBodies()):              
        body_id = env.sim.physics_client.getBodyUniqueId(i)
        info = env.sim.physics_client.getBodyInfo(body_id)

        print("body_id : ", body_id)
        print("info : ", info)

        
    Body Id for the sphere… when one “target sphere” (the green one) is removed from the the simulation environment 
    and a new one is generated, the bodyID of the new sphere is always 1 and that of the robot is 0.
    """
    done = False
    

    for j in range(200):
        action = env.action_space.sample()                      # random action

        print("action : ", action)
        observation, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.07)
        print("reward: ", reward)
        
        if terminated or truncated:
            done = True
            break

    #env.sim.physics_client.removeBody(i)

env.close()