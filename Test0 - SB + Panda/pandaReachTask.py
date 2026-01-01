from panda_gym.envs.core import Task

import numpy as np
import random
from typing import Any, Dict



class pandaReachTask(Task):
    """Task is for the pendulum to reach the desired goal location"""


    def __init__(self, 
                 sim, 
                 get_ee_position,
                 get_ee_velocity,
                 distanceThreshold= 0.1,
                 goal_range = 0.75) -> None:
        """
        Parameters:
            sim                 ---> simulator
            get_ee_position     ---> function that returns the end effector position
            get_ee_velocity     ---> function that returns the end effector velocity
            distanceThreshold   ---> distance needed to consider that the task is complete
            goal_range          ---> range from which we sample the goal location
        """
        super().__init__(sim)
        self.get_ee_position = get_ee_position
        
        self.get_ee_velocity = get_ee_velocity

        self.distanceThreshold = distanceThreshold
        # self.goal_range = goal_range
        self.goal_range_low = np.array([-goal_range/2, -goal_range/2, 0])
        self.goal_range_high = np.array([goal_range/2, goal_range/2, goal_range])
        self.max_steps = 200
        self.episodic_steps = 0
        with self.sim.no_rendering():
            self._create_scene()
        

        
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.1,
            mass=0.0,
            ghost=True,
            position= np.zeros(3),
            rgba_color= np.array([0.1, 0.9, 0.1, 0.7]),
        )



    def reset(self) -> None:
        """
        Reset the task: sample a new goal.
        """
        self.episodic_steps = 0
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


    
    def get_obs(self) -> np.ndarray:
        """
        Return the observation associated to the task. Returns the distance between the end effector and the goal location.
        """
        ee_position = self.get_achieved_goal()
        goal = self.goal
        self.observation = goal - ee_position
        #self.observation = np.array([])
        return self.observation


    
    def get_achieved_goal(self) -> np.ndarray:
        """
        Returns the ee position of the robot.
        """
        ee_position = np.array(self.get_ee_position())
        return ee_position



    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Returns whether the achieved goal match the desired goal. Success tells us if the EE is within the distance threshold of 
        of the desired location.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        success = np.array(dist < self.distanceThreshold, dtype=bool)
        return success

    

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the achieved and the desired goal.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        dist_reward = -np.square(dist)
        # dist_reward = -dist

        success = self.is_success(achieved_goal, desired_goal)
        if success:
           success_reward = 0.0
        else:
           success_reward = -0.1

        ee_velocity = np.array(self.get_ee_velocity())
        vel_sqr = np.square(np.linalg.norm(ee_velocity))
        vel_reward = -0.1 * vel_sqr

        # print(f"distance reward: {dist_reward}")
        # print(f"velocity reward: {vel_reward}")
        # print(f"Success reward: {success_reward}")
        reward = dist_reward + vel_reward + success_reward 

        return reward



    def _sample_goal(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        
        return goal
    


