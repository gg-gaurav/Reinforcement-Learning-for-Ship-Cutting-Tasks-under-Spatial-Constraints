from panda_gym.envs.core import Task

import numpy as np
import random
from typing import Any, Dict



class pandaFollowTask(Task):
    """Task is for the pendulum to reach the desired goal location"""


    def __init__(self, 
                 sim, 
                 get_ee_position,
                 get_ee_velocity,
                 distanceThreshold= 0.05,
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
        
        self.goal_range_low = np.array([-goal_range/2, -goal_range/2, 0])
        self.goal_range_high = np.array([goal_range/2, goal_range/2, goal_range])
        self.max_steps = 200
        self.episodic_steps = 0

        self.goal = None

        self.flag = False               # flag determines whether our EE has reached the start position or not.
        self.start = np.zeros(3)        # placeholder for the start point
        self.end = np.zeros(3)          # placeholder for the end point
        self.container = None           # placeholder for the container
        self.prev_dist = None           # placeholder for the distance of the EE from goal in the previous timestep

        # Parameters for Follow Reward:
        self.alpha = -0.3
        self.D = 18.0
        self.k = -0.2

        with self.sim.no_rendering():
            self._create_scene()
        

        
    def _create_scene(self) -> None:
        """
        Creates the scene for the experiment
        """
        # Body 0 : the panda robot
        # Body 1 : the surface
        self.sim.create_plane(z_offset=-0.4)                # surface/ floor at z = -0.4
        
        # Body 2 : the table
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3) # table with table top at z = 0
        
        # Body 3 : the start point
        self.sim.create_sphere(                             # start positon = [0.1, 0.1, 0.1]
            body_name="start",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position= np.array([0.2, 0.2, 0.1]),
            rgba_color= np.array([0.9, 0.1, 0.1, 0.7]),
        )

        # Body 4 : the end point
        self.sim.create_sphere(                             # end position = [0.5, 0.5, 0.1]
            body_name="end",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position= np.array([0.6, 0.2, 0.1]),
            rgba_color= np.array([0.1, 0.9, 0.1, 0.7]),
        )
        
        self.start = self.sim.get_base_position("start")
        self.end = self.sim.get_base_position("end")
        print(f"Start : {self.start}")
        print(f"End : {self.end}")


        # Body 5: the Container
        center_x = (self.start[0] + self.end[0])/2
        center_y = (self.start[1] + self.end[1])/2
        center_z = 0.1
        print(f"Container center: {center_x, center_y, center_z}")

        half_extent_x = np.abs(self.start[0]-center_x)
        half_extent_y = 0.05
        half_extent_z = 0.05
        print(f"Container half extents: {half_extent_x, half_extent_y, half_extent_z}")

        rgba_color = np.array([0.1, 0.1, 0.9, 0.3])                         # colour for the sphere
        halfExtents = np.array([half_extent_x, half_extent_y, half_extent_z]) 
        basePosition = np.array([center_x, center_y, center_z])
        baseOrientation = self.sim.physics_client.getQuaternionFromEuler([1.0 ,0.0, 0.0])
        print(baseOrientation)
        visual_kwargs = {
            "halfExtents": halfExtents,
            "specularColor": None,
            "rgbaColor": rgba_color 
        }
        baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                         **visual_kwargs)
        baseCollisionShapeIndex = -1                                        # for ghost shapes CollisionShapeIndex = -1
        self.container = self.sim.physics_client.createMultiBody(
            baseMass = 0.0,
            baseCollisionShapeIndex = baseCollisionShapeIndex,
            baseVisualShapeIndex = baseVisualShapeIndex,
            basePosition = basePosition,
            baseOrientation = baseOrientation
        )

        # Listing all the shapes:
        for i in range(self.sim.physics_client.getNumBodies()):              
            body_id = self.sim.physics_client.getBodyUniqueId(i)
            info = self.sim.physics_client.getBodyInfo(body_id)
            print("body_id : ", body_id)
            print("info : ", info)



    def reset(self) -> None:
        """
        Reset the task: sample a new goal.
        """
        self.episodic_steps = 0
        self.flag = False
        self.goal = self.start
        #self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


    
    def get_obs(self) -> np.ndarray:
        """
        Return the observation associated to the task. Returns the distance between the end effector and the goal location.
        """
        ee_position = self.get_achieved_goal()

        if not self.flag:
            self.goal = self.start
        else:
            self.goal = self.end
        
        self.observation = self.goal - ee_position
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

    
    def reached_start(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}):
        """
        Used to check the EE has reached the start location; if yes then we update the flag and swtich the reward to path reward
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        success = np.array(dist < self.distanceThreshold, dtype=bool)
        if success:
            self.flag = True
        


    def within_bounds(self, achieved_goal: np.ndarray, info: Dict[str, Any]= {}):
        """
        Used to check whether we are withing the boundaries of the container or not. We create a small sphere at the EE location and
        """

        # Generating a small sphere at the location of the end effector
        ee_position = achieved_goal                                         # current position of the EE
        
        radius = 1e-6                                                       # radius of EE sphere
        rgba_color = np.array([0.1, 0.9, 0.1, 0.7])                         # colour for the sphere
        visual_kwargs = {
            "radius": radius,
            "specularColor": None,
            "rgbaColor": rgba_color 
        }
        baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_SPHERE,
                                                                         **visual_kwargs)
        baseCollisionShapeIndex = -1                                        # for ghost shapes CollisionShapeIndex = -1
        ee = self.sim.physics_client.createMultiBody(
            baseMass = 0.0,
            baseCollisionShapeIndex = baseCollisionShapeIndex,
            baseVisualShapeIndex = baseVisualShapeIndex,
            basePosition = ee_position
        )

        # Checking if the generated sphere is within the container
        points = self.sim.physics_client.getClosestPoints(bodyA=self.container, bodyB = ee, distance=0.0)
        """print(f"Closest Points: {points}")"""

        return len(points) > 0

        

    def point_to_line_distance(self, point, line_start, line_end):
        """
        Gives us the distance between a point and line made by joining 2 other points in 3D
        Args:
            point (np.ndarray): The point P as [x, y, z]
            line_start (np.ndarray): Point A on the line [x1, y1, z1]
            line_end (np.ndarray): Point B on the line [x2, y2, z2]
        """
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)

        ab = line_end - line_start
        ap = point - line_start

        cross_prod = np.cross(ab, ap)
        distance = np.linalg.norm(cross_prod) / np.linalg.norm(ab)

        return distance



    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the achieved and the desired goal. Each episode will have 2 phases:
            Phase 1: Make end effector reach the start point. For this task the desired goal is the start location of the path following taks.
            Phase 2: Make end effector follow the path and make it go from the start point to the end along a particular path.
        self.flag determines which phase we are in:
            Phase 1: self.flag = False
            Phase 2: self.flag = True
        """
        
        # PHASE 1: Until the EE reaches the start location
        if not self.flag:
            """print(f"Reward : Phase 1")"""
            goal = self.start
            
            # Distance reward
            distToStart = np.linalg.norm(achieved_goal - goal)
            dist_reward = -distToStart
        
            # Velocity Reward
            ee_velocity = np.array(self.get_ee_velocity())
            vel_sqr = np.square(np.linalg.norm(ee_velocity))
            vel_reward = -0.1 * vel_sqr
            
            # Total Reward
            reach_reward = dist_reward + vel_reward 
            reward = reach_reward

            # Checks if EE has reached the start position or not
            self.reached_start(achieved_goal= achieved_goal, desired_goal= goal)
            
        # PHASE 2: Once EE has reached the start location
        else:
            """print("Reward : Phase 2")"""
            goal = self.end

            """print(f"Is ee withing container: {self.within_bounds(achieved_goal)}")"""
            distToEnd = np.linalg.norm(achieved_goal - goal)
            W = self.point_to_line_distance(achieved_goal, self.start, self.end)
            
            # Reached Target Reward
            if distToEnd < self.distanceThreshold:
                reward = 1.0

            # Within Boundary Reward
            elif not self.within_bounds(achieved_goal):
                reward = -1.0
            
            # Moving away or close to target:
            else: 
                
                # Moving away from Target Reward
                if distToEnd > self.prev_dist: 
                    reward = -(distToEnd/ self.D) + self.k + self.alpha * W
                
                # Moving close to Target Reward
                else: 
                    reward = -(distToEnd/ self.D) + self.alpha * W
        
            # Path Reward
            self.prev_dist = distToEnd
            reward = np.float32(reward)
        

        return reward



    def _sample_goal(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        
        return goal
    


