from panda_gym.envs.core import Task

import numpy as np
import random
from typing import Any, Dict



class pandaMultiSizedBoxesTask(Task):
    """Task is to go from start to end always stay within the container. Whenever it leaves the container it experiences a penalty. 
    After each episode, the end position of the previous episode becomes the start position of the new one."""


    def __init__(self, 
                 sim, 
                 get_ee_position,
                 get_ee_velocity,
                 distanceThreshold= 0.05,
                 goal_range = 1.0) -> None:
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

        self.flag = False               # flag determines whether our EE has reached the start position or not.
        
        self.start = np.zeros(3)        # placeholder for the start point
        self.goal  = np.zeros(3)        # placeholder for the goal point
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
            position= self.start,
            rgba_color= np.array([0.9, 0.1, 0.1, 0.7]),
        )

        # Body 4 : the end point
        self.sim.create_sphere(                             # end position = [0.5, 0.5, 0.1]
            body_name="goal",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position= self.goal,
            rgba_color= np.array([0.1, 0.9, 0.1, 0.7]),
        )
        
        self.start = self.sim.get_base_position("start")
        self.goal = self.sim.get_base_position("goal")
        print(f"Start : {self.start}")
        print(f"Goal : {self.goal}")


        # Body 5: the Container
        self.container = self.setup_container(self.start, self.goal)
        """
        # Listing all the shapes:
        for i in range(self.sim.physics_client.getNumBodies()):              
            body_id = self.sim.physics_client.getBodyUniqueId(i)
            info = self.sim.physics_client.getBodyInfo(body_id)
            print("body_id : ", body_id)
            print("info : ", info)
        """

    
    def setup_container(self, start, goal):
        """
        Setup the new container that goes from the start point to the end point. 
        """
        # Center
        container_center = (np.array(goal) + np.array(start))/2


        # HalfExtents
        half_extent_x = np.linalg.norm(goal - start)/2
        half_extent_y, half_extent_z = self.sample_box_dim()
        halfExtents = np.array([half_extent_x, half_extent_y, half_extent_z]) 


        # Orientation
        direction = np.array(goal) - np.array(start)
        norm = np.linalg.norm(direction)
        v_ref = np.array([1.0, 0.0, 0.0]) # When RPY set to zero then we get quaternion = [ 0, 0, 0, 1] [ x, y, z, w]

        if norm == 0.0:
            v = np.array([0.0, 0.0, 0.0]) # Both start and end points are the same
            print("Warning: start and goal points are the same. Zero direction vector")
            return None
        else:
            v = direction/ norm
            cross = np.cross(v_ref, v)
            dot = np.dot(v_ref, v)

            theta = np.arccos(dot)
            axis = cross / np.linalg.norm(cross)

            w = np.array([np.cos(theta/2)])
            xyz = np.sin(theta/2) * axis
            orientation = np.concatenate((xyz, w))
            
            rgba_color = np.array([0.1, 0.1, 0.9, 0.5])                      # colour for the sphere
            basePosition = container_center
            baseOrientation = orientation
            visual_kwargs = {
                "halfExtents": halfExtents,
                "specularColor": None,
                "rgbaColor": rgba_color 
            }
            baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                            **visual_kwargs)
            baseCollisionShapeIndex = -1                                    # for ghost shapes CollisionShapeIndex = -1
            container = self.sim.physics_client.createMultiBody(
                baseMass = 0.0,
                baseCollisionShapeIndex = baseCollisionShapeIndex,
                baseVisualShapeIndex = baseVisualShapeIndex,
                basePosition = basePosition,
                baseOrientation = baseOrientation
            )
            return container
    

    def reset(self) -> None:
        """
        Reset the task:
            1. Set number of steps taken in current episode to be 0
            2. Set the current position of EE as the start position
            3. Sample a new goal position.
            4. Move the start and goal spheres to new location
            5. Remove old container and set the new one
        """
        self.episodic_steps = 0
        self.start = np.array(self.get_ee_position())
        self.goal = self._sample_goal()
        self.sim.set_base_pose("start", self.start, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("goal", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        
        if self.container != None:
            self.sim.physics_client.removeBody(self.container)
        self.container = self.setup_container(self.start, self.goal)

    
    def get_obs(self) -> np.ndarray:
        """
        Return the observation associated to the task. Returns the distance between the end effector and the goal location.
        """
        ee_position = self.get_achieved_goal()
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



    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the task. We have 3 types of reward:
            1. Distance Reward: For reaching the target
            2. Velocity Reward: For slowing down when we are near the target
            3. Within Bounds Reward: For ensuring the EE stays within the container
        """
        
        # 1. Distance Reward
        dist = np.linalg.norm(achieved_goal - desired_goal)
        dist_reward = -dist

        # 2. Velocity Reward
        ee_velocity = np.array(self.get_ee_velocity())
        vel_sqr = np.square(np.linalg.norm(ee_velocity))
        vel_reward = -0.1 * vel_sqr

        # 3. Within Bounds Reward
        if not self.within_bounds(achieved_goal):
            within_reward = -0.5
        else:
            within_reward = 0.0
        
        #print(f"Dist Reward: {dist_reward}, Velocity Reward: {vel_reward}, Within Bounds Reward: {within_reward}")
        reward = dist_reward + vel_reward + within_reward
        
        return reward



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
    


    def _sample_goal(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return goal
    


    def sample_box_dim(self) -> np.ndarray:
        """
        Sample a dimensions for the container within which the end effector should stay
        
        Return:
        dims (np.ndarray) --- dimension values
        """
        low = [0.02, 0.02]
        high = [0.1, 0.1]
        dims = np.random.uniform(low, high)
        return dims


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
