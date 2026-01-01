from panda_gym.envs.core import Task

import numpy as np
import random
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R



class pandaFollowTask(Task):
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
        
        self.start = np.zeros(3)        # placeholder for the start point
        self.goal  = np.zeros(3)        # placeholder for the goal point
        self.container = None           # placeholder for the container
        self.box_pos = np.zeros(3)      # default box position
        self.box_hE = np.zeros(3)       # default box halfextents 
        self.box_ori = np.array([0.0, 0.0, 0.0, 1.0]) # default box orientation


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
        half_extent_y = 0.1
        half_extent_z = 0.05
        halfExtents = np.array([half_extent_x, half_extent_y, half_extent_z]) 


        # Orientation
        orientation = self.get_orient(start, goal)
        
        # Setting up the container
        visual_kwargs = {
            "halfExtents": halfExtents,
            "specularColor": None,
            "rgbaColor": np.array([0.1, 0.1, 0.9, 0.5]) 
        }
        baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                        **visual_kwargs)
        baseCollisionShapeIndex = -1                                    # for ghost shapes CollisionShapeIndex = -1
        container = self.sim.physics_client.createMultiBody(
            baseMass = 0.0,
            baseCollisionShapeIndex = baseCollisionShapeIndex,
            baseVisualShapeIndex = baseVisualShapeIndex,
            basePosition = container_center,
            baseOrientation = orientation
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
            4. Success Reward: For reaching the goal location
        """

        # Parameters for Reward:
        alpha = 1.0
        beta = 0.35
        gamma = 0.25
        delta = 1.0
        
        # 1. Distance Reward: negative penalty for distance from goal
        dist = np.linalg.norm(achieved_goal - desired_goal)
        dist_reward = -dist

        # 2. Velocity Reward: negative penalty for high velocities
        ee_velocity = np.array(self.get_ee_velocity())
        vel_sqr = np.square(np.linalg.norm(ee_velocity))
        vel_reward = -vel_sqr

        # 3. Within Bounds Reward: negative penalty for not staying in the box
        within_reward = self.within_bounds(point= achieved_goal, 
                        box_position= self.box_pos, 
                        box_orientation= self.box_ori,
                        halfExtents= self.box_hE)

        # 4. Success Reward: reward for reaching goal
        success_reward= 1.0 if dist < self.distanceThreshold else 0.0

        """
        print(f"distance reward: {self.alpha*dist_reward}")
        print(f"velocity reward: {self.beta*vel_reward}")
        print(f"within reward: {self.gamma*within_reward}")
        """
        reward = alpha * dist_reward + beta * vel_reward + gamma * within_reward + delta* success_reward
        
        return reward



    def within_bounds(self, point, box_position, box_orientation, halfExtents) -> np.float32:
        """
        Used to check whether we are withing the boundaries of the container or not. 
        """
        # Step 1: Move to the local frame of the box. For that we take the center position and orientation of the box and calculate the inverse transform
        # We then apply these inverse transforms to out ee_position
        inv_pos, inv_orn = self.sim.physics_client.invertTransform(box_position, box_orientation)
        local_pt, _= self.sim.physics_client.multiplyTransforms(
                positionA= inv_pos,
                orientationA= inv_orn,
                positionB= point,
                orientationB= np.array([0.0, 0.0, 0.0, 1.0])
                ) 

        # Step 2: Check if the EE position (in the local frame of the box) is within bounds        
        penalty = 0.0     
        
        for i in range(3):
            if local_pt[i] < -halfExtents[i] or local_pt[i] > halfExtents[i]:
                penalty -= (np.abs(local_pt[i])-halfExtents[i]) + 0.5* np.square(np.abs(local_pt[i])-halfExtents[i])
                
        return penalty
    


    def _sample_goal(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        
        return goal
    


    def get_orient(self, start, target):
            """
            Gets the orientation of the container between start and target. Also ensures that there is no roll about x-axis when we rotate 
            the current X axis to its new orientation

            Returns:
            quat_or(np.ndarray) --- orientation quaternion for the container
            z_axis(np.ndarry) --- z_axis that is equivalent to the surface normal
            """
            # New x-axis
            direction= np.array(target)-np.array(start)
            norm= np.linalg.norm(direction)

            if norm < 1e-6:
                print(f"Start and end points are too close to each other")
                return None
            
            x_axis= direction/norm

            # Global Up direction
            up = np.array([0.0, 0.0, 1.0])

            y_axis = np.cross(up, x_axis)
            if np.linalg.norm(y_axis) < 1e-6:
                # Edge Case: direction is straight up/ down
                up = np.array([0.0, 1.0, 0.0])
                y_axis = np.cross(up, x_axis)

            y_axis = y_axis/ np.linalg.norm(y_axis)

            # Compute z_axis 
            z_axis = np.cross(x_axis, y_axis)

            # Build rotation matrix from X, Y, Z
            rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

            # get quaternion
            rot = R.from_matrix(rot_matrix)
            quat_orient = rot.as_quat()
            return quat_orient



