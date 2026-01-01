from panda_gym.envs.core import Task

import numpy as np
import random
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R



class pandaLTask(Task):
    """Goal of the robot is to go from start to end and also pass through all the intermediate points.
    The Agent should recieve a reward for the following cases:
        1. Reaching intermediate points.
        2. Reaching goal location.
        3. Staying at goal location until the end of episode.
        4. Staying withing the defined bounds while doing all the traversing.
    """
    def __init__(self, 
                 sim, 
                 get_ee_position,
                 get_ee_velocity,
                 distanceThreshold= 0.05) -> None:
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
        
        """
        goal_range = 1.0
        self.goal_range_low = np.array([-goal_range/2, -goal_range/2, 0])
        self.goal_range_high = np.array([goal_range/2, goal_range/2, goal_range])
        """
        
        # Selecting goals from an area above the table
        self.goal_range_low = np.array([0.30, -0.35, 0])
        self.goal_range_high = np.array([0.85, 0.35, 1.0])


        self.max_steps = 200
        self.episodic_steps = 0

        
        # stores the start and end points
        self.start = np.zeros(3)        # placeholder for the start point
        self.goal  = np.zeros(3)        # placeholder for the goal point

        # For scene creation
        self.numPoints = 4              # Total number of intermediate points (including the start and goal)
        self.targetPoints = []          # Stores the coordinates of all the points
        self.target_spheres = []        # Stores the ids of the target spheres
        self.containers = []            # Stores the ids of the containers
        self.box_center = []            # Stores the centers of the containers
        self.box_hE = []                # Stores the half Extents of the containers
        self.box_ori = []               # Stores the orientations of the containers

        # Stores how many intermediate points the EE has crossed
        self.flags = np.zeros(self.numPoints) 

        # Create the scene
        with self.sim.no_rendering():
            self._create_scene()



    def reset(self) -> None:
        """
        Reset the task:
            1. Set number of steps taken in current episode to be 0
            2. Set the current position of EE as the start position
            3. Sample a new goal position.
            4. Move the start and goal spheres to new location
            5. Remove old container and set the new one
        """
        # Reset total steps taken for the current episode
        self.episodic_steps = 0
        
        # Reset start location
        self.start = np.array(self.get_ee_position())
        self.sim.set_base_pose("start", self.start, np.array([0.0, 0.0, 0.0, 1.0]))
        
        # Reset intermediate points and flags
        self.targetPoints = []
        self.flags = np.zeros(self.numPoints)
        
        # Remove existing containers and spheres
        if self.containers != None:
            for id in self.containers:
                self.sim.physics_client.removeBody(id)
        
        if self.target_spheres != None:
            for id in self.target_spheres:
                self.sim.physics_client.removeBody(id)

        # Setup intermediate points and new containers
        self.target_spheres = []        
        self.containers = []            
        self.box_center = []            
        self.box_hE = []                
        self.box_ori = []               
        self.containers, self.target_spheres = self.setupContainers(self.start, self.numPoints)
        
        # Assign new goal location
        self.goal_id = self.target_spheres[-1]
        self.goal = self.targetPoints[-1]
        


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
        self.start = self.sim.get_base_position("start")


        # Body 4: the Container & Target spheres
        self.containers, self.target_spheres = self.setupContainers(self.start, self.numPoints)
        
        self.goal_id = self.target_spheres[-1]
        self.goal = self.targetPoints[-1]
        


    def setupContainers(self, start, numPoints) -> np.ndarray:
        """
        Sample new points using the _sample_target() function. Set up a container between the current and new point. Additionally, create a 
        visual sphere at the new point.

        Params:
        start(np.ndarray)       --- Start point of the task (Ideally the location of the EE when the episode begins)
        numPoints(np.float32)   --- Total number of points we have to sample

        Returns:
        containers(np.ndarray)  --- Array containing the ids of the containers
        target_spheres(np.ndarray)--- Array containing the ids of the target_spheres   
        """
    
        self.targetPoints.append(start)
        containers = [] 
        target_spheres = []

        for i in range(numPoints-1):
            
            start = self.targetPoints[i]
            target = self._sample_target()
            
            # Add target to the list of points we want to traverse through
            self.targetPoints.append(target)

            # Container Center
            container_center = (np.array(target) + np.array(start))/2
            self.box_center.append(container_center)

            # HalfExtents                 
            hE_x = np.linalg.norm(target-start)/2
            hE_y = 0.05                    
            hE_z = 0.02                                       
            halfExtents = np.array([hE_x, hE_y, hE_z]) 
            self.box_hE.append(halfExtents)

            # Orientation
            orientation = self.get_orient(start, target)
            self.box_ori.append(orientation)
                
            # Setting up the container       
            visual_kwargs = {
                "halfExtents": halfExtents,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.1, 0.9, 0.5])
            }
            baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                            **visual_kwargs)
            baseCollisionShapeIndex = -1                                    # for ghost shapes CollisionShapeIndex = -1
            box = self.sim.physics_client.createMultiBody(
                baseMass = 0.0,
                baseCollisionShapeIndex = baseCollisionShapeIndex,
                baseVisualShapeIndex = baseVisualShapeIndex,
                basePosition = container_center,
                baseOrientation = orientation
            )
            containers.append(box)

            # At each of the target locations we set up a sphere
            visual_kwargs = {
                "radius": self.distanceThreshold,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.9, 0.1, 0.5]) }
            baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_SPHERE,
                                                                                    **visual_kwargs)
            baseCollisionShapeIndex = -1                                    # for ghost shapes CollisionShapeIndex = -1
            sphere = self.sim.physics_client.createMultiBody(
                        baseMass = 0.0,
                        baseCollisionShapeIndex = baseCollisionShapeIndex,
                        baseVisualShapeIndex = baseVisualShapeIndex,
                        basePosition = target)
            target_spheres.append(sphere)

        return containers, target_spheres
    
    

    def get_orient(self, start, target):
            """
            Gets the orientation of the container between start and target. Also ensures that there is no roll about x-axis when 
            we rotate the current X axis to its new orientation

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


    
    def check_flag(self) -> np.int64:
        """
        Check the self.flags list and returns the index where it finds the first 0

        Returns:
            The index of the flag that we are on
        """
        for i, val in enumerate(self.flags):
            if val == 0:
                return i
        # if all the values in the list are 1 (all point have been reached), send the last send the last index value so 
        # that the EE stays in the final location
        return (self.numPoints - 1) 



    # Abstract Function
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the task. We have 3 types of reward:
            1. Distance Reward: For reaching the target
            2. Velocity Reward: For slowing down when we are near the target
            3. Within Bounds Reward: For ensuring the EE stays within the container
            4. Success Reward: For reaching intermediate points
            5. Dwell Reward: For reaching and staying at the final goal location
        """

        flag_index = self.check_flag()
        desired_goal = self.targetPoints[flag_index]              # intermediate target set
        
        # Reward Weights:
        alpha = 1.0         # Distance Reward
        beta = 0.35         # Velocity Reward
        gamma = 0.45        # Within Bounds Reward
        delta = 2.0         # Success Reward

        # 1. Distance Reward: negative penalty for distance from goal
        dist = np.linalg.norm(achieved_goal - desired_goal)
        dist_reward = -dist

        # 2. Velocity Reward: negative penalty for high velocities
        ee_velocity = np.array(self.get_ee_velocity())
        vel_sqr = np.square(np.linalg.norm(ee_velocity))
        vel_reward = -vel_sqr

        # 3. Within Bounds Reward: negative penalty for not staying in the box
        within_reward = self.get_within_bounds_reward(achieved_goal= achieved_goal, 
                                           flag_index= flag_index)
        
        # 4. Success Reward: reward for reaching goal
        success_reward = self.get_success_reward(achieved_goal= achieved_goal,
                                               flag_index = flag_index)
        
        # 5. Dwell Reward: reward for staying at the final goal location
        dwell_reward = self.get_dwell_reward(achieved_goal= achieved_goal)
        
        """
        print(f"distance reward: {alpha * dist_reward}")
        print(f"velocity reward: {beta * vel_reward}")
        print(f"within reward: {gamma * within_reward}")
        print(f"target reward: {delta * success_reward}")
        print(f"flags array: {self.flags}")
        """
        reward = alpha * dist_reward + beta * vel_reward + gamma * within_reward + delta * success_reward + dwell_reward
        
        return reward



    def get_within_bounds_reward(self, achieved_goal: np.ndarray, flag_index: np.int64):
        """
        Checks whether the EE is within the bounds of the current container. First it checks which container are we considering and then if the
        EE is within the bounds of this container.

        Params: 
        achieved_goal(np.ndarray)   --- Current location of the EE
        flags(np.ndarray)           --- Array containing information about how many points have been reached. One flag value for each target
                                        location including the start and the end

        Returns:
        """ 
        if flag_index == 0:
            return 0
        else:
            # Step 1: Check which intermediate positions have been reached
            box_center = self.box_center[flag_index-1]
            box_hE = self.box_hE[flag_index-1]
            box_ori = self.box_ori[flag_index-1]

            # Step 2: Move to the local frame of the box. For that we take the center position and orientation of the box and calculate the inverse transform
            # We then apply these inverse transforms to out ee_position
            inv_pos, inv_orn = self.sim.physics_client.invertTransform(box_center, box_ori)
            local_pt, _= self.sim.physics_client.multiplyTransforms(
                    positionA= inv_pos,
                    orientationA= inv_orn,
                    positionB= achieved_goal,
                    orientationB= np.array([0.0, 0.0, 0.0, 1.0])
                    ) 

            # Step 3: Check if the EE position (in the local frame of the box) is within bounds        
            penalty = 0.0    
            for i in range(3):
                if local_pt[i] < -box_hE[i] or local_pt[i] > box_hE[i]:
                    penalty -= (np.abs(local_pt[i]) - box_hE[i]) + 0.5 * np.square(np.abs(local_pt[i]) - box_hE[i])
                    
            return penalty
        

    
    def get_success_reward(self, achieved_goal: np.ndarray, flag_index: np.int64):
        """
        Checks whether the EE has reached any of the intermediate locations and rewards based on that. Also updates the flag incase 
        one of the reward locations have been reached.

        Parameters: 
            achieved_goal(np.ndarray)   --- current EE location
            flag_index(int)             --- index in the target location array that we want to reach
        """
        reward = 0
        dist = np.linalg.norm(achieved_goal - self.targetPoints[flag_index])

        # When flag_index = 0, we are at the start location; directly change that flag to 1
        if flag_index == 0:
            self.flags[flag_index] = 1.0

        # For any other flag_index other than 0;
        # Check if that target hasn't been reached and if dist is less than the threshold
        elif self.flags[flag_index] == 0  and dist < self.distanceThreshold:
            reward += (flag_index + 1)
            self.flags[flag_index] = 1.0
            
        return reward



    def get_dwell_reward(self, achieved_goal: np.ndarray):
        """
        If the agent has reached all the target locations and is in the final goal location it returns extra reward for 
        staying there.
        
        Parameters:
            achieved_goal(np.ndarray)       --- Current location of the EE
        
        Returns:
            dwell_reward(np.float32)        --- Reward for staying in the final goal location
        """
        dwell_reward = 0.0
        dist = np.linalg.norm(achieved_goal - self.targetPoints[-1])

        if all(x == 1.0 for x in self.flags) and dist < self.distanceThreshold:
            dwell_reward += 1.0
        else:
            dwell_reward += 0.0
        
        return dwell_reward
            


    def get_achieved_goal(self) -> np.ndarray:
        """
        Returns the ee position of the robot.
        """
        ee_position = np.array(self.get_ee_position())
        return ee_position


    
    # Abstract Function
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Returns whether the achieved goal match the desired goal. Success tells us if the EE is within the distance threshold of 
        of the desired location.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        success = np.array(dist < self.distanceThreshold, dtype=bool)
        return success



    # Abstract Function
    def get_obs(self) -> np.ndarray:
        """
        Return the observation associated to the task. Returns the distance between the end effector and the goal location.
        """
        ee_position = self.get_achieved_goal()
        
        flag_index = self.check_flag()
        goal = self.targetPoints[flag_index]

        self.observation = goal - ee_position
        return self.observation
    


    def _sample_target(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        target = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return target
    



     