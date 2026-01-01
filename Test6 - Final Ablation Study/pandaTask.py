from panda_gym.envs.core import Task

import random
import numpy as np
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R
from collections import deque




    


class pandaTask(Task):
    """Goal of the robot is to go from start to end and also pass through all the intermediate points. As the End effector is going from
    one point to another, it should stay within bounds of the container and also it should have an orientation perpendicular to base of the 
    container.
    The Agent recieves a reward for the following cases:
        1. Reaching intermediate points.
        2. Reaching goal location.
        3. Staying at goal location until the end of episode.
        4. Staying withing the defined bounds while doing all the traversing.
        5. Maintaining orientation while traversing"""


    def __init__(self, 
                 sim,
                 body_id, 
                 get_ee_position,
                 get_ee_velocity,
                 get_ee_orientation,
                 get_joint_angle,
                 get_joint_velocity,
                 set_joint_angles,
                 distanceThreshold= 0.05) -> None:
        """
        Parameters:
            sim                 ---> simulator
            body_id             ---> robot id
            get_ee_position     ---> function that returns the end effector position
            get_ee_velocity     ---> function that returns the end effector velocity
            get_ee_orientation  ---> function that returns the end effector orientation
            get_joint_angle     ---> function that returns the joint angles of the robot
            get_joint_velocity  ---> function that returns the joint velocities of the robot
            set_joint_angles    ---> function that sets the joint angle to a particular value
            distanceThreshold   ---> distance needed to consider that the task is complete
        """
        super().__init__(sim)
        self.body_id = body_id
        self.get_ee_position = get_ee_position
        self.get_ee_velocity = get_ee_velocity
        self.get_ee_orientation = get_ee_orientation
        self.get_joint_angle = get_joint_angle
        self.get_joint_velocity = get_joint_velocity
        self.set_joint_angles = set_joint_angles
        self.distanceThreshold = distanceThreshold

        # home position for all controllable links
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11                                                            
        
        # Selecting goals from an area above the table
        self.goal_range_low = np.array([0.25, -0.30, 0])           #np.array([0.30, -0.35, 0])
        self.goal_range_high = np.array([0.80, 0.30, 0.80])        #np.array([0.85, 0.35, 0.85])
        
        self.max_steps = 200
        self.episodic_steps = 0

        # stores the start and end points
        self.start = None               # placeholder for the start point
        self.goal  = None               # placeholder for the goal point

        self.numPoints = 6              # Total number intermediate points the EE needs to traverse (includes the start and end points)
        self.targetPoints = []          # Stores the coordinates of the points
        self.target_spheres = []        # Stores the ids for all the target_spheres
        self.containers = []            # Stores the ids for all the containers
        self.box_center = []            # Stores the centers of the containers
        self.box_hE = []                # Stores the half Extents of the containers
        self.box_ori = []               # Stores the orientations of the containers
        self.surface_normals = []       # Stores the quaternions for surface normals for each of the containers
        self.viz_normals = []           # if we visualize the surface normals then it stores the object ids

        self.flags = np.zeros(self.numPoints - 1)       # Stores how many intermediate points the EE has crossed. n-1 points if we remove the start point
        self.orientation_window = deque(maxlen= 10)     # Stores the orientation value over last 10 timesteps
        self.velocity_window = deque(maxlen= 10)         # Stores velocity value over last 10 timesteps

        ## REWARD STORAGE:
        self.waypointR = 0.0
        self.dwellR = 0.0
        self.velocityR = 0.0
        self.boundsR = 0.0
        self.orientationR = 0.0
        
        
        
        with self.sim.no_rendering():
            self._create_scene()

            
    # Abstract Method
    def reset(self) -> None:
        """
        Reset the task:
            1. Set number of steps taken in current episode to be 0
            2. Set the current position of EE as the start position. Move the start sphere to the current EE position
            3. Delete previous containers if they exist
            4. Delete previous target sphers
            5. Create new intermediate points 
            6. Set up new containers and target spherers
            7. Set final goal position and orientation

        """
        # Reset total steps taken for the current episode
        self.episodic_steps = 0
        
        # Reset start location:
        # if robot is stuck in a singularity go to the neutral location
        manipulability = self.get_singularity_penalty()
        if manipulability < 5e-2:
            self.set_joint_angles(self.neutral_joint_values)

        self.start = np.array(self.get_ee_position())
        self.sim.set_base_pose("start", self.start, np.array([0.0, 0.0, 0.0, 1.0]))

        # Reset intermediate points and flags
        self.targetPoints = []
        self.flags = np.zeros(self.numPoints - 1)

        # Remove existing containers and spheres
        if self.containers != None:
            for id in self.containers:
                self.sim.physics_client.removeBody(id)

        if self.target_spheres != None:
            for id in self.target_spheres:
                self.sim.physics_client.removeBody(id)

        if self.viz_normals != None:
            for id in self.viz_normals:
                self.sim.physics_client.removeBody(id)

        # Setup intermediate points and new containers
        self.target_spheres = []        
        self.containers = []            
        self.box_center = []            
        self.box_hE = []                
        self.box_ori = []      
        self.surface_normals = []
        self.viz_normals = []         
        self.containers, self.surface_normals, self.target_spheres = self.setupContainers(self.start, self.numPoints)

        # Store goal location and goal_id separately
        self.goal_id = self.target_spheres[-1]
        self.goal = self.targetPoints[-1]

        ## REWARD STORAGE:
        self.waypointR = 0.0
        self.dwellR = 0.0
        self.velocityR = 0.0
        self.boundsR = 0.0
        self.orientationR = 0.0

        

        """
        print(f"Target Points: {self.targetPoints}")
        print(f"Start Point: {self.start}")
        print(f"Surface Normals: {self.surface_normals}")
        """



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
            radius= self.distanceThreshold,
            mass=0.0,
            ghost=True,
            position= self.start,
            rgba_color= np.array([0.9, 0.1, 0.1, 0.7]),
        )
        self.start = self.sim.get_base_position("start")


        # Body 4: the Container & Target spheres
        self.containers, self.surface_normals, self.target_spheres = self.setupContainers(self.start, self.numPoints)

        # Body 5: the end point
        self.goal_id = self.target_spheres[-1]
        self.goal = self.targetPoints[-1]
        


    def setupContainers(self, 
                        start, 
                        numPoints):
        """
        Sample new points using _sample_target() function. Set up a container between the current point and the next. Additionally,
        create a visual sphere at the new point.

        Parameters:
        start(np.ndarray)       --- Start point of the task (Ideally the location of the EE when the episode begins)
        numPoints(np.ndarry)    --- Total number of points we have to sample

        Returns:
        containers(np.ndarray)      --- Array containing the ids of the containers
        surface_normals(np.ndarray) --- Array containing the surface normals for each of the containers
        target_spheres(np.ndarray)  --- Array containing the ids of the target_spheres
        """
        self.targetPoints.append(start)             # add start to index = 0 of the list of targetPoints
    
        containers = []                             # stores container IDs
        target_spheres = []                         # stores target sphere IDs
        surface_normals = []                        # stores surface normals
        
        for i in range(numPoints-1):               
            
            start = self.targetPoints[i]
            target = self._sample_target()          # sample the next location

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
            orientation, z_axis = self.get_container_orient(start, target)
            self.box_ori.append(orientation)
                          
            # Setting Up the Container
            visual_kwargs = {
                "halfExtents": halfExtents,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.1, 0.9, 0.7]) 
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


            # Surface Normal Calculation 
            surface_normal = self.get_surface_normal(z_axis)
            surface_normals.append(surface_normal)

            
            # Use cylinders for visualizing the surface normals 
            # Dimensions
            visual_kwargs = {
                "radius": 0.01,
                "length": 0.05,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.9, 0.1, 0.9])  
            }
            baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType = self.sim.physics_client.GEOM_CYLINDER,
                                                                            **visual_kwargs)
            baseCollisionShapeIndex = -1  
            cylinder = self.sim.physics_client.createMultiBody(
                baseMass = 0.0,
                baseCollisionShapeIndex = baseCollisionShapeIndex,
                baseVisualShapeIndex = baseVisualShapeIndex,
                basePosition = container_center,
                baseOrientation = surface_normal
            )
            self.viz_normals.append(cylinder)
            
            
            # At each of the target locations we set up a sphere
            visual_kwargs = {
                "radius": self.distanceThreshold,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.9, 0.1, 0.5]) }
            baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_SPHERE,
                                                                            **visual_kwargs)
            baseCollisionShapeIndex = -1                                    
            sphere = self.sim.physics_client.createMultiBody(
                        baseMass = 0.0,
                        baseCollisionShapeIndex = baseCollisionShapeIndex,
                        baseVisualShapeIndex = baseVisualShapeIndex,
                        basePosition = target)
            target_spheres.append(sphere)
        
        return containers, surface_normals, target_spheres

    

    def get_surface_normal(self, 
                           z_axis):
        """
        Returns the surface norm for the container. Note: The normal that it returns points inwards rather than outwards

        Parameters:
            z_axis(np.ndarry)       --- z_axis of the container
        Returns:
            orient_norm(np.ndarray) --- orientation quaternion for the surface normal pointing inwards
        """
        z_axis = -z_axis/ np.linalg.norm(z_axis)    # -ve z_axis because we want it to point inwards
        z_axis_ref = np.array([0.0, 0.0, 1.0])

        dot = np.dot(z_axis_ref, z_axis)            # comparing reference z_axis with z_axis of container
        orient_norm = None  

        if np.allclose(dot, 1.0):                   # np.allclose() ---> returns true if dot and 1.0 are element-wise within a tolerance range (default = 1e-8)
            orient_norm =  R.identity().as_quat()   # when z_axis of container is same as that of the reference z_axis --->
        
        elif np.allclose(dot, -1.0):
            # 180° rotation around X or Y (any perpendicular axis)
            orient_norm =  R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()
        else:
            axis = np.cross(z_axis_ref, z_axis)
            axis = axis / np.linalg.norm(axis)
            theta = np.arccos(dot)

            orient_norm = R.from_rotvec(theta * axis).as_quat()
        return orient_norm



    def get_container_orient(self, 
                             start, 
                             target):
        """
        Gets the orientation of the container between start and target. Also ensures that there is no roll about x-axis when
        we rotate the current X axis to its new orientation

        Returns:
        quat_or(np.ndarray) --- orientation quaternion for the container
        z_axis(np.ndarry)   --- z_axis that is equivalent to the surface normal
        """
        # Compute the direction of the new x-axis
        direction = np.array(target) - np.array(start)
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            print(f"Start and target are too close to each other")
            return None
        
        x_axis = direction/norm                 # container's new X axis

        # Selecting the global "up" direction
        up = np.array([ 0.0, 0.0, 1.0])

        # Compute y_axis = (up) X (x_axis)
        y_axis = np.cross( up, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
           
            print(f"Edge Case: direction is straight up/ down")
            
            if target[2] > start[2]:
                print(f"Target above start") 
                up = np.array([0.0, 1.0, 0.0])
            else:
                print(f"Target below start")
                up = np.array([0.0, -1.0, 0.0])
            y_axis = np.cross(up, x_axis)

        y_axis = y_axis/ np.linalg.norm(y_axis)

        # Compute z_axis 
        z_axis = np.cross(x_axis, y_axis)

        # Build rotation matrix from X, Y, Z
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # get quaternion
        rot = R.from_matrix(rot_matrix)
        quat_orient = rot.as_quat()
        return quat_orient, z_axis
    


    def check_flag(self) -> np.int64:
        """
        Check the self.flags list and returns the index where it finds the first 0. Doesn't include the start point so the indexes follow
        numpoint = [start, 0, 1, 2]
        flags = [0, 1, 2]
        Returns the index wherever it finds the first 0. Else if all the values in self.flags is 1.0 it returns -1.

        Returns:
            The index of the flag that we are on
        """
        #self.flags = np.array([1.0, 1.0, 1.0])
        for i, val in enumerate(self.flags):
            if val == 0.0:
                return i
        return None             # All flags complete 



    # Abstract Method
    def compute_reward(self, 
                       achieved_goal: np.ndarray, 
                       desired_goal: np.ndarray, 
                       info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the task. We have 3 types of reward:
            1. Waypoint Reward: For reaching the intermediate and goal locations.
            2. Dwell Reward: For staying at the final goal location.
        """
        flag_index = self.check_flag()
        
        if flag_index is None:
            # All points have been reached so target location is the final point where we dwell
            flag_index = len(self.flags) - 1

        desired_goal = self.targetPoints[flag_index + 1]              # intermediate target set
        

        # 1. Waypoint Reward:
        waypoint_reward = self.get_waypoint_reward(flag_index= flag_index,
                                                achieved_goal= achieved_goal, 
                                                desired_goal= desired_goal,
                                                reward_type= "sparse",
                                                reward_shaping= True)
        #print(f"Waypoint Reward: {waypoint_reward}")                
        
        # 1. Dwell Reward: reward for staying at the final goal location
        dwell_reward = self.get_dwell_reward(achieved_goal= achieved_goal)
        #print(f"Dwell Reward: {dwell_reward}")


        # 2. Velocity Reward: reward for maintaining low velocity
        vel_reward = self.get_velocity_reward()
        #print(f"Velocity Reward: {vel_reward}")

        # 3. Bounds Reward: reward for staying within specified bounds
        bounds_reward = self.get_bounds_reward(flag_index= flag_index,
                                                achieved_goal= achieved_goal,
                                                reward_type = "dense",
                                                reward_shaping = True)
        #print(f"Bounds Reward: {bounds_reward}")

        
        # 4. Orientation Reward: reward for maintaining the right orientation
        orientation_reward = self.get_orientation_reward(flag_index= flag_index,
                                                        reward_type='dense',
                                                        reward_shaping= False) 

        #orientation_reward /= max_reward
        #print(f"Orientation Reward: {orientation_reward}")
        

        ### REWARD STORE:
        self.waypointR += waypoint_reward
        self.dwellR += dwell_reward
        self.velocityR += vel_reward
        self.boundsR += bounds_reward
        """self.orientationR += orientation_reward"""
        

        reward =  waypoint_reward + dwell_reward + vel_reward + bounds_reward #+ orientation_reward
        return reward



    def get_waypoint_reward(self, 
                            flag_index: np.int64, 
                            achieved_goal: np.ndarray, 
                            desired_goal: np.ndarray, 
                            reward_type: str, 
                            reward_shaping : bool):
        """
        Reward that the agent gets for reaching intermediate waypoints.
        
        Parameters:
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which target location we are supposed to reach.
        achieved_goal(np.ndarray)   --- Current location of the EE
        desired_goal(np.ndarry)     --- Next location the EE has to reach
        reward_type(str)            --- sparse | dense
        reward_shaping(bool)        --- if any sort of reward shaping is being used to speeden up the process
        
        Returns:
        reward (np.float)           --- Penalty that the agent receives for staying outside the container
        """
        true_reward = 0.0
        shaped_reward = 0.0

        true_reward = self.get_success_reward(flag_index= flag_index, 
                                              achieved_goal= achieved_goal,
                                              reward_type= reward_type)
        
        if reward_shaping:
           dist = np.linalg.norm(achieved_goal - desired_goal)
           shaped_reward = -dist

        reward = true_reward + shaped_reward
        return reward
    


    def get_success_reward(self, 
                           flag_index: np.int64, 
                           achieved_goal: np.ndarray, 
                           reward_type: str):
        """
        Checks whether the EE has reached any of the intermediate locations and rewards based on that. Also updates the flag incase 
        one of the reward locations have been reached.

        Parameters: 
            achieved_goal(np.ndarray)   --- current EE location
            flag_index(int)             --- index in the target location array that we want to reach
            reward_type(str)            --- sparse | dense
        """
        success_reward = 0.0

        # All flags completed (Dwelling phase)
        if np.all(self.flags == 1.0): 
            return success_reward

        # All flags not reached yet
        target_point = self.targetPoints[flag_index + 1]
        dist = np.linalg.norm(achieved_goal - target_point)

        if self.flags[flag_index] == 0 and dist < self.distanceThreshold:
            self.flags[flag_index] = 1.0                        # switches flag to indicate goal has been reached
            success_reward += (flag_index + 1) * 10             # using 10 as the reward multiplier
        
        elif reward_type == "dense": 
            success_reward -= 1.0                               # when agent doesn't reach any of the target points

        return success_reward
    


    def get_dwell_reward(self, 
                         achieved_goal: np.ndarray):
        """
        If the agent has reached all the target locations and is in the final goal location it returns extra reward for 
        staying there.
        
        Parameters:
            achieved_goal(np.ndarray)       --- Current location of the EE
        
        Returns:
            dwell_reward(np.float32)        --- Reward for staying in the final goal location
        """
        dwell_reward = 0.0
        target_point = self.targetPoints[-1]
        dist = np.linalg.norm(achieved_goal - target_point)

        if np.all(self.flags == 1.0) and dist < self.distanceThreshold:
            dwell_reward += 1.0

        elif np.all(self.flags == 1.0) and dist > self.distanceThreshold:
            dwell_reward += -1.0
        
        return dwell_reward



    def get_velocity_reward(self,
                            lambda_vel = 1.0):
        """
        Penalizes high velocities of the end effector. Higher the end effector velocity, lower is the reward that we get. A velocity window is used to average the 
        velocity across last 'n' timesteps and try to keep that in control. 
        The velocity_reward after multiplying with lambda_vel and dividing by the max_reward returns a value which around 5-20% of the waypoint reward.

        Parameters:
            lambda_vel(np.float64)      --- multiplier for the velocity reward
        """

        ee_velocity = np.array(self.get_ee_velocity())
        
        self.velocity_window.append(ee_velocity)
        vel_mean = np.mean(self.velocity_window, axis = 0)
        
        vel_reward = -lambda_vel * np.square(np.linalg.norm(vel_mean))

        return vel_reward



    def get_bounds_reward(self, 
                           flag_index: np.int64, 
                           achieved_goal: np.ndarray,
                           reward_type: str,
                           reward_shaping: bool):
        """
        Checks whether the EE is within the bounds of the current container. First it checks which container are we considering and then if the
        EE is within the bounds of this container.

        Params: 
        achieved_goal(np.ndarray)   --- Current location of the EE
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which target location we are supposed to reach.
        gamma_bounds(np.float64)    --- Multiplier
        reward_type(str)            --- dense | sparse
        reward_shaping(bool)        --- If shaped reward is true, -ve of the distance from the desired orientation is used as the shaped reward

        Returns:
        penalty (np.float)          --- Penalty that the agent receives for staying outside the container
        """
        bounds_reward = 0.0

        # when we are at the final location... switch off bounds penalty focus on staying within threshold
        # All flags completed (Dwelling phase)
        if np.all(self.flags == 1.0): 
            return bounds_reward
      
        # Step 1: Check which intermediate positions have been reached
        box_center = self.box_center[flag_index]
        box_hE = self.box_hE[flag_index]
        box_ori = self.box_ori[flag_index]

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
        for i in range(3):
            
            excess = max(0, (np.abs(local_pt[i]) - box_hE[i]))
            
            # Sparse Reward
            if reward_type == "sparse":
                if excess > 0: 
                    bounds_reward += 0
                else:
                    bounds_reward += 0.1
            
            # Dense Reward
            elif reward_type == "dense":
                if excess > 0:
                    bounds_reward -= 1/3
                else: 
                    bounds_reward += 0.1
            
            else:
                print(f"Wrong reward type for bounds reward.")

            # Shaped Reward
            if reward_shaping:
                if excess > 0:
                    bounds_reward -= excess + 0.5 * np.square(excess)     

        return bounds_reward



    def get_orientation_reward(self, 
                               flag_index: np.int64,
                               reward_type: str = "sparse",
                               reward_shaping: bool = False,
                               lambda_orient:np.float64 = 0.1):
        """
        Checks the orientation of the EE and compares it to that of surface normal of the current container. First it checks which container
        we are considering, then checks whether the orientation matches.

        Params: 
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which surface normal we are supposed to match.
        reward_type(str)            --- dense | sparse
        reward_shaping(bool)        --- If shaped reward is true, -ve of the distance from the desired orientation is used as the shaped reward
        lambda_orient(np.float)     --- Multiplicative Factor
        
        Returns:
        orientation_reward(np.float)--- Reward for approaching the desired orientation
        """
        true_reward = 0.0
        shaped_reward = 0.0
        #threshold = 0.25          # radians
        threshold = 0.087          # 5 degrees

        orientation_reward = 0.0

        # Step 0: If in Dwell Phase turn off orientation reward
        if np.all(self.flags == 1.0):
            return orientation_reward 

        # Step 1: Get current EE orientation
        ee_orientation = self.get_ee_orientation()
        r_ee = R.from_quat(ee_orientation)
        ee_rpy = r_ee.as_euler('xyz', degrees = False)
        
        # Step 2: Average the orientation over last 'n' timesteps
        self.orientation_window.append(ee_rpy)
        ee_rpy_mean = np.mean(self.orientation_window, axis= 0)
        #print(f"EE RPY mean: {ee_rpy_mean}")

        # Step 3: Get the target orientation
        ori_target = self.surface_normals[flag_index]               # the orientation received is 180 deg opposite to the required orientation
        r_target = R.from_quat(ori_target)                          # Rotation object for the target orientation
        target_rpy = r_target.as_euler('xyz', degrees = False)
        #print(f"Target RPY: {target_rpy}")

        # Step 4: Calculate difference
        #diff = np.abs(target_rpy - ee_rpy_mean)
        diff = np.linalg.norm(target_rpy - ee_rpy_mean)
        check =  diff <  threshold                                  # Checks if each of the diff values satisfy the above condition or not

        # Step 5: Check reward_type
        if reward_type == "sparse":
            #if np.all(check):
            if check:
                true_reward += (flag_index + 1) * 0.1               # using 0.1 as the reward multiplier

        else:
            #if np.all(check):
            if check:
                true_reward += (flag_index + 1) * 0.1               # using 0.1 as the reward multiplier
            else:
                true_reward += -0.1                                 # -0.1 for every timestep with wrong orientation
        
        # Step 6: Check for reward shaping
        #if reward_shaping and not np.all(check):
        if reward_shaping and not check:
            shaped_reward += -lambda_orient * np.linalg.norm(diff)  # -ve distance from target as penalty
        
        orientation_reward = true_reward + shaped_reward
        
        return orientation_reward



    def get_singularity_penalty(self):
        """
        Calculates the joint angles of robot and checks if its configuration is close to kinematic singularity. 
        Returns the manipulability of the robotic arm.
        """
        joint_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
        joint_states = self.sim.physics_client.getJointStates(self.body_id, joint_indices)

        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        joint_accelerations = [0.0] * len(joint_indices)

        jac_t, jac_r = self.sim.physics_client.calculateJacobian(
            bodyUniqueId=self.body_id,
            linkIndex=self.ee_link,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=joint_accelerations
            )
        
        jacobian = np.vstack((jac_t, jac_r))
        JJT = np.dot(jacobian, jacobian.T)  # shape (6,6)
        manipulability = np.sqrt(np.linalg.det(JJT))
        
        return manipulability
        


    def get_achieved_goal(self) -> np.ndarray:
        """
        Returns the ee position of the robot.
        """
        ee_position = np.array(self.get_ee_position())
        return ee_position



    # Abstract Method
    def is_success(self, 
                   achieved_goal: np.ndarray, 
                   desired_goal: np.ndarray, 
                   info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Checks whether the EE has reached the Final Destination. The function overall isn't of much use as it just add True or False to the
        "terminated" variable. We overwrite it anyways in the PandaOrientTaskEnv as we want our episode to end after 200 steps not after
        it reachs the goal location.

        achieved_goal --- observation["achieved_goal"] ---> self.get_achieved_goal()
        desired_goal --- self.get_goal() ---> self.goal  
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        success = np.array(dist < self.distanceThreshold, dtype=bool)
        return success



    # Abstract Method
    def get_obs(self) -> np.ndarray:
        """
        Return the observation associated to the task. Returns the distance between the end effector and the goal location.
        """
        # Progress made based on number of points traversed
        flag_index = self.check_flag()                              # checks the number of points we have reached

        
        if flag_index is None:
            # All points have been reached so target location is the final point where we dwell
            flag_index = len(self.flags) - 1

        ee_progress = (flag_index + 1)/ len(self.flags)

        # relative distance between current location and target location
        target = self.targetPoints[flag_index + 1]                  # target location   ---> next point to reach
        ee_position = self.get_achieved_goal()                      # goal location     ---> final point
        ee_delta = target - ee_position

        # Relative orientation between the target and the current orientation quaternion
        ee_orientation = self.get_ee_orientation()
        r_current = R.from_quat(ee_orientation)                     # current orientation of the end effector

        ori_target = self.surface_normals[flag_index]                
        r_target = R.from_quat(ori_target)                          # desired orientation of the end effector 

        r_relative = r_current.inv() * r_target                     # the output quaternion (r_relative) gives us the rotation we need to apply to 
        r_relative = r_relative.as_quat()                           # the current quaternion to get the target quaternion
        
        # Final observation
        self.observation = np.concatenate((ee_delta, r_relative, [ee_progress])) 
        self.observation = np.array(self.observation, dtype= np.float32)
        
        return self.observation



    def _sample_target(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        target = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return target
    

