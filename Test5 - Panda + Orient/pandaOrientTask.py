from panda_gym.envs.core import Task


import numpy as np
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R


class pandaOrientTask(Task):
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
                 get_ee_position,
                 get_ee_velocity,
                 get_ee_orientation,
                 distanceThreshold= 0.05) -> None:
        """
        Parameters:
            sim                 ---> simulator
            get_ee_position     ---> function that returns the end effector position
            get_ee_velocity     ---> function that returns the end effector velocity
            get_ee_orientation  ---> function that returns the end effector orientation
            distanceThreshold   ---> distance needed to consider that the task is complete
        """
        super().__init__(sim)
        self.get_ee_position = get_ee_position
        self.get_ee_velocity = get_ee_velocity
        self.get_ee_orientation = get_ee_orientation
        self.distanceThreshold = distanceThreshold
        
        # Selecting goals from an area above the table
        self.goal_range_low = np.array([0.30, -0.35, 0])
        self.goal_range_high = np.array([0.85, 0.35, 0.85])
        
        self.max_steps = 200
        self.episodic_steps = 0

        # stores the start and end points
        self.start = None               # placeholder for the start point
        self.goal  = None               # placeholder for the goal point

        self.numPoints = 4              # Total number intermediate points the EE needs to traverse
        self.targetPoints = []          # Stores the coordinates of the points
        self.target_spheres = []        # Stores the ids for all the target_spheres
        self.containers = []            # Stores the ids for all the containers
        self.box_center = []            # Stores the centers of the containers
        self.box_hE = []                # Stores the half Extents of the containers
        self.box_ori = []               # Stores the orientations of the containers
        self.surface_normals = []       # Stores the quaternions for surface normals for each of the containers
        self.viz_normals = []           # if we visualize the surface normals then it stores the object ids

        # Stores how many intermediate points the EE has crossed
        self.flags = np.zeros(self.numPoints)

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
        self.containers, self.surface_normals, self.target_spheres = self.setupContainers(self.start, self.numPoints)

        
        
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
        self.containers, self.surface_normals, self.target_spheres = self.setupContainers(self.start, self.numPoints)

        # Body 5: the end point
        self.goal_id = self.target_spheres[-1]
        self.goal = self.targetPoints[-1]
        


    def setupContainers(self, start, numPoints):
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
        self.targetPoints.append(start)
        containers = []
        target_spheres = []
        surface_normals = []

        
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
            orientation, z_axis = self.get_container_orient(start, target)
            self.box_ori.append(orientation)
                          
            # Setting Up the Container
            visual_kwargs = {
                "halfExtents": halfExtents,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.1, 0.9, 0.9]) 
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
                "length": 0.1,
                "specularColor": None,
                "rgbaColor": np.array([0.1, 0.9, 0.1, 0.5])  
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

    

    def get_surface_normal(self, z_axis):
        """
        Returns the surface norm for the container. Note: The normal that it returns points inwards rather than outwards

        Parameters:
            z_axis(np.ndarry)       --- z_axis of the container
        Returns:
            orient_norm(np.ndarray) --- orientation of the inward pointing surface norm 
        """
        z_axis = -z_axis/ np.linalg.norm(z_axis)    # -ve z_axis because we want it to point inwards
        z_axis_ref = np.array([0.0, 0.0, 1.0])

        dot = np.dot(z_axis_ref, z_axis)
        orient_norm = None

        if np.allclose(dot, 1.0):
            orient_norm =  R.identity().as_quat()
        
        elif np.allclose(dot, -1.0):
            # 180° rotation around X or Y (any perpendicular axis)
            orient_norm =  R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()
        else:
            axis = np.cross(z_axis_ref, z_axis)
            axis = axis / np.linalg.norm(axis)
            theta = np.arccos(dot)

            orient_norm = R.from_rotvec(theta * axis).as_quat()
        return orient_norm



    def get_container_orient(self, start, target):
        """
        Gets the orientation of the container between start and target. Also ensures that there is no roll about x-axis when
        we rotate the current X axis to its new orientation

        Returns:
        quat_or(np.ndarray) --- orientation quaternion for the container
        z_axis(np.ndarry) --- z_axis that is equivalent to the surface normal
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
        return quat_orient, z_axis
    


    def check_flag(self) -> np.int64:
        """
        Check the self.flags list and returns the index where it finds the first 0

        Returns:
            The index of the flag that we are on
        """
        for i, val in enumerate(self.flags):
            if val == 0:
                return i
        # if all the values in the list are 1 (all point have been reached), send the last index value so 
        # that the EE stays in the final location
        return (self.numPoints - 1) 



    # Abstract Method
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """
        Compute reward associated to the task. We have 3 types of reward:
            1. Distance Penalty: For reaching the target
            2. Velocity Penalty: For slowing down when we are near the target
            3. Within Bounds Penalty: For ensuring the EE stays within the container
            4. Success Reward: For reaching intermediate points
            5. Dwell Reward: For reaching and staying at the final goal location
            6. Orientation Penalty: For maintaining the right orientation while traversing between points
            7. Along the centerline Reward (?)
        
        Note: All penalties have -ve values and all rewards have +ve values 
        """
        flag_index = self.check_flag()
        desired_goal = self.targetPoints[flag_index]              # intermediate target set
        
        # Reward Weights:
        alpha = 1.0         # Distance Penalty
        beta = 0.35         # Velocity Penalt
        gamma = 0.45        # Bounds Penalty
        delta = 2.0         # Success Reward
        omega = 0.5         # Orientation Penalty

        # 1. Distance Penalty
        dist = np.linalg.norm(achieved_goal - desired_goal)
        dist_penalty = -dist

        # 2. Velocity Reward
        ee_velocity = np.array(self.get_ee_velocity())
        vel_sqr = np.square(np.linalg.norm(ee_velocity))
        vel_penalty = -vel_sqr

        # 3. Within Bounds Reward
        bounds_penalty = self.get_bounds_penalty(achieved_goal= achieved_goal,
                                                      flag_index= flag_index)
        
        # 4. Success Reward: reward for reaching goal
        success_reward = self.get_success_reward(achieved_goal= achieved_goal,
                                               flag_index = flag_index)
        
        # 5. Dwell Reward: reward for staying at the final goal location
        dwell_reward = self.get_dwell_reward(achieved_goal= achieved_goal)

        # 6. Orientation Penalty: reward for maintaining the right orientation
        orient_penalty = self.get_orientation_penalty(achieved_goal= achieved_goal, flag_index= flag_index)
        
        """
        print(f"distance reward: {alpha * dist_penalty}")
        print(f"velocity reward: {beta * vel_penalty}")
        print(f"within reward: {gamma * bounds_penalty}")
        print(f"target reward: {delta * success_reward}")
        print(f"dwell reward: {dwell_reward}")
        print(f"orientation reward: {omega * orient_penalty}")
        print(f"flags array: {self.flags}")
        """

        reward = alpha*dist_penalty + beta*vel_penalty + gamma*bounds_penalty + delta*success_reward + dwell_reward + omega*orient_penalty
        
        return reward



    def get_bounds_penalty(self, achieved_goal: np.ndarray, flag_index: np.int64):
        """
        Checks whether the EE is within the bounds of the current container. First it checks which container are we considering and then if the
        EE is within the bounds of this container.

        Params: 
        achieved_goal(np.ndarray)   --- Current location of the EE
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which target location we are supposed to reach.

        Returns:
        penalty (np.float)          --- Penalty that the agent receives for staying outside the container
        """
        penalty = 0.0
        if flag_index == 0:
            return penalty
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
    


    def get_orientation_penalty(self, achieved_goal: np.ndarray, flag_index: np.int64):
        """
        Checks the orientation of the EE and compares it to that of surface normal of the current container. First it checks which container
        we are considering, then checks whether the orientation matches

        Params: 
        achieved_goal(np.ndarray)   --- Current location of the EE
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which surface normal we are supposed to match.
        
        Returns:
        penalty(np.float)           --- Penalty the agent receives for not matching the desired orientation
        """
        ee_orientation = self.get_ee_orientation()
        penalty = 0.0
        if flag_index == 0:                     # n_container = n_flags - 1.... so when the flag_index is 0 we just dont consider the orientation
            return penalty
        else:
            # Step 1: Get the index 
            ori_index = flag_index-1
            ori_target = self.surface_normals[ori_index]            # the orientation received is 180 deg opposite to the required orientation
            
            # Step 2: Calculate the loss
            ori_target = R.from_quat(ori_target)
            dot = np.dot(ori_target, ee_orientation)
            penalty -= 1 - dot
            return penalty



    def get_achieved_goal(self) -> np.ndarray:
        """
        Returns the ee position of the robot.
        """
        ee_position = np.array(self.get_ee_position())
        return ee_position



    # Abstract Method
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
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
        ee_position = self.get_achieved_goal()

        flag_index = self.check_flag()
        target = self.targetPoints[flag_index]

        self.observation = target - ee_position
        return self.observation



    def _sample_target(self) -> np.ndarray:
        """
        Sample a goal location in the given space which the robot has to reach.

        Return:
        goal (ndarry) --- returns the goal location
        """
        target = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return target
    

