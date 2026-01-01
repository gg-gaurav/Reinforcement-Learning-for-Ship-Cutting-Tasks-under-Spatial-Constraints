from panda_gym.envs.core import Task


import numpy as np
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R
from collections import deque
import time

class templateTask(Task):
    """
    This task as a template to check the quality of any setup that needs to be tested.
    The environment consists of the robot, a table and a single cube. The agent starts
    from the start location, follows the bounds to the cube. It then tracks around the 
    maintaining an orientation perpendicular to it at all times.
    """


    def __init__(self, 
                 sim, 
                 body_id,
                 get_ee_position,
                 get_ee_velocity,
                 get_ee_orientation,
                 get_joint_angle,
                 get_joint_velocity,
                 set_joint_angles,
                 inverse_kinematics,
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
           inverse_kinematics  ---> function for inverse kinematics
           distanceThreshold   ---> distance needed to consider that the task is complete
         """
        super().__init__(sim)
        self.body_id= body_id,
        self.get_ee_position = get_ee_position
        self.get_ee_velocity = get_ee_velocity
        self.get_ee_orientation = get_ee_orientation
        self.get_joint_angle = get_joint_angle
        self.get_joint_velocity = get_joint_velocity
        self.set_joint_angles = set_joint_angles
        self.inverse_kinematics = inverse_kinematics
        self.distanceThreshold = distanceThreshold
        
        # home position for all controllable links
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11

        # (WONT BE NEEDING THESE) Bounds for selecting the target point. 
        self.goal_range_low = np.array([0.25, -0.30, 0])           
        self.goal_range_high = np.array([0.80, 0.30, 0.80]) 

        self.max_steps = 200
        self.episodic_steps = 0
        
        # Stores start and end points
        self.start = np.array([0.20, 0.3, 0])          # placeholder for the start point
        self.goal  = np.array([0.20, -0.3, 0])         # placeholder for the goal point


        # TASK SPECIFIC REQUIREMENTS:
        self.numPoints = 6              # Total Number of points through which we need to traverse (INCLUDING START AND END)
        
        self.targetPoints = []          # Stores coordinates of the points
        self.target_spheres = []        # Stores ids for all the target spheres
        self.box_IDs = []               # Stores the IDs of the containers
        self.box_center = []            # Stores the centers of the containers
        self.box_hE = []                # Stores the half extents of the containers
        self.box_orients = []           # Stores the orientations of the containers
        self.surface_normals=[]         # Stores the quaternions for surface normals for each of the containers
        self.viz_normals = []           # if we visualize the surface normals then it stores the object ids 

        self.flags = np.zeros(self.numPoints - 1)       # Stores how many intermediate points the EE has crossed. n-1 points if we remove the start point
        self.orientation_window = deque(maxlen= 10)     # Stores the orientation value over last 10 timesteps
        self.velocity_window = deque(maxlen= 10)         # Stores velocity value over last 10 timesteps

        self.object_IDs = []            # Stores ids for all the objects around which we need to traverse
        self.object_centers = []        # Stores the centers of the objects
        self.object_hEs = []            # Stores the half Extents of the objects
        self.object_orients = []        # Stores the orientation of the objects

        ## REWARD STORAGE:
        self.waypointR = 0.0
        self.dwellR = 0.0
        self.velocityR = 0.0
        self.boundsR = 0.0

        
        with self.sim.no_rendering():
            self._create_scene()

            
    # Abstract Method
    def reset(self) -> None:
        """
        Reset the task:
            1. Set number of steps taken in current episode to be 0
            2. Send EE to start position. 
            3. Send object to original location
            4. Reset flags
            5. Set final goal position and orientation

        """
        # Reset total steps taken for the current episode
        self.episodic_steps = 0
        
        # Set EE to start location of task
        start_joint_angles = self.inverse_kinematics(
            link= self.ee_link, position= self.start, orientation= np.array([1.0, 0.0, 0.0, 0.0])
        )
        start_joint_angles = start_joint_angles[:7]
        self.set_joint_angles(start_joint_angles)
        self.sim.set_base_pose("start", self.start, np.array([0.0, 0.0, 0.0, 1.0]))

        """
        # Testing --- Orientation working perfectly
        print(f"Surface Normals: {self.surface_normals}")
        for i,element in enumerate(self.surface_normals):
            R_obj = R.from_quat(element)
            print(f"Normal Quat{i}: {R_obj.as_euler('xyz', degrees=True)}")

            joint_angles= self.inverse_kinematics(
                link = self.ee_link, position = self.start, orientation = element
            )
            joint_angles = joint_angles[:7]
            self.set_joint_angles(joint_angles)
            print("Set to new orientation!!")
            time.sleep(20)
        """
        
        # Send box to original location
        self.sim.physics_client.resetBasePositionAndOrientation(
                                                bodyUniqueId = self.object_IDs[0], 
                                                posObj = self.object_centers[0],
                                                ornObj = self.object_orients[0])

        # Reseting flags
        self.flags = np.zeros(self.numPoints-1)

        # Set end location of task
        self.sim.set_base_pose("goal", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

        ## REWARD STORAGE:
        self.waypointR = 0.0
        self.dwellR = 0.0
        self.velocityR = 0.0
        self.boundsR = 0.0

        
        

    def _create_scene(self) -> None:
        """
        Creates the scene for the experiment:
            0: Panda Robot
            1: Base Plane
            2: Table
            3: Start Location
            4: Main Box
            5: Bounding Boxes
            6: Goal Location
        """
        # Body 0 : The panda robot
        # Body 1 : The surface
        self.sim.create_plane(z_offset=-0.4)                # surface/ floor at z = -0.4
        
        # Body 2 : The table
        # Table top at z=0
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3) 
        
        # Body 3 : Start Location
        self.sim.create_sphere(                             # start positon = [0.1, 0.1, 0.1]
            body_name="start",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position= self.start,
            rgba_color= np.array([0.9, 0.1, 0.1, 0.7]),
        )
        self.start = self.sim.get_base_position("start")
        
    
        self.set_joint_angles(self.neutral_joint_values)
        self.sim.set_base_pose("start", self.start, np.array([0.0, 0.0, 0.0, 1.0]))


        # Body 4: Main Box
        object_center = np.array([0.5, 0, 0.1])
        object_orient = np.array([1.0, 0, 0, 0])
        object_halfExtents = np.array([0.1, 0.1, 0.1])
        visual_kwargs= {
            "halfExtents": object_halfExtents,
            "specularColor": None,
            "rgbaColor": np.array([0.5, 0.5, 0.5, 1.0])
        }
        collision_kwargs= {
            "halfExtents": object_halfExtents
        }
        baseVisualShapeIndex = self.sim.physics_client.createVisualShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                         **visual_kwargs)
        baseCollisionShapeIndex = self.sim.physics_client.createCollisionShape(shapeType= self.sim.physics_client.GEOM_BOX,
                                                                               **collision_kwargs)
        box = self.sim.physics_client.createMultiBody(
                baseMass= 1.0,
                baseCollisionShapeIndex= baseCollisionShapeIndex,
                baseVisualShapeIndex = baseVisualShapeIndex,
                basePosition= object_center,
                baseOrientation= object_orient 
            )
        
        real_center, real_orientation = self.sim.physics_client.getBasePositionAndOrientation(box)
        self.object_IDs.append(box)
        self.object_centers.append(real_center)
        self.object_orients.append(real_orientation)
        self.object_hEs.append(object_halfExtents)

        ## GET TARGET POINTS
        targetPoints = self.getTargetPoints(
                                        self.object_IDs,
                                        self.object_centers,
                                        self.object_orients,
                                        self.object_hEs
                                        )
        self.targetPoints.append(self.start)
        for i in targetPoints:              # Its a nested array. Along the dept direction we get the points for each box
            for j in i:
                self.targetPoints.append(j)
        self.targetPoints.append(self.goal)
        print(f"target points: {self.targetPoints}")
        


        # Body 5: Bounding Boxes
        self.containers, self.surface_normals, self.target_spheres = self.setupContainers( 
                                                                                self.numPoints,
                                                                                self.targetPoints)
        



        # Body 6: Goal Location
        self.sim.create_sphere(                   
            body_name="goal",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position= self.goal,
            rgba_color= np.array([0.9, 0.1, 0.1, 0.7]),
        )
       


    def getTargetPoints(self,
                        object_IDs,
                        object_centers,
                        object_orients,
                        object_hEs) -> np.ndarray:
        """
        Uses the list of boxCenters, boxHalfExtents and boxOrientations to generate a list of target points through which the end effector has
        to pass through
        Parameters:
            object_IDs      : np.ndarray
                IDs of the objects around which the agent should traverse
            object_centers  : np.ndarray
            object_orients  : np.ndarray
            object_hEs      : np.ndarray
        Returns
            target_points   : np.ndarray)
                List of target points 
        """
        # STORE THE EXTRACTED POINTS IN 2D ARRAY
        # (n_point, n_axes) ---> n_rows : number of points
        #                        n_cols : number of axes    
        target_points = []                                         
        
        for i in range(len(object_centers)):
            center = object_centers[i]
            orientation = object_orients[i]
            halfExtents = object_hEs[i]
           
            # Local Coordinates:
            # WE WILL BE CUTTING ACROSS THE X AXIS. The intermediate points are generated according to this assumption and are on the Y-Z plane
            hE_x, hE_y, hE_z = halfExtents
            A_local = np.array([0, -hE_y, hE_z])
            B_local = np.array([0, -hE_y, -hE_z])
            C_local = np.array([0, hE_y, -hE_z])
            D_local = np.array([0, hE_y, hE_z])

            # each row represents a new point for that specific square
            local_points = np.array([A_local, B_local, C_local, D_local]) 
            
            # Get the rotation matrix
            rot = R.from_quat(orientation)

            # Rotate and Translate each point
            global_points = [(rot.apply(p) + center) for p in local_points]
            global_points= np.stack(global_points, axis=0)

            # Check if global_points are according to our needed shape
            assert global_points.shape == (4, 3)

            target_points.append(global_points)

        
        target_points = np.stack(target_points, axis=0)                     # (n_cubes, n_points, 3D)
        return target_points     



    def setupContainers(self,  
                        numPoints,
                        targetPoints):
        """
        Set up a container between the current point and the next. Additionally,
        create a visual sphere at the new point.

        Parameters:
        numPoints(np.ndarry)    --- Total number of points we have to sample
        targetPoints(np.ndarray)--- List of target point that need to be reached

        Returns:
        containers(np.ndarray)      --- Array containing the ids of the containers
        surface_normals(np.ndarray) --- Array containing the surface normals for each of the containers
        target_spheres(np.ndarray)  --- Array containing the ids of the target_spheres
        """

        containers = []                             # stores container IDs
        target_spheres = []                         # stores target sphere IDs
        surface_normals = []                        # stores surface normals
    
        for i in range(numPoints-1):               
            
            start = targetPoints[i]
            target = targetPoints[i+1]
            if i == 0:
                print(f"Start : {start}")
                print(f"Target : {target}")
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
            self.box_orients.append(orientation)
            print(f"z_axis {i}: {z_axis}")
                          
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

            #print(f"Edge Case: direction is straight up/ down")
            
            if target[2] > start[2]:
                #print(f"Target above start") 
                up = np.array([0.0, 1.0, 0.0])
            else:
                #print(f"Target below start")
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
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
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
        #print(f"flags: {self.flags}")
        #print(f"flag_index: {flag_index}")
        #print(f"desired_goal: {desired_goal}")
        

        # 1. Waypoint Reward:
        waypoint_reward = self.get_waypoint_reward(flag_index= flag_index,
                                                achieved_goal= achieved_goal, 
                                                desired_goal= desired_goal,
                                                reward_type= "sparse",
                                                reward_shaping= True)               
        
        # 1. Dwell Reward: reward for staying at the final goal location
        dwell_reward = self.get_dwell_reward(achieved_goal= achieved_goal)

        # 2. Velocity Reward: reward for maintaining low velocity
        vel_reward = self.get_velocity_reward()

        # 3. Bounds Reward: reward for staying within specified bounds
        bounds_reward = self.get_bounds_reward(flag_index= flag_index,
                                                 achieved_goal= achieved_goal)

        """
        # 4. Orientation Reward: reward for maintaining the right orientation
        orientation_reward = self.get_orientation_reward(flag_index= flag_index,
                                                         reward_type='sparse',
                                                         reward_shaping= True)
        orientation_reward /= max_reward
        #print(f"Orientation Reward: {orientation_reward}")
        """


        ### REWARD STORE:
        self.waypointR += waypoint_reward
        self.dwellR += dwell_reward
        self.velocityR += vel_reward
        self.boundsR += bounds_reward

        reward =  waypoint_reward + dwell_reward + vel_reward + bounds_reward
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
                           gamma_bounds = 0.5):
        """
        Checks whether the EE is within the bounds of the current container. First it checks which container are we considering and then if the
        EE is within the bounds of this container.

        Params: 
        achieved_goal(np.ndarray)   --- Current location of the EE
        flag_index(np.ndarray)      --- Index value for self.flags. Tells us which target location we are supposed to reach.
        gamma_bounds(np.float64)    --- Multiplier

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
        box_ori = self.box_orients[flag_index]

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
            if excess > 0:
                bounds_reward -= 1/3
                
        return gamma_bounds * bounds_reward



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
        threshold = 0.25

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
        diff = np.abs(target_rpy - ee_rpy_mean)
        check =  diff <  threshold                                  # Checks if each of the diff values satisfy the above condition or not

        # Step 5: Check reward_type
        if reward_type == "sparse":
            if np.all(check):
                true_reward += (flag_index + 1) * 0.1               # using 0.1 as the reward multiplier

        else:
            if np.all(check):
                true_reward += (flag_index + 1) * 0.1               # using 0.1 as the reward multiplier
            else:
                true_reward += -1.0                                 # -1 for every timestep with wrong orientation
        
        # Step 6: Check for reward shaping
        if reward_shaping and not np.all(check):
            shaped_reward += -lambda_orient * np.linalg.norm(diff)  # -ve distance from target as penalty
        
        orientation_reward = true_reward + shaped_reward
        
        return orientation_reward



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
        'terminated' variable. We overwrite it anyways in the PandaOrientTaskEnv as we want our episode to end after 200 steps not after
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
    
    