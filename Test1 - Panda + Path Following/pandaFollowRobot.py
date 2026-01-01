from typing import Optional

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

from gymnasium import spaces
import numpy as np


class pandaFollowRobot(PyBulletRobot):
    """
    Observation Space

    The Observation Space is a `ndarray` with shape `(6,)` representing the end effector positions and the
    end effector velocities.

    | Num | Action      | Min  | Max |
    |-----|-------------|------|-----|
    | 0   | ee_x        | -1.0 | 1.0 |
    | 1   | ee_y        | -1.0 | 1.0 |
    | 2   | ee_z        | -1.0 | 1.0 |
    | 3   | ee_vx       | -1.0 | 1.0 |
    | 4   | ee_vy       | -1.0 | 1.0 |
    | 5   | ee_vz       | -1.0 | 1.0 |
    
    Note: Min and Max values here are just placeholders. The values can vary as they want.

    ### Action Space

    The Action Space is an ndarray and can take 2 forms. 
        Case 1: [joints] Joint Angle Control (Joint Trajectory Control) ---> control the joint angles 

        | Num | Observation      | Min    | Max   |
        |-----|------------------|--------|-------|
        | 0   | Shoulder Pan     | -87.0  | 87.0  |
        | 1   | Shoulder Lift    | -87.0  | 87.0  |
        | 2   | Elbow            | -87.0  | 87.0  |
        | 3   | Forearm Roll     | -87.0  | 87.0  |
        | 4   | Wrist Pitch      | -12.0  | 12.0  |
        | 5   | Wrist Roll       | -120.0 | 120.0 |
        | 6   | Wrist Yaw        | -120.0 | 120.0 |
        | 9   | Gripper Finger 1 | -170.0 | 170.0 |
        | 10  | Gripper Finger 2 | -170.0 | 170.0 |

        Note: the min and max values here are values of the torque not the angle that the joint can rotate. It written
        for book keeping and not for setting limits to the angle that the joint can rotate.
        Note: Joints 7 and 8 (between 6 and 9) are often fixed or internal helper joints in some URDFs — 
        hence skipped in indexing here.

        Case 2: [ee] End Effector Control ---> control the end effector position
        | Num | Observation   | Min   | Max  |
        |-----|---------------|-------|------|
        | 0   | x             | -1.0  | 1.0  |
        | 1   | y             | -1.0  | 1.0  |
        | 2   | z             | -1.0  | 1.0  |
        
        Note: Min and Max values here are just placeholders. X, Y, Z can vary as they want. 
    

    Parameters:
    sim (PyBullet):                         Simulation instance.
    block_gripper (bool, optional):         Whether the gripper is blocked. Defaults to True (can't use the gripper).
    base_position (np.ndarray, optional):   Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
    control_type (str, optional):           [joints | ee] "ee" to control end-effector displacement or "joints" to control joint angles.
                                            Defaults to "joints".
    
    """

    def __init__(self, 
                 sim: PyBullet,
                 block_gripper: bool = True,
                 base_position: Optional[np.ndarray] = None,
                 control_type = "ee") -> None:
        
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        
        self.n_actions = 7 if self.control_type == "joints" else 3  # Action Space: 
                                                                    #   Joint Angle Control ==> 7
                                                                    #   End Effector Control ==> 3  
        self.n_actions += 0 if self.block_gripper else 1            # Extra DoF added when the gripper isn't blocked (block_gripper = False)
        action_space = spaces.Box(-1.0, 1.0, shape= (self.n_actions,), dtype= np.float32)
        
        super().__init__(
            sim,
            body_name= "panda",                                     # choose the name you want
            file_name= "franka_panda/panda.urdf",                   # the path of the URDF file
            base_position= self.base_position,                      # the position of the base
            action_space= action_space,
            joint_indices= np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),  # list of joint indices that can be controlled
            joint_forces= np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),           
                                                                    # max force applicable for each of the joints
        )
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
                                                                    # home position for all controllable links
        self.distanceThreshold = 0.1
        self.fingers_indices = np.array([9, 10])
        self.ee_link = 11
    


    def set_action(self, action: np.ndarray) -> None:
        """
        Takes in an action and executes for a timestep. 
        
        Parameters:
        action (ndarray) --- joint angle control - (7,) | EE control - (3,) 
        """
        action = action.copy()                              # copying the given action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_type == "joints":
            joint_disp = action[:7]
            target_joint_angles = self.JointAngleControl_To_JointAngles(joint_disp)
        else:
            ee_disp = action[:3]
            target_joint_angles = self.EEControl_To_JointAngles(ee_disp)
        
        # Setting the finger grippers to 0
        target_joint_angles = np.concatenate((target_joint_angles, [0, 0]))

        self.control_joints(target_angles= target_joint_angles)



    def JointAngleControl_To_JointAngles(self, joint_disp) -> np.ndarray:
        """
        Takes in the joint angle change to reach the final position as given by the network and gives the joint angles for the robot for reaching 
        the next position for 1 timestep after displacement

        Parameters:
        joint_disp (ndarray) --- desired change of joint angles to reach the goal position as predicted by the network

        Returns:
        target_joint_angles (ndarray) --- joint angle for the next position of the robot
        """
        percent = 0.05                                  # percentage of the action that we will be able to execute for the timestep
        joint_disp = joint_disp[0:7] * percent          # limiting the amount of displacement in a timestep
        
        # get the current joint angles
        current_joint_angles = np.array([self.get_joint_angle(joint= i) for i in range(7)])

        target_joint_angles = current_joint_angles + joint_disp
        return target_joint_angles



    def EEControl_To_JointAngles(self, ee_disp: np.ndarray) -> np.ndarray:
        """
        Takes in the end effector displacement required to reach the goal position as given by the network and gives the joint angles for the 
        robot for reaching the next position for 1 timestep after displacement

        Parameters:
        ee_disp (ndarray) --- desired displacement of the ee to reach the goal position as predicted by the network

        Returns:
        target_joint_angles (ndarray) --- joint angle for the next position of the robot
        """
        percent = 0.05                                  # percentage of the action that we will be able to execute for the timestep
        ee_disp = ee_disp[:3] * percent                 # limiting the amount of displacement in a timestep                      
        current_ee_position = self.get_ee_position() 
        target_ee_position = current_ee_position + ee_disp
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        
        # Will have to change orientation here according to the path
        target_joint_angles = self.inverse_kinematics(
            link= self.ee_link, position= target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )

        target_joint_angles = target_joint_angles[:7]       # remove fingers angles
        return target_joint_angles

      

    def get_obs(self):
        """
        Returns the current observation based on the robot parameters. Observation includes EE position and velocity.

        Returns:
        self.observation (ndarry) --- (ee_position, ee_velocity)
        """
        ee_position = self.get_ee_position()
        ee_velocity = self.get_ee_velocity()

        self.observation = np.concatenate((ee_position, ee_velocity))
        self.observation = np.array(self.observation, dtype=np.float32)

        return self.observation



    def reset(self):
        """
        Resets the robot back to its neutral position. 
        """
        self.set_joint_angles(angles= self.neutral_joint_values)


    def get_ee_position(self) -> np.ndarray:
        """
        Returns:
        ee_link_position (ndarray) --- the position of the end-effector as (x, y, z)
        """
        ee_link_position = self.get_link_position(self.ee_link)
        return ee_link_position
    


    def get_ee_velocity(self) -> np.ndarray:
        """
        Returns:
        ee_link_velocity (ndarray) --- the velocity of the end-effector as (vx, vy, vz)
        """
        ee_link_velocity = self.get_link_velocity(self.ee_link)
        return ee_link_velocity




