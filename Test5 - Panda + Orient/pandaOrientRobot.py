from typing import Optional

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

from gymnasium import spaces
import numpy as np


class pandaOrientRobot(PyBulletRobot):
    """
    Observation Space

    The Observation Space is a `ndarray` with shape `(12,)` representing the end effector positions and the
    end effector velocities.

    | Num | Action      | Description                           |
    |-----|-------------|---------------------------------------|
    | 0   | ee_x        | EE x coord                            |
    | 1   | ee_y        | EE y coord                            |
    | 2   | ee_z        | EE z coord                            |
    | 3   | ee_vx       | EE velocity in x dir                  |
    | 4   | ee_vy       | EE velocity in y dir                  |
    | 5   | ee_vz       | EE velocity in z dir                  |
    | 6   | q_x         | x of quaternion for EE orientation    |
    | 7   | q_y         | y of quaternion for EE orientation    |
    | 8   | q_z         | z of quaternion for EE orientation    |
    | 9   | q_w         | w(scalar) part of quaternion          |
    | 10  | dx          | dist btwn EE_x and goal_x             |
    | 11  | dy          | dist btwn EE_y and goal_y             |
    | 12  | dz          | dist btwn EE_z and goal_z             |

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
        Note: Since we are not using the gripper the values for joint 9 and 10 stay fixed

        Case 2: [ee] End Effector Control ---> control the end effector position
        | Num | Observation   | Desicription                                    |                                    
        |-----|---------------|-------------------------------------------------|
        | 0   | x             | Disp_x needed to reach target position          |
        | 1   | y             | Disp_y needed to reach target position          |
        | 2   | z             | Disp_z needed to reach target orientation       |
        | 3   | qx            | quaternion x to reach target orientation        |
        | 4   | qy            | quaternion y to reach target orientation        |
        | 5   | qz            | quaternion z to reach target orientation        |
        | 6   | qw            | quaternion w to reach target orientation        |
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
        
        self.n_actions = 7 if self.control_type == "joints" else 7  # Action Space: 
                                                                    #   Joint Angle Control ==> 7
                                                                    #   End Effector Control ==> 7  
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
        self.fingers_indices = np.array([9, 10])
        self.ee_link = 11
    


    # Abstract method
    def set_action(self, action: np.ndarray) -> None:
        """
        Takes in an action from the NN and executes for a timestep. 
        
        Parameters:
        action (ndarray) --- joint angle control - (7,) | EE control - (7,) 
        """
        action = action.copy()                              # copying the given action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_type == "joints":
            joint_disp = action[:7]
            target_joint_angles = self.JointAngleControl_To_JointAngles(joint_disp)
        else:
            ee_disp = action[:7]
            target_joint_angles = self.EEControl_To_JointAngles(ee_disp)
        
        # Setting the finger grippers to 0 and adding it to the target list
        target_joint_angles = np.concatenate((target_joint_angles, [0, 0]))

        self.control_joints(target_angles= target_joint_angles)



    def JointAngleControl_To_JointAngles(self, joint_disp) -> np.ndarray:
        """
        Takes in the joint angle change to reach the final position as given by the network and gives the joint angles for the robot for reaching 
        the next position for 1 timestep after displacement

        Parameters:
        joint_disp (ndarray) --- joint displacement; desired change of joint angles to reach the goal position as predicted by the network

        Returns:
        target_joint_angles (ndarray) --- joint angle for the next position of the robot
        """
        percent = 0.05                                  # percentage of the action that we will be able to execute for the timestep
        joint_disp = joint_disp[0:7] * percent          # limiting the amount of displacement in a timestep
        
        # get the current joint angles
        current_joint_angles = np.array([self.get_joint_angle(joint= i) for i in range(7)])

        target_joint_angles = current_joint_angles + joint_disp
        return target_joint_angles



    def EEControl_To_JointAngles(self, ee_delta: np.ndarray) -> np.ndarray:
        """
        Takes in the end effector displacement required to reach the goal position as given by the network and gives the joint angles for the 
        robot for reaching the next position for 1 timestep after displacement

        Parameters:
        ee_delta (ndarray) --- end effector displacement, desired displacement and rotation of the ee to reach the goal position as predicted 
                                by the network

        Returns:
        target_joint_angles (ndarray) --- joint angle for the next position of the robot
        """
        # Position
        percent = 0.05                                   # percentage of the action that we will be able to execute for the timestep
        ee_disp = ee_delta[:3] * percent                 # limiting the amount of displacement in a timestep                      
        target_ee_position = self.get_ee_position()  + ee_disp
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        
        # Orientation : 
        ee_orient_disp = ee_delta[3:] * percent
        # Addition in Euler
        target_ee_orientation = self.get_ee_orientation() + ee_orient_disp
        # Get orientation quaternion
        target_orientation = target_ee_orientation

        target_joint_angles = self.inverse_kinematics(
            link= self.ee_link, position= target_ee_position, orientation= target_orientation) 

        target_joint_angles = target_joint_angles[:7]       # remove fingers angles
        return target_joint_angles

      

    # Abstract Method
    def get_obs(self):
        """
        Returns the current observation based on the robot parameters. Observation includes EE position, velocity and orientation.

        Returns:
        self.observation (ndarry) --- (ee_position, ee_velocity, ee_orientation)
        """
        ee_position = self.get_ee_position()
        ee_velocity = self.get_ee_velocity()
        ee_orientation = self.get_ee_orientation()              # orientation quaternion for the EE [qx, qy, qz, w] 

        self.observation = np.concatenate((ee_position, ee_velocity, ee_orientation))
        self.observation = np.array(self.observation, dtype=np.float32)

        return self.observation



    # Abstract Method
    def reset(self):
        """
        We don't want to do anything here. The reset function in pandaOrientTask.py takes care of everything
        """
        pass



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



    def get_ee_orientation(self) -> np.ndarray:
        """
        Returns:
        ee_link_orientation (ndarry) --- the orientation quaternion of the end-effector (rx, ry, rz, w)
        """
        ee_link_orientation = self.sim.get_link_orientation(body= self.body_name, link = self.ee_link)
        return ee_link_orientation



