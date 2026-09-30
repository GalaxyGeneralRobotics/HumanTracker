# Derived from GalaxyGeneralRobotics/Humanoid-GPT, de8498f (Apache-2.0).
# Evaluation-only extraction; see ../SOURCES.md for modifications.
import os
import numpy as np

FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_GEOMS = [
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
]

RIGHT_FEET_GEOMS = [
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
]

NUM_JOINT = 29

class MotorName:
    LEG_L = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    LEG_R = [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    WAIST = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    ARM_L = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    ARM_R = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R

TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])

DEFAULT_QPOS = np.float32([
	0, 0, 0.78,      # base xyz
	1, 0, 0, 0,     # base quat (w, x, y, z)
	-0.1, 0, 0, 0.3, -0.2, 0,    # left leg
	-0.1, 0, 0, 0.3, -0.2, 0,    # right leg
	0, 0, 0,                      # waist (yaw, roll, pitch but only yaw used)
	0.2, 0.3, 0, 1.28, 0, 0, 0,   # left arm (only pitch, roll, yaw, elbow, wrist_roll used)
	0.2,-0.3, 0, 1.28, 0, 0, 0,   # right arm
])

BASE_KPs = np.float32([
    40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012455 ,
    28.5012455 , 40.17923737, 99.09842682, 40.17923737, 99.09842682,
    28.5012455 , 28.5012455 , 40.17923737, 28.5012455 , 28.5012455 ,
    14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275,
    16.77832794, 16.77832794, 14.25062275, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 16.77832794, 16.77832794
])

BASE_KDs = np.float32([
    2.5578897 , 6.30880165, 2.5578897 , 6.30880165, 1.81444573,
    1.81444573, 2.5578897 , 6.30880165, 2.5578897 , 6.30880165,
    1.81444573, 1.81444573, 2.5578897 , 1.81444573, 1.81444573,
    0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287,
    1.06814146, 1.06814146, 0.90722287, 0.90722287, 0.90722287,
    0.90722287, 0.90722287, 1.06814146, 1.06814146
])

KPT_NAMES = [
    # lower body
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    # upper body
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

OBS_JOINT_NAMES = ACTION_JOINT_NAMES

BASE_KP_KD_SCALE = float(os.environ.get("BASE_KP_KD_SCALE", "1.0"))

JOINT_KP_KD_SCALE = {
    # left leg
    "left_hip_pitch_joint": 1,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 1,
    "left_knee_joint": 1,
    "left_ankle_pitch_joint": 1,
    "left_ankle_roll_joint": 1,
    # right leg
    "right_hip_pitch_joint": 1,
    "right_hip_roll_joint": 1,
    "right_hip_yaw_joint": 1,
    "right_knee_joint": 1,
    "right_ankle_pitch_joint": 1,
    "right_ankle_roll_joint": 1,
    # waist
    "waist_yaw_joint": 1,
    "waist_roll_joint": 1,
    "waist_pitch_joint": 1,
    # left arm
    "left_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 1,
    "left_shoulder_yaw_joint": 1,
    "left_elbow_joint": 1,
    "left_wrist_roll_joint": 1,
    "left_wrist_pitch_joint": 1,
    "left_wrist_yaw_joint": 1,
    # right arm
    "right_shoulder_pitch_joint": 1,
    "right_shoulder_roll_joint": 1,
    "right_shoulder_yaw_joint": 1,
    "right_elbow_joint": 1,
    "right_wrist_roll_joint": 1,
    "right_wrist_pitch_joint": 1,
    "right_wrist_yaw_joint": 1,
}

JOINT_KP_KD_SCALE = np.array(list(JOINT_KP_KD_SCALE.values()), dtype=np.float32)

KPs = BASE_KPs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE ** 2

KDs = BASE_KDs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE

ACTION_SCALE = TORQUE_LIMIT / KPs
