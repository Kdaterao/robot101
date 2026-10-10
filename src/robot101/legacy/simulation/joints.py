import numpy as np


'''
    Helpful functions for 


'''


#============================
# JOINT NAMES AND OFFSETS
#============================


# ARM
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# GRIPPER
GRIPPER_JOINT = "gripper" 

# ALL 
MOTOR_NAMES = (*ARM_JOINTS, GRIPPER_JOINT)



# JOINT OFFSETS FOR DISCREPANCY BETWEEN IRL AND SIM
JOINT_SIGN = {name: 1.0 for name in ARM_JOINTS}
JOINT_OFFSET_DEG = {
    "shoulder_pan": 15.00,
    "shoulder_lift": 0.46,
    "elbow_flex":  -6.18,
    "wrist_flex":  -1.58,
    "wrist_roll": 100.31,
}





#===============================================
# CONVERSION FUNCTIONS (REAL TO SIM AND SIM TO REAL)
#===============================================

#---- GRIPPER ---------


# GRIPPER CONVERSION FUNCTIONS (HELPERS TO THE FUNCTIONS BELOW)
GRIPPER_CLOSED_RAD = -0.17453297762778586
GRIPPER_OPEN_RAD = 1.7453291995659765

def gripper_real_to_sim(value: float) -> float:
    t = float(np.clip(value / 100.0, 0.0, 1.0))
    return GRIPPER_CLOSED_RAD + t * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def gripper_sim_to_real(q: float) -> float:
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    return float(np.clip((q - GRIPPER_CLOSED_RAD) / span * 100.0, 0.0, 100.0))


#------ TOTAL ----------

# REAL TO SIM CONVERSION FUNCTIONS 

def real_to_sim(pose: dict) -> dict[str, float]:
    """
        LEROBOT --> MUJUCO CONVERSTION
        
        Converts lerobot unit of degrees to mujcoo unit of radians 
        Converts lerobot unit for gripper (percentage) to unit of radians 
    """
    sim = {
        name: JOINT_SIGN[name] * float(np.deg2rad(pose[f"{name}.pos"]))
        + float(np.deg2rad(JOINT_OFFSET_DEG[name]))
        for name in ARM_JOINTS
    }
    sim[GRIPPER_JOINT] = gripper_real_to_sim(float(pose["gripper.pos"]))
    return sim


def sim_to_real(qpos: dict[str, float]) -> dict[str, float]:
    """
        MUJUCO --> LEROBOT CONVERSTION

        Converts mujcoo unit of radians to lerobot unit of degrees 
        Converts mujcoo unit for gripper (percentage) to unit of radians 
    """
    real = {
        f"{name}.pos": float(
            np.rad2deg(qpos[name] - np.deg2rad(JOINT_OFFSET_DEG[name]))
            / JOINT_SIGN[name]
        )
        for name in ARM_JOINTS
    }
    real["gripper.pos"] = gripper_sim_to_real(qpos[GRIPPER_JOINT])
    return real
