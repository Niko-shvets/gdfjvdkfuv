import numpy as np
import math


def point_angle(p1:np.array, p2:np.array)->float:
    """
    calcualte angle between 2 points
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])

    return math.degrees((ang1 - ang2) % (2 * np.pi))


def get_angle(a:np.array,b:np.array,c:np.array)->float:

    """
    calculate angle between 3 points via vectors
    """
    
    ba = a - b
    cb = b - c
    
    ba_angle = math.atan2(ba[0], ba[1])
    cb_anlge = math.atan2(cb[0], cb[1])
    
    cba_angle = cb_anlge - ba_angle
    cba_angle_deg = math.degrees(cba_angle)
    
    return cba_angle_deg

def human_angles(coords:dict)->dict:
    
    angles = {}
    
    angles['left_shoulder'] = get_angle(coords['left_elbow'],coords['left_shoulder'],coords['left_hip']) - 90
    angles['right_shoulder'] = get_angle(coords['right_elbow'],coords['right_shoulder'],coords['right_hip']) + 90
    
    angles['left_elbow'] = get_angle(coords['left_wrist'],coords['left_elbow'],coords['left_shoulder']) 
    angles['right_elbow'] = get_angle(coords['right_wrist'],coords['right_elbow'],coords['right_shoulder'])
    
        
    angles['left_hip_body'] = get_angle(coords['left_knee'],coords['left_hip'],coords['left_shoulder'])
    angles['right_hip_body'] = get_angle(coords['right_knee'],coords['right_hip'],coords['right_shoulder'])
        
    
    angles['left_knee'] = get_angle(coords['left_ankle'],coords['left_knee'],coords['left_hip'])
    angles['right_knee'] = get_angle(coords['right_ankle'],coords['right_knee'],coords['right_hip'])
    
    angles['head'] = point_angle(coords['head'][:2],coords['neck'][:2])
        

    return angles


def get_forward_vec(a:np.array,b:np.array,c:np.array)->np.array:
    """
    Returns normalized vector of a plane given by 3 points
    """
    v1 = b - a
    v2 = c - a
    
    forward_vec = np.cross(v2, v1)
    forward_vecn = forward_vec / np.linalg.norm(forward_vec)
    
    return forward_vecn

def get_angles_oxoy(vec:np.array)->float:
    # initial position of a vector is [0,0,1]. It leads to additional rotation by +- 90 degrees
    
    angle_ox = math.degrees(math.atan2(vec[2], vec[1])) - 90    
    angle_oy = math.degrees(math.atan2(vec[2], vec[0])) + 90
    
    return angle_ox, angle_oy