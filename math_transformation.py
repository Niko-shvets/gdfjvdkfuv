import numpy as np
import math
from params import *



unity_body_length = (hip + spine + spine1 + spine2 + neck)[1]



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

def bone_length(bone1: np.array,bone2: np.array)->float:
    """
    Calculate length between 2 points
    """
    
    x1,y1 = bone1[0],bone1[1]
    x2,y2 = bone2[0],bone2[1]
    
    dist = np.sqrt((x2-x1)**2 + (y2 - y1)**2)
    
    return dist

def check_sides(coords:dict,trsh:int)->bool:

    """
    This function compare length of left and right of bones paar: left hand and right hand
    This function allow to understand how correct is  detected skeleton 
    return bool value
    """
    
    count = 0
    valid = True
    
    for left, right in zip(left_side_bones,right_side_bones):
        left_bones_dist = bone_length(coords[left[0]],coords[left[1]])
        right_bones_dist = bone_length(coords[right[0]],coords[right[1]])
        
        length_difference = abs(right_bones_dist - left_bones_dist) 
        # print(left, right,left_bones_dist,right_bones_dist)
        # print(length_difference)
        if length_difference > trsh:
            count += 1
        
    if count >= 2:
        valid = False
    
    return valid


def normalize_body(body_dict:dict, unity_body_length:float)->dict:
    """
    Returns normalized body skeleton. Normalization is in accordance to Unity model
    """
    
    scale_factor = unity_body_length / np.linalg.norm((body_dict['ass'] - body_dict['neck'])) # change -1 to 1 and reverse initial point of other body vectors
    
    new_body_dict = {'ass': np.array([0,0,0], dtype=np.float32)}
    new_body_dict['neck'] = (body_dict['ass'] - body_dict['neck']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['chest'] = (body_dict['ass'] - body_dict['chest']) * scale_factor * np.array([-1,1,-1])
    
    new_body_dict['right_hip'] = (body_dict['ass'] - body_dict['right_hip']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['right_knee'] = (body_dict['right_hip'] - body_dict['right_knee']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['right_ankle'] = (body_dict['right_knee'] - body_dict['right_ankle']) * scale_factor * np.array([-1,1,-1])

    new_body_dict['left_hip'] = (body_dict['ass'] - body_dict['left_hip']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['left_knee'] = (body_dict['left_hip'] - body_dict['left_knee']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['left_ankle'] = (body_dict['left_knee'] - body_dict['left_ankle']) * scale_factor * np.array([-1,1,-1])
    
    new_body_dict['right_shoulder'] = (body_dict['chest'] - body_dict['right_shoulder']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['right_elbow'] = (body_dict['right_shoulder'] - body_dict['right_elbow']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['right_wrist'] = (body_dict['right_elbow'] - body_dict['right_wrist']) * scale_factor * np.array([-1,1,-1])
    
    new_body_dict['left_shoulder'] = (body_dict['chest'] - body_dict['left_shoulder']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['left_elbow'] = (body_dict['left_shoulder'] - body_dict['left_elbow']) * scale_factor * np.array([-1,1,-1])
    new_body_dict['left_wrist'] = (body_dict['left_elbow'] - body_dict['left_wrist']) * scale_factor * np.array([-1,1,-1])
    
    return new_body_dict