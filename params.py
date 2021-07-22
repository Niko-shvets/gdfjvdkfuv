import numpy as np

left_side_bones = [['left_shoulder','left_elbow'],['left_elbow','left_wrist'],
                  ['left_hip','left_knee'],['left_knee','left_ankle']]

right_side_bones = [['right_shoulder','right_elbow'],['right_elbow','right_wrist'],
                  ['right_hip','right_knee'],['right_knee','right_ankle']]

# for unity_body_length calculation
hip = np.array([0,0,0])
spine = np.array([0, 0.1018159, 0.001315209])
spine1 = np.array([0, 0.1008345, -0.01000804])
spine2 = np.array([0, 0.09100011, -0.01373417])
neck = np.array([0, 0.1667167, -0.02516168])
left_up_leg = np.array([-0.08207782, -0.06751714, -0.01599556])
left_knee = np.array([0, -0.4437047, 0.002846426])
left_ankle = np.array([0, -0.4442787, -0.02982191])
left_shoulder = np.array([-0.1059237, -0.005245829, -0.0223212])
left_elbow = np.array([-0.2784152, 0, 0])
left_wrist = np.array([-0.2832884, 0, 0])
unity_ass_position = -4 # current camera position; it will be used to adjust body position at particular frame

# for coordinate extractor file
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
num_layers = 50
heads =  {'hm': 16, 'depth': 16}
device = 'cpu'

edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

COCO_KEYPOINT_INDEXES = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'ass',
    7: 'chest',
    8: 'neck',
    9: 'head',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist',
}

weight_path_3d = 'MODELS/fusion_3d_var.pth'


# For second model, detection quantaty of people and image boarders break

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

NUM_KPTS = 17