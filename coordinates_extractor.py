from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from model import create_model
from utils.image import get_affine_transform#, transform_preds for 2d
from utils.eval import  get_preds_3d#,get_preds for 2d
from models.msra_resnet import get_pose_net


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

# Neural Network
model = get_pose_net(num_layers, heads)
checkpoint = torch.load(
  'models/fusion_3d_var.pth', map_location=lambda storage, loc: storage)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()


def coords_create(image_name:str)->dict:
    # read image
    image = cv2.imread(image_name)
    
    
    #image preprocessing
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
      c, s, 0, [256, 256])
    inp = cv2.warpAffine(image, trans_input, (256, 256),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(device)
    #coordinates prediction
    out = model(inp)[-1]

    ## 2D coordinates
    # pred = get_preds(out['hm'].detach().cpu().numpy())[0]
    # pred = transform_preds(pred, c, s, (64, 64))
    ## 3D coordinates
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]
    
    
    
    coordinates = {}

    for label, coordinate in zip(COCO_KEYPOINT_INDEXES.values(),pred_3d):

        coordinates[label] = coordinate
    
    return coordinates