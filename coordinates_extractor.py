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
from utils.image import get_affine_transform, transform_preds# for 2d
from utils.eval import  get_preds_3d,get_preds# for 2d
from models.msra_resnet import get_pose_net
from params import *



# Neural Network
model = get_pose_net(num_layers, heads)
checkpoint = torch.load(
  weight_path_3d, map_location=lambda storage, loc: storage)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()


def dict_create(skeleton_dict: dict, preds:np.array)->dict:
    
    coordinates = {}

    for label, coordinate in zip(skeleton_dict.values(),preds):

        coordinates[label] = coordinate
        
    return coordinates
    
    


def coords_create(image:np.array)->dict:
    # read image
    # image = cv2.imread(image_name)
    
    
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
    pred_2d = get_preds(out['hm'].detach().cpu().numpy())[0]
    pred_2d = transform_preds(pred_2d, c, s, (64, 64))
    ## 3D coordinates
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]
    
    
    
    three_d_coords = dict_create(COCO_KEYPOINT_INDEXES,pred_3d)
    two_d_coords = dict_create(COCO_KEYPOINT_INDEXES,pred_2d)
    
    return three_d_coords,two_d_coords