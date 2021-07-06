from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
from params import *

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
parser.add_argument('--cfg', type=str, default='inference-config.yaml')
parser.add_argument('--image',type=str)
parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

args = parser.parse_args("--image ttt.jpg".split())

args.modelDir = ''
args.logDir = ''
args.dataDir = ''
args.prevModelDir = ''






CTX =  torch.device(device)
print('load model')
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


update_config(cfg, args)

box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
box_model.to(CTX)
box_model.eval()

pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)

# if cfg.TEST.MODEL_FILE:
    # print('=> loading model from {}'.format('models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'))
pose_model.load_state_dict(torch.load('src/lib/models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth',map_location = 'cpu'), strict=False)


pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
pose_model.to(CTX)
pose_model.eval()
# print(pose_model)





def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
   
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale




def solver(image_bgr:np.array):
   
    # else:
    #     print('expected model defined in config at TEST.MODEL_FILE')

    


    # image_bgr = cv2.imread(image)

    print('start')
    last_time = time.time()
    image = image_bgr[:, :, [2, 1, 0]]
    shape = image.shape

    input = []
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
    input.append(img_tensor)

    # object detection box
    pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
    predictions = []
    # pose estimation
    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            predictions.append(pose_preds)
            # if len(pose_preds)>=1:
            #     for kpt in pose_preds:
            #         draw_pose(kpt,image_bgr)
                    
    
    # if save:
    #     save_path = 'output.jpg'
    #     cv2.imwrite(save_path,image_bgr)
    #     print('the result image has been saved as {}'.format(save_path))

    # if show:
    #     cv2.imshow('demo',image_bgr)
    #     if cv2.waitKey(0) & 0XFF==ord('q'):
    #         cv2.destroyAllWindows()
    
    return pred_boxes, pose_preds, shape


# def coordinates_update(pose_preds):
    
#     coordinates = {}

#     for label, coordinate in zip(COCO_KEYPOINT_INDEXES.values(),pose_preds):

#         coordinates[label] = coordinate
        
#     coordinates['ass'] = (coordinates['right_hip'] + coordinates['left_hip'])/2
#     coordinates['chest'] = (coordinates['right_shoulder'] + coordinates['left_shoulder'])/2


#     new_coordinates = {}

#     new_coordinates['left_hip'] = coordinates['ass'] - coordinates['left_hip']
#     new_coordinates['right_hip'] = coordinates['ass'] - coordinates['right_hip']

#     new_coordinates['left_knee'] = coordinates['ass'] - coordinates['left_knee']
#     new_coordinates['right_knee'] = coordinates['ass'] - coordinates['right_knee']


#     new_coordinates['left_ankle'] = coordinates['ass'] - coordinates['left_ankle']
#     new_coordinates['right_ankle'] = coordinates['ass'] - coordinates['right_ankle']


#     new_coordinates['chest'] = coordinates['ass'] - coordinates['chest']

#     new_coordinates['left_shoulder'] = coordinates['ass'] - coordinates['left_shoulder']
#     new_coordinates['right_shoulder'] = coordinates['ass'] - coordinates['right_shoulder']

#     new_coordinates['left_elbow'] = coordinates['ass'] - coordinates['left_elbow']
#     new_coordinates['right_elbow'] = coordinates['ass'] - coordinates['right_elbow']

#     new_coordinates['left_wrist'] = coordinates['ass'] - coordinates['left_wrist']
#     new_coordinates['right_wrist'] = coordinates['ass'] - coordinates['right_wrist']
    
#     new_coordinates['ass'] = np.array([0,0])
        
#     return new_coordinates,coordinates

def initial_check(boxes, coord, shape):
    
    valid = True
    
    if any(coord[0][:,1] > shape[0]) or any(coord[0][:,0] > shape[1]):
        
        valid = False
    if len(boxes) > 1:
        valid = False
        
    return valid