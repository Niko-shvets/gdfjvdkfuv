import face_alignment
import cv2
import numpy as np


face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

#initializie NN for landmarks predictions
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)


def face_landmarks(image: np.array):
    landmarks = {}
    
    preds = fa.get_landmarks(image)[-1]
    
    landmarks['left_eyebrow'] = preds[17:22]
    landmarks['right_eyebrow'] = preds[22:27]
    landmarks['nose'] = preds[27:31][np.argmax(preds[27:31][:,1])]
    landmarks['left_eye'] = np.mean(preds[36:42],axis=0)
    landmarks['right_eye'] = np.mean(preds[42:48],axis=0)
    landmarks['lips'] = preds[48:60]
    
    return landmarks
    