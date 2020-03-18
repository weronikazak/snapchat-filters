import cv2
import numpy as np
# import tensorflow as tf
import dlib
from imutils import face_utils

jaw = [0, 17]
r_eyebrow, l_eyebrow = [18, 22], [23, 27]
nose = [30, 36]
r_eye, l_eye = [37, 42], [43, 48]
mouth = [49, 68]

padding = 10
scale = 1.3

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)



def create_bigger_eye(frame, frame_2, rang, right=False):
    (x, y, w, h) = cv2.boundingRect(np.array([rang]))

    x1 = y-padding
    x2 = y+padding + h
    y1 = x-padding
    y2 = x+padding + w

    # create roi
    roi = frame[x1:x2, y1:y2]
    bigger_eye = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)))

    # create mask
    mask = np.zeros(frame.shape, frame.dtype)
    mask[x1:x2, y1:y2] = (255, 255, 255)

    # center = (rang[3][0],rang[3][1])
    
    center_y = int((x2 + x1) / 2)
    center_x = int((y2 + y1) / 2)

    frame[x1:x1 + bigger_eye.shape[0], y1:y1+bigger_eye.shape[1]] = bigger_eye 

    out = cv2.seamlessClone(frame, frame_2, mask, (center_x, center_y), cv2.NORMAL_CLONE)

    return frame, out


def create_smaller_nose(frame, frame_2, rang, right=False):
    (x, y, w, h) = cv2.boundingRect(np.array([rang]))

    x1 = y-padding
    x2 = y+padding + h
    y1 = x-padding
    y2 = x+padding + w

    # create roi
    roi = frame[x1:x2, y1:y2]
    smaller_nose = cv2.resize(roi, (int(roi.shape[1]*0.8), int(roi.shape[0]*0.8)))

    s_x = int(smaller_nose.shape[0]/2)
    s_y = int(smaller_nose.shape[1]/2)

    x1 = y - s_x
    x2 = y + s_x
    y1 = x - s_y
    y2 = x + s_y

    # create mask
    mask = np.zeros(frame.shape, frame.dtype)
    mask[x1:x2, y1:y2] = (255, 255, 255)

    center = (rang[0][0],rang[0][1])
    
    smaller_nose = cv2.resize(smaller_nose, (2 * s_y, 2 * s_x))
    frame[x1:x2, y1:y2] = smaller_nose

    out = cv2.seamlessClone(frame, frame_2, mask, center, cv2.NORMAL_CLONE)

    return frame, out


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    frame_2 = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        i, j = l_eye[0], l_eye[1]
        frame, frame_2 = create_bigger_eye(frame, frame_2, shape[i:j])

        i, j = r_eye[0], r_eye[1]
        frame, frame_2 = create_bigger_eye(frame, frame_2, shape[i:j])

        i, j = nose[0], nose[1]
        frame, frame_2 = create_smaller_nose(frame, frame_2, shape[i:j])

    cv2.imshow('frame', frame_2)
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()