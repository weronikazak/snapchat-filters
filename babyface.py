import cv2
import numpy as np
import dlib
from imutils import face_utils


padding = 10
scale = 1.3

p = "data/shape_predictor_68_face_landmarks.dat"
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


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    # frame_2 = frame.copy()
    eye_layer = np.zeros(frame.shape, frame.dtype)
    eye_mask = cv2.cvtColor(eye_layer, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        l_eye, r_eye = shape[36:42], shape[42:48]

        cv2.fillPoly(eye_mask, [l_eye], 255)
        cv2.fillPoly(eye_mask, [r_eye], 255)
        # frame, frame_2 = create_bigger_eye(frame, frame_2, shape[i:j])

        # frame, frame_2 = create_bigger_eye(frame, frame_2, shape[i:j])

        
    cv2.imshow('frame', frame)
    cv2.imshow('eye', eye_mask)
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()