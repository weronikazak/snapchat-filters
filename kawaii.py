import cv2
import numpy as np
# import tensorflow as tf
import dlib
from imutils import face_utils

jaw = [0, 17]
r_eyebrow, l_eyebrow = [18, 22], [23, 27]
nose = [28, 36]
r_eye, l_eye = [37, 42], [43, 48]
mouth = [49, 68]

padding = 10
x_offset = y_offset = 10

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)



def create_bigger_eye(frame, mask, rang, right=False):
    mask = np.zeros(frame.shape, frame.dtype)

    scale = 1.3
    # get the eye
    (xx, yy, w, h) = cv2.boundingRect(np.array([rang]))
    roi = frame[yy-padding:yy+padding + h, xx-padding:xx+padding + w]
    bigger_eye = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)))
    
    x1 = yy-padding
    x2 = yy+padding + h
    y1 = xx-padding
    y2 = xx+padding + w

    # rectangle ver
    mask[x1:x2, y1:y2] = (255, 255, 255)
    
    center_y = int((x2 + x1) / 2)
    center_x = int((y2 + y1) / 2)

    # ellipse ver
    # mask = cv2.ellipse(mask, (center_x, center_y), (int(w*scale), int(h*scale)), 0, 0, 360, (255, 255, 255), -1)

    frame[x1:x1 + bigger_eye.shape[0], y1:y1+bigger_eye.shape[1]] = bigger_eye 

    return frame, mask, center_y, center_x



def blur_edges(src, dest, mask, center):
    out = cv2.seamlessClone(src, dest, mask, center, cv2.NORMAL_CLONE)
    return out


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_2 = frame.copy()
    mask = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        i, j = l_eye[0], l_eye[1]

        frame, mask, a, b = create_bigger_eye(frame, mask, shape[i:j])
        frame_2 = blur_edges(frame, frame_2, mask, (b, a))

        i, j = r_eye[0], r_eye[1]

        bigger_eye, mask, a, b = create_bigger_eye(frame, mask, shape[i:j], True)
        frame_2 = blur_edges(frame, frame_2, mask, (b, a))

    cv2.imshow('frame', frame_2)
    cv2.imshow("mask.jpg", mask)
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()