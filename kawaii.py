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

scale = 10
padding = 10
x_offset = y_offset = 10

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)


def create_bigger_eye(frame, rang, right=False):
    (xx, yy, w, h) = cv2.boundingRect(np.array([rang]))

    roi = frame[yy-padding:yy+padding + h, xx-padding:xx+padding + w]
    bigger_eye = cv2.resize(roi, (int(roi.shape[1]*1.2), int(roi.shape[0]*1.2)))

    b_e_pos_y = int(y - 1.5*padding)
    b_e_pos_x = x - padding

    if right == True:
        b_e_pos_x -= padding

    return bigger_eye, b_e_pos_y, b_e_pos_x


def blur_edges(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh, bin_red = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    return opening


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        i, j = l_eye[0], l_eye[1]

        for (x, y) in shape[i:j]:

            bigger_eye, a, b = create_bigger_eye(frame, shape[i:j])
            
        frame[a:a + bigger_eye.shape[0], b:b+bigger_eye.shape[1]] = bigger_eye
            
        i, j = r_eye[0], r_eye[1]

        for (x, y) in shape[i:j]:
            bigger_eye, a, b = create_bigger_eye(frame, shape[i:j], True)
        
        frame[a:a + bigger_eye.shape[0], b:b+bigger_eye.shape[1]] = bigger_eye

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()