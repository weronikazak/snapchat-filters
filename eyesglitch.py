import cv2
import dlib
from imutils import face_utils, resize, translate
import numpy as np

camera = cv2.VideoCapture(0)

p = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

EYES_LENGTH = 10
eyelist = []

while True:
    ret, frame = camera.read()

    # frame = resize(frame, 800)
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

        eye_layer = cv2.bitwise_and(frame, frame, eye_mask)

        x, y, w, h = cv2.boundingRect(eye_mask)

        if len(eyelist) >= EYES_LENGTH:
            eyelist.pop(0)
        
        eyelist.append([x, y])


    cv2.imshow("frame", frame)
    cv2.imshow("mask", eye_mask)


    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()