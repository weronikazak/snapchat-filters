import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import random
import math

camera = cv2.VideoCapture(0)

p = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

colours = ["blue", "red", "black", "brown", "green", "purple", "vampire"]
eye_lens = cv2.imread("images/lenses/" + random.choice(colours) + ".png")

offset = 4


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    frame_2 = frame.copy()

    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(frame.shape, frame.dtype)

    l_eye, r_eye = 0, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # eyes

        l_eye, r_eye = shape[36:42], shape[42:48]

        (lx, ly, lw, lh) = cv2.boundingRect(l_eye)
        (rx, ry, rw, rh) = cv2.boundingRect(r_eye)

        l_eye = gray[ly-offset:ly+lh+offset, lx-offset:lx+lw+offset]
        r_eye = gray[ry-offset:ry+rh+offset, rx-offset:rx+rw+offset]

        center_ly = lx + int(lw / 2)
        center_lx = ly + int(lh / 2)
        center_ry = rx + int(rw / 2)
        center_rx = ry + int(rh / 2)

       
        l_eye = cv2.resize(l_eye, (l_eye.shape[1]*5, l_eye.shape[0]*5), interpolation = cv2.INTER_AREA)
        
        gray_roi = cv2.GaussianBlur(l_eye, (7, 7), 0)
        _, thresh = cv2.threshold(gray_roi, 55, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(l_eye, contours, -1, (0, 255, 9), 1)
        
        # x_mean = [50]
        # y_mean = [50]
        # for cnt in contours:
        #     # area = cv2.contourArea(cnt)
        #     (x, y, w, h) = cv2.boundingRect(cnt)
        #     x_mean.append(x)
        #     y_mean.append(y)

        # x_mean = int(np.mean(x_mean))
        # y_mean = int(np.mean(y_mean))

        # cv2.circle(l_eye, (x_mean, y_mean), 10, (0, 255, 0), -1)

        

    cv2.imshow("frame", frame)
    cv2.imshow("mask", l_eye)
    cv2.imshow("thresh", thresh)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
