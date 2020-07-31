import cv2
import numpy as np
import dlib
from imutils import face_utils

camera = cv2.VideoCapture(0)

p = "../data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

offset = 4
scale = 1.3

_,  out = camera.read()


while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(frame.shape, frame.dtype)
    frame_2 = frame.copy()

    eye_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_mask = np.zeros(frame.shape, frame.dtype)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        l_eye, r_eye = shape[36:42], shape[42:48]

        (lx, ly, lw, lh) = cv2.boundingRect(l_eye)
        (rx, ry, rw, rh) = cv2.boundingRect(r_eye)

        l_eye = frame[ly-offset:ly+lh+offset, lx-offset:lx+lw+offset]
        r_eye = frame[ry-offset:ry+rh+offset, rx-offset:rx+rw+offset]

        center_ly = lx + int(lw / 2)
        center_lx = ly + int(lh / 2)
        center_ry = rx + int(rw / 2)
        center_rx = ry + int(rh / 2)

        # --------------
        # SCALING COORDS
        # --------------

        ly_scaled = int((l_eye.shape[1]*scale)/2)
        lx_scaled = int((l_eye.shape[0]*scale)/2)
        ry_scaled = int((r_eye.shape[1]*scale)/2)
        rx_scaled = int((r_eye.shape[0]*scale)/2)

        l_eye = cv2.resize(l_eye, (ly_scaled*2, lx_scaled*2), interpolation = cv2.INTER_AREA)
        r_eye = cv2.resize(r_eye, (ry_scaled*2, rx_scaled*2), interpolation = cv2.INTER_AREA)
    
        # ---------------
        # SETTLE ON FRAME
        # ---------------

        frame[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = l_eye
        mask[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = 255
        frame[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = r_eye
        mask[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = 255

        final_center_x = int(np.mean([center_lx, center_rx]))
        final_center_y = int(np.mean([center_ly, center_ry]))

        out = cv2.seamlessClone(frame_2, frame, eye_mask, (final_center_y, final_center_x), cv2.NORMAL_CLONE)

        
    cv2.imshow('frame', frame)    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()