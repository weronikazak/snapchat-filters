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
lens_choice = random.choice(colours)

offset = 2

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
# detector_params.minArea = 800
detector_params.maxArea = 1500
detectorB = cv2.SimpleBlobDetector_create(detector_params)


def detect_keypoints(img):
    # img = cv2.resize(img, (img.shape[1]*5, img.shape[0]*5), interpolation = cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_roi = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray_roi, 55, 255, cv2.THRESH_BINARY)

    thresh = cv2.erode(thresh, None, iterations=2) #1
    # thresh = cv2.dilate(thresh, None, iterations=8) #2
    thresh = cv2.medianBlur(thresh, 5) #3

    M = cv2.moments(thresh)

    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except:
        cX, cY = 0, 0

    return (cX, cY)



while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    frame_2 = frame.copy()

    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(frame.shape, frame.dtype)

    l_eye, r_eye = 0, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    lens =  cv2.imread("images/lenses/" + lens_choice + ".png", -1)


    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # eyes

        l_eye, r_eye = shape[36:42], shape[42:48]

        (lx, ly, lw, lh) = cv2.boundingRect(l_eye)
        (rx, ry, rw, rh) = cv2.boundingRect(r_eye)

        l_eye = frame[ly-offset:ly+lh+offset, lx+offset:lx-offset+lw]
        r_eye = frame[ry-offset:ry+rh+offset, rx+offset:rx-offset+rw]

        center_left = detect_keypoints(l_eye)
        center_right = detect_keypoints(r_eye)

        eye_size = int(lh*0.4)

        lens = cv2.resize(lens, (eye_size, eye_size), interpolation = cv2.INTER_AREA)

        alpha_s = lens[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        half_eye_size = int(eye_size/2)

        x1 = center_left[0] - half_eye_size
        x2 = center_left[0] + half_eye_size
        y1 = center_left[1] + half_eye_size
        y2 = center_left[1] - half_eye_size

        print(x1, x2, y1, y2)

        l_eye = cv2.addWeighted(l_eye, 0.4, lens, 0.6, 0)

        # for c in range(0, 3):
        #     l_eye[y1:y2, x1:x2, c] = (alpha_s * lens[:, :, c] + 
        #                             alpha_l * l_eye[y1:y2, x1:x2, c])


        # l_eye = cv2.circle(l_eye, center_left, eye_size, (0, 255, 0), -1)
        # r_eye = cv2.circle(r_eye, center_right, eye_size, (0, 255, 0), -1)
        
        frame[ly-offset:ly+lh+offset, lx+offset:lx-offset+lw] = l_eye
        # frame[ry-offset:ry+rh+offset, rx+offset:rx-offset+rw] = r_eye


    cv2.imshow("frame", frame)
    # cv2.imshow("mask", l_eye)
    cv2.imshow("r_eye", lens)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
