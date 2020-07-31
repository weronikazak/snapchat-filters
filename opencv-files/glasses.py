import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

camera = cv2.VideoCapture(0)

p = "../data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


while True:
    ret, frame = camera.read()
    glasses = cv2.imread("../images/glasses.png", -1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        glasses_width = int(abs(shape[36][0] - shape[32][0]) * 4)
        glasses_height = int(glasses_width * 0.7)

        (glasses_x, glasses_y) = shape[30]
        glasses_y -= 20

        half_width = int(glasses_width/2.0)
        half_height = int(glasses_height/2.0)

        y1, y2 = glasses_y - half_height, glasses_y + half_height
        x1, x2 = glasses_x - half_width, glasses_x + half_width

        glasses = cv2.resize(glasses, (half_width*2, half_height*2), interpolation = cv2.INTER_AREA)

        alpha_s = glasses[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + 
                                    alpha_l * frame[y1:y2, x1:x2, c])

        # -------------
        # -------------


    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
