import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

camera = cv2.VideoCapture(0)

p = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


while True:
    ret, frame = camera.read()
    dog_nose = cv2.imread("images/nose.png", -1)
    dog_ears = cv2.imread("images/ears.png", -1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # ------------
        #     EARS
        # ------------

        ears_width = int(abs(shape[0][0] - shape[16][0]) * 1.5)
        ears_height = int(ears_width * 0.4)

        ears_x = int((shape[22][0] + shape[23][0])/2)
        ears_y = shape[20][1] - 50

        half_width = int(ears_width/2.0)
        half_height = int(ears_height/2.0)

        y1, y2 = ears_y - half_height, ears_y + half_height
        x1, x2 = ears_x - half_width, ears_x + half_width

        dog_ears = cv2.resize(dog_ears, (half_width*2, half_height*2), interpolation = cv2.INTER_AREA)

        alpha_s = dog_ears[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * dog_ears[:, :, c] + 
                                    alpha_l * frame[y1:y2, x1:x2, c])

        # ------------
        #     NOSE
        # ------------

        nose_width = int(abs(shape[36][0] - shape[32][0]) * 1.7)
        nose_height = int(nose_width * 0.7)

        (nose_x, nose_y) = shape[30]

        half_width = int(nose_width/2.0)
        half_height = int(nose_height/2.0)

        y1, y2 = nose_y - half_height, nose_y + half_height
        x1, x2 = nose_x - half_width, nose_x + half_width

        dog_nose = cv2.resize(dog_nose, (half_width*2, half_height*2), interpolation = cv2.INTER_AREA)

        alpha_s = dog_nose[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * dog_nose[:, :, c] + 
                                    alpha_l * frame[y1:y2, x1:x2, c])

        # -------------
        # -------------


    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
