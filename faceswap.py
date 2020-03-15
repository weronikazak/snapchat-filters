import dlib
import cv2
import numpy as np

img2 = cv2.imread("jim_carrey.jpg", 0)

jaw = [0, 17]
r_eyebrow, l_eyebrow = [18, 22], [23, 27]
nose = [28, 36]
r_eye, l_eye = [37, 42], [43, 48]
mouth = [49, 68]

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    rects = detector(gray, 0)
    landmarks = []
    for (i, rect) in enumerate(rects):
        landmarks.append(rect)
    
    points = np.array(rect, np.int32)
    convexhull = cv2.convexHull(points)

    cv2.fillConvexPoly(mask, convexhull, 255)

    face = cv2.bitwise_and(frame, frame, mask=mask)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()