import dlib
import cv2
import numpy as np
from imutils import face_utils


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)

# -----------
# FIRST IMAGE
# -----------

img2 = cv2.imread("images/woman.jpg")
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask_img2 = np.zeros(gray_img2.shape, gray_img2.dtype)

rects = detector(gray_img2, 0)
for rect in rects:
    shape = predictor(gray_img2, rect)
    shape = face_utils.shape_to_np(shape)

    convex2 = cv2.convexHull(shape)
    cv2.fillPoly(mask_img2, [convex2], 255)
    face1 = cv2.bitwise_and(img2, img2, mask=mask)

    delaunay_traingle(convex2, shape, )

#------------

def delaunay_traingle(convexHull, points, frame):
    rect = cv2.boundingRect(convexHull)

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)


    for t in triangles:
        A, B, C = (t[0], t[1]), (t[2], t[3]), (t[4], t[5]) # xs and ys of main points

        id_triangles = []

        cv2.line(frame, A, B, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, B, C, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, A, C, (255, 255, 255), 1, cv2.LINE_AA, 0)
        id_triangles.append([A, B, C])
    return frame



while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    faces = detector(gray, 0)
    for face in faces:
        landmark = predictor(gray, face)
        landmark_points = []
        for n in range(68):
            x = landmark.part(n).x
            y = landmark.part(n).y
            landmark_points.append((x, y))
    
        points = np.array(landmark_points, np.int32)
        convexhull = cv2.convexHull(points)

        cv2.fillConvexPoly(mask, convexhull, 255)

        face = cv2.bitwise_and(frame, frame, mask=mask)

        gray = delaunay_traingle(convexhull, landmark_points, gray)

    cv2.imshow("frame", gray)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()