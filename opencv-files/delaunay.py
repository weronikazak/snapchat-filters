import dlib
import cv2
import numpy as np

# img2 = cv2.imread("jim_carrey.jpg", 0)

jaw = [0, 17]
r_eyebrow, l_eyebrow = [18, 22], [23, 27]
nose = [28, 36]
r_eye, l_eye = [37, 42], [43, 48]
mouth = [49, 68]

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)


def delaunay_traingle(convexHull, points, frame):
    rect = cv2.boundingRect(convexHull)

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)


    for t in triangles:
        A, B, C = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])

        # id_triangles = []

        # if rect_contains(r, A) and rect_contains(r, B) and rect_contains(r, C):
        cv2.line(frame, A, B, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, B, C, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, A, C, (255, 255, 255), 1, cv2.LINE_AA, 0)
        #     triangle = [A, B, C]
        #     id_triangles.append(triangle)
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