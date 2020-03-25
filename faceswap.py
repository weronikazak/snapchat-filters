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

img = cv2.imread("images/woman.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask_img = np.zeros(gray_img.shape, gray_img.dtype)
landmark_points2 = []

rects = detector(gray_img, 0)
for rect in rects:
    shape = predictor(gray_img, rect)
    shape = face_utils.shape_to_np(shape)
    for n in range(68):
        x = shape.part(n).x
        y = shape.part(n).y
        landmark_points2.append((x, y))

    points2 = np.array(landmark_points2, np.int32)

    convex2 = cv2.convexHull(shape)
    cv2.fillPoly(mask_img, [convex2], 255)
    face1 = cv2.bitwise_and(img, img, mask=mask_img)

    id_triangles = delaunay_traingle(convex2, landmark_points2)

#------------

def delaunay_traingle(convexHull, points):
    rect = cv2.boundingRect(convexHull)

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    id_triangles = []
    for t in triangles:
        A, B, C = (t[0], t[1]), (t[2], t[3]), (t[4], t[5]) # xs and ys of main points

        id_A = np.where((points == A))[0]
        id_A = search_id(id_A)

        id_B = np.where((points == B))[0]
        id_A = search_id(id_A)

        id_C = np.where((points == C))[0]
        id_A = search_id(id_A)
        
        if id_A != None and id_B != None and id_C != None:
            tr = [id_A, id_B, id_C]
            id_triangles.append(tr)

    return id_triangles


def search_id(array):
    id_ = None
    for i in array[0]:
        id_ = i
        break
    return id_


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

        tr_indexes = delaunay_traingle(convexhull, landmark_points, gray)

        lines_mask = np.zeros(gray.shape, gray.dtype)
        lines_img = np.zeros(frame.shape, frame.dtype)

        for tr_id in id_triangles:
            tr_A1 = landmark_points[tr_id[0]]
            tr_B1 = landmark_points[tr_id[1]]
            tr_C1 = landmark_points[tr_id[2]]
            triangle = np.array([tr_A1, tr_B1, tr_C1], np.int32)

            (x, y, w, h) = cv2.boundingRect(triangle)

            crop_triangle1 = frame[y:y+h, x+x+w]
            crop_mask1 = np.zeros((h, w), np.uint8)

            pts1 = np.array([[tr_A1[0] - x, tr_A1[1] - y],
                            [tr_B1[0] - x, tr_B1[1] - y],
                            [tr_C1[0] - x, tr_C1[1] - y]], np.int32)

            cv2.fillConvexPoly(crop_mask1, pts1, 255)


            tr_A2 = landmark_points[tr_id[0]]
            tr_B2 = landmark_points[tr_id[1]]
            tr_C2 = landmark_points[tr_id[2]]
            triangle = np.array([tr_A1, tr_B1, tr_C1], np.int32)

            (x, y, w, h) = cv2.boundingRect(triangle)

            crop_triangle1 = frame[y:y+h, x+x+w]
            crop_mask1 = np.zeros((h, w), np.uint8)

            pts1 = np.array([[tr_A1[0] - x, tr_A1[1] - y],
                            [tr_B1[0] - x, tr_B1[1] - y],
                            [tr_C1[0] - x, tr_C1[1] - y]], np.int32)

            cv2.fillConvexPoly(crop_mask1, pts1, 255)

    cv2.imshow("frame", gray)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()