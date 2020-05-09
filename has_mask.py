# import cv2
# import dlib
# import numpy as np

# camera = cv2.VideoCapture(0)

# nose_cascade = "cascades/haarcascade_mcs_nose.xml"
# eyes_cascade = "cascades/haarcascade_eye_tree_eyeglasses.xml"

# while True:
#     frame, _ = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break




# cv2.destroyAllWindows()
# camera.release()