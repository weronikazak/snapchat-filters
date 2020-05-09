import cv2
import numpy as np
from imutils import face_utils
import time
import os
import dlib

camera = cv2.VideoCapture(0)

p = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def blurr_face(image):
	(h, w) = image.shape[:2]

	kernel_width = int(w/3.0)
	kernel_height = int(h/3.0)

	if kernel_width % 2 == 0:
		kernel_width -= 1
	else: kernel_width = 5

	if kernel_height % 2 == 0:
		kernel_height -= 1
	else: kernel_height = 5

	return cv2.GaussianBlur(image, (kernel_width, kernel_height), 0)


def pixel_face(image):
	blocks = 16
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks+1, dtype="int")
	ySteps = np.linspace(0, h, blocks+1, dtype="int")

	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]

			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	return image


while True:
	_, frame = camera.read()
	face = 0
	# mask = np.zeros(frame.shape, frame.dtype)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
		# mask[y:y+h, x:x+w] = 255
		face = frame[y:y+h, x:x+w]
		face = blurr_face(face)
		face = pixel_face(face)

		frame[y:y+h, x:x+w] = face

	# (h, w) = frame.shape[:2]
	
	cv2.imshow("frame", frame)
	cv2.imshow("face", face)


	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


cv2.destroyAllWindows()
camera.release()
