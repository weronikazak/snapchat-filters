import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
	ret, frame = camera.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 3)
	edges = cv.Canny(img,gray.shape[0],gray.shape[1])
	
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)

	color = cv2.bilateralFilter(frame, 9, 150, 0.25)
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	cv2.imshow("frame", cartoon)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()