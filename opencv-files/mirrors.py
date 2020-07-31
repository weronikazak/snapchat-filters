import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
	ret, frame = camera.read()

	split = frame.shape[1] // 2
	one_half = frame[:, :split, :]
	sec_half = cv2.flip(one_half, 1)

	frame = np.hstack((one_half, sec_half))
	cv2.imshow("frame", frame)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()