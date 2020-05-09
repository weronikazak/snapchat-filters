import cv2
import numpy as np
from vcam import vcam, meshGen

camera = cv2.VideoCapture(0)

# x = int(input("Gimme mode nr: "))
x = 3


def modes(mode):
	ret = 0
	if mode == 0:
		ret += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
	elif mode == 1:
		ret += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
	elif mode == 2:
		ret -= 10*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
	elif mode == 3:
		ret -= 10*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
	elif mode == 4:
		ret += 100*np.sqrt((plane.X*1.0/plane.W)**2 + (plane.Y*1.0/plane.H)**2)
	else: exit(-1)

	return ret


# GET FIRST IMAGE

_, image = camera.read()
(h, w) = image.shape[:2]
fps = 30

cam = vcam(H=h, W=w)
plane = meshGen(h, w)

plane.Z += modes(x)

# --------


pts3d = plane.getPlane()

pts2d = cam.project(pts3d)

map_x, map_y = cam.getMaps(pts2d)


while True:
	_, frame = camera.read()

	frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=4)
	frame = cv2.flip(output, 1)

	cv2.imshow("frame", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break