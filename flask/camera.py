import cv2
import numpy as np
import dlib
from imutils import face_utils, translate

class Camera(object):
	def __init__(self):
		self.camera = cv2.VideoCapture(0)

		p = "../data/shape_predictor_68_face_landmarks.dat"
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(p)
		self.effect = "contours"


	def __del__(self):
		self.camera.release()


	def return_jpg(self, frame):
		ret, jpeg = cv2.imencode('.jpeg', frame)
		return jpeg.tobytes()



	def return_effect(self):
		if self.effect == "contours":
			frame = self.effect_canny()

		elif self.effect == "baby":
			frame = self.effect_baby_face()

		elif self.effect == "blurr":
			frame = self.effect_bluring_face()

		elif self.effect == "cartoon":
			frame = self.effect_cartoon()

		elif self.effect == "doggy":	
			frame = self.effect_dog_face()

		elif self.effect == "large":	
			frame = self.effect_enlarged()

		elif self.effect == "mirrors":	
			frame = self.effect_mirror()

		elif self.effect == "triangle":	
			frame = self.effect_delaunay_triangle()

		elif self.effect == "glasses":	
			frame = self.effect_glasses()

		return frame



	# ---------------
	#    BABY FACE
	# ---------------
	def effect_baby_face(self):
		ret, frame = self.camera.read()
		if not ret:
			return False
		offset = 4
		scale = 1.3

		frame_2 = frame.copy()
		mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		mask = np.zeros(frame.shape, frame.dtype)

		eye_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		eye_mask = np.zeros(frame.shape, frame.dtype)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.detector(gray, 0)

		for rect in rects:
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			l_eye, r_eye = shape[36:42], shape[42:48]

			(lx, ly, lw, lh) = cv2.boundingRect(l_eye)
			(rx, ry, rw, rh) = cv2.boundingRect(r_eye)

			l_eye = frame[ly-offset:ly+lh+offset, lx-offset:lx+lw+offset]
			r_eye = frame[ry-offset:ry+rh+offset, rx-offset:rx+rw+offset]

			center_ly = lx + int(lw / 2)
			center_lx = ly + int(lh / 2) + 20
			center_ry = rx + int(rw / 2)
			center_rx = ry + int(rh / 2) + 20

			mouth = shape[48:69]

			(mx, my, mw, mh) = cv2.boundingRect(mouth)
			mouth = frame[my-offset:my+mh+offset, mx-offset:mx+mw+offset]

			center_my = mx + int(mw / 2)
			center_mx = my + int(mh / 2)

			ly_scaled = int((l_eye.shape[1]*scale)/2)
			lx_scaled = int((l_eye.shape[0]*scale)/2)
			ry_scaled = int((r_eye.shape[1]*scale)/2)
			rx_scaled = int((r_eye.shape[0]*scale)/2)

			l_eye = cv2.resize(l_eye, (ly_scaled*2, lx_scaled*2), interpolation = cv2.INTER_AREA)
			r_eye = cv2.resize(r_eye, (ry_scaled*2, rx_scaled*2), interpolation = cv2.INTER_AREA)

			frame[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = l_eye
			mask[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = 255
			frame[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = r_eye
			mask[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = 255

			final_center_x = int(np.mean([center_lx, center_rx]))
			final_center_y = int(np.mean([center_ly, center_ry]))

			frame = cv2.seamlessClone(frame, frame_2, mask, (final_center_y, final_center_x), cv2.NORMAL_CLONE)

		return self.return_jpg(frame)


	# ------------------
	#    ENLARGED EYES
	# ------------------
	def effect_enlarged(self):
		offset = 4
		scale = 2
		ret, frame = self.camera.read()
		if not ret:
			return False
		frame_2 = frame.copy()

		mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		mask = np.zeros(frame.shape, frame.dtype)

		l_eye, r_eye = 0, 0

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.detector(gray, 0)

		for rect in rects:
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			l_eye, r_eye = shape[36:42], shape[42:48]

			(lx, ly, lw, lh) = cv2.boundingRect(l_eye)
			(rx, ry, rw, rh) = cv2.boundingRect(r_eye)

			l_eye = frame[ly-offset:ly+lh+offset, lx-offset:lx+lw+offset]
			r_eye = frame[ry-offset:ry+rh+offset, rx-offset:rx+rw+offset]

			center_ly = lx + int(lw / 2)
			center_lx = ly + int(lh / 2) + 20
			center_ry = rx + int(rw / 2)
			center_rx = ry + int(rh / 2) + 20

			mouth = shape[48:69]

			(mx, my, mw, mh) = cv2.boundingRect(mouth)
			mouth = frame[my-offset:my+mh+offset, mx-offset:mx+mw+offset]

			center_my = mx + int(mw / 2)
			center_mx = my + int(mh / 2)

			ly_scaled = int((l_eye.shape[1]*1.7)/2)
			lx_scaled = int((l_eye.shape[0]*1.7)/2)
			ry_scaled = int((r_eye.shape[1]*1.7)/2)
			rx_scaled = int((r_eye.shape[0]*1.7)/2)

			l_eye = cv2.resize(l_eye, (ly_scaled*2, lx_scaled*2), interpolation = cv2.INTER_AREA)
			r_eye = cv2.resize(r_eye, (ry_scaled*2, rx_scaled*2), interpolation = cv2.INTER_AREA)

			my_scaled = int((mouth.shape[1]*scale)/2)
			mx_scaled = int((mouth.shape[0]*scale)/2)

			mouth = cv2.resize(mouth, (my_scaled*2, mx_scaled*2), interpolation = cv2.INTER_AREA)

			frame[center_mx-mx_scaled:center_mx+mx_scaled, center_my-my_scaled:center_my+my_scaled] = mouth
			mask[center_mx-mx_scaled:center_mx+mx_scaled, center_my-my_scaled:center_my+my_scaled] = 255

			frame[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = l_eye
			mask[center_lx-lx_scaled:center_lx+lx_scaled, center_ly-ly_scaled:center_ly+ly_scaled] = 255
			frame[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = r_eye
			mask[center_rx-rx_scaled:center_rx+rx_scaled, center_ry-ry_scaled:center_ry+ry_scaled] = 255

			final_center_x = int(np.mean([center_lx, center_mx, center_rx]))
			final_center_y = int(np.mean([center_ly, center_my, center_ry]))

			frame = cv2.seamlessClone(frame, frame_2, mask, (final_center_y, final_center_x), cv2.NORMAL_CLONE)

		return self.return_jpg(frame)

	
	# ------------------
	#    BLURRING FACE
	# ------------------
	def effect_bluring_face(self):
		ret, frame = self.camera.read()
		if not ret:
			return False
		face = 0

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.detector(gray, 0)

		for rect in rects:
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			(x, y, w, h) = face_utils.rect_to_bb(rect)

			face = frame[y:y+h, x:x+w]
			face = blurr_face(face)
			face = pixel_face(face)

			frame[y:y+h, x:x+w] = face
		
		return self.return_jpg(frame)


	# ------------------------
	#    DELAUNAY TRIANGLE
	# ------------------------
	def effect_delaunay_triangle(self):
		ret, frame = self.camera.read()
		if not ret:
			return False

		jaw = [0, 17]
		r_eyebrow, l_eyebrow = [18, 22], [23, 27]
		nose = [28, 36]
		r_eye, l_eye = [37, 42], [43, 48]
		mouth = [49, 68]

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		mask = np.zeros_like(gray)

		faces = self.detector(gray, 0)
		for face in faces:
		    landmark = self.predictor(gray, face)
		    landmark_points = []
		    for n in range(68):
		        x = landmark.part(n).x
		        y = landmark.part(n).y
		        landmark_points.append((x, y))

		    points = np.array(landmark_points, np.int32)
		    convexhull = cv2.convexHull(points)

		    cv2.fillConvexPoly(mask, convexhull, 255)

		    face = cv2.bitwise_and(frame, frame, mask=mask)

		    gray = delaunay_traingle(convexhull, landmark_points, gray, landmark_points)

		return self.return_jpg(gray)


	# --------------
	#    DOG FACE
	# --------------
	def effect_dog_face(self):
		ret, frame = self.camera.read()
		if not ret:
			return False
		dog_nose = cv2.imread("../images/nose.png", -1)
		dog_ears = cv2.imread("../images/ears.png", -1)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.detector(gray, 0)

		for rect in rects:
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

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

		return self.return_jpg(frame)


	# -----------------
	#    FUNNY GLASSES
	# -----------------
	def effect_glasses(self):
		ret, frame = self.camera.read()
		if not ret:
			return False
		glasses = cv2.imread("../images/glasses.png", -1)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.detector(gray, 0)

		for rect in rects:
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

		glasses_width = int(abs(shape[36][0] - shape[32][0]) * 4)
		glasses_height = int(glasses_width * 0.7)

		(glasses_x, glasses_y) = shape[30]
		glasses_y -= 20

		half_width = int(glasses_width/2.0)
		half_height = int(glasses_height/2.0)

		y1, y2 = glasses_y - half_height, glasses_y + half_height
		x1, x2 = glasses_x - half_width, glasses_x + half_width

		glasses = cv2.resize(glasses, (half_width*2, half_height*2), interpolation = cv2.INTER_AREA)

		alpha_s = glasses[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s

		for c in range(0, 3):
			frame[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + 
		                            alpha_l * frame[y1:y2, x1:x2, c])


		return self.return_jpg(frame)


	# ----------------------
	#    CARTOON-ISH
	# ----------------------
	def effect_cartoon(self):
		ret, frame = self.camera.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray, 5)
		edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)

		color = cv2.bilateralFilter(frame, 9, 150, 0.25)
		cartoon = cv2.bitwise_and(color, color, mask=edges)

		return self.return_jpg(cartoon)


	# ------------
	#    CANNY
	# ------------
	def effect_canny(self):
		ret, frame = self.camera.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)

		median = np.median(blurred)
		l_edge = int(max(0, 0.77 * median))
		u_edge = int(max(0, 1.33 * median))

		canny = cv2.Canny(blurred, l_edge, u_edge)

		return self.return_jpg(canny)


	# ------------
	#    MIRRORS
	# ------------
	def effect_mirror(self):
		ret, frame = self.camera.read()

		split = frame.shape[1] // 2
		one_half = frame[:, :split, :]
		sec_half = cv2.flip(one_half, 1)

		frame = np.hstack((one_half, sec_half))

		return self.return_jpg(frame)



# ---------------------
# ADDITIONAL FUNCTIONS
# ---------------------

def blurr_face(image):
	(h, w) = image.shape[:2]

	kernel_w = int(w/3.0)
	kernel_h = int(h/3.0)

	if kernel_w % 2 == 0:
		kernel_w -= 1
	else: kernel_w = 5

	if kernel_h % 2 == 0:
		kernel_h -= 1
	else: kernel_h = 5

	img = cv2.GaussianBlur(image, (kernel_w, kernel_h), 0)
	return img


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



def delaunay_traingle(convexHull, points, frame, landmark_points):
    rect = cv2.boundingRect(convexHull)

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)


    for t in triangles:
        A, B, C = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])

        cv2.line(frame, A, B, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, B, C, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(frame, A, C, (255, 255, 255), 1, cv2.LINE_AA, 0)

    return frame