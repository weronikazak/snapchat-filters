import cv2
import numpy as np
# import tensorflow as tf
import dlib
from imutils import face_utils

jaw = [0, 17]
r_eyebrow, l_eyebrow = [18, 22], [23, 27]
nose = [28, 36]
r_eye, l_eye = [37, 42], [43, 48]
mouth = [49, 68]

scale = 10
padding = 10
x_offset = y_offset = 10

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)



def create_bigger_eye(frame, mask, rang, right=False):
    (xx, yy, w, h) = cv2.boundingRect(np.array([rang]))

    roi = frame[yy-padding:yy+padding + h, xx-padding:xx+padding + w]
    mask = mask[yy-padding:yy+padding + h, xx-padding:xx+padding + w]
    bigger_eye = cv2.resize(roi, (int(roi.shape[1]*1.2), int(roi.shape[0]*1.2)))
    mask = cv2.resize(mask, (int(mask.shape[1]*1.2), int(mask.shape[0]*1.2)))

    b_e_pos_y = int(y - 1.5*padding)
    b_e_pos_x = x - padding

    if right == True:
        b_e_pos_x -= padding

    return bigger_eye, mask, b_e_pos_y, b_e_pos_x



def blur_edges(src, dest, mask, center):
    out = cv2.seamlessClone(src, dest, mask, center, cv2.NORMAL_CLONE)
    return out


def prepare_mask(img, rang):
    src_mask = np.zeros(img.shape, img.dtype)
    poly = np.array(rang, np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    return src_mask



while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    org_frame = frame.copy()
    frame_2 = frame.copy()
    mask = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        i, j = l_eye[0], l_eye[1]

        for (x, y) in shape[i:j]:
            # print(shape[i:j])
            mask = prepare_mask(frame, shape[i:j])
            bigger_eye, mask, a, b = create_bigger_eye(frame, mask, shape[i:j])
            
        # frame[a:a + bigger_eye.shape[0], b:b+bigger_eye.shape[1]] = bigger_eye
        frame_2 = blur_edges(bigger_eye, frame, mask, (b, a))

        i, j = r_eye[0], r_eye[1]

        # for (x, y) in shape[i:j]:
        #     bigger_eye, mask, a, b = create_bigger_eye(frame, mask, shape[i:j], True)
        
        # frame[a:a + bigger_eye.shape[0], b:b+bigger_eye.shape[1]] = bigger_eye

    cv2.imshow('frame', frame_2)
    cv2.imshow("mask.jpg", mask)
    cv2.imshow("frame1.jpg", bigger_eye)
    # cv2.imwrite("frame2.jpg", frame_2)
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()