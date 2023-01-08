import cv2
import numpy as np
import dlib
from math import hypot

#učitavanje kamere i maske
cap = cv2.VideoCapture(0)
mask_image = cv2.imread("maska.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
mask_mask = np.zeros((rows, cols), np.uint8)

#detekcija lica
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    mask_mask.fill(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray, face)

        #potrebne koordinate
        left_side = (landmarks.part(0).x, landmarks.part(0).y)
        right_side = (landmarks.part(16).x, landmarks.part(16).y)
        bottom_side = (landmarks.part(8).x, landmarks.part(8).y)
        top_side = (landmarks.part(29).x, landmarks.part(29).y)
        center = (landmarks.part(51).x, landmarks.part(51).y)

        mask_width = int(hypot(left_side[0] - right_side[0], left_side[1] - right_side[1]))
        mask_height = int(mask_width * 0.84)
        
        #pozicija maske
        top_left = (int(center[0] - mask_width / 2), int(center[1] - mask_height / 2))
        bottom_right = (int(center[0] + mask_width / 2), int(center[1] + mask_height / 2))

        #dodavanje maske
        face_mask = cv2.resize(mask_image, (mask_width, mask_height))
        face_mask_gray = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
        _, mask_mask = cv2.threshold(face_mask_gray, 25, 255, cv2.THRESH_BINARY_INV)

        mask_area = frame[top_left[1]: top_left[1] + mask_height, 
                    top_left[0]: top_left[0] + mask_width, 0]
        mask_area_without_mask = cv2.bitwise_and(mask_area, mask_area, mask=mask_mask)
        final_mask = cv2.add(mask_area_without_mask, face_mask[:, :, 0])

        frame[top_left[1]: top_left[1] + mask_height, 
                    top_left[0]: top_left[0] + mask_width, 0] = final_mask
 
        cv2.imshow("Face mask", face_mask)
        cv2.imshow("Mask area", mask_area)
        cv2.imshow("Mask mask", mask_mask)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

