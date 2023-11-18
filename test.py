# load library
from scipy.spatial.distance import euclidean
from datetime import datetime as dt
from imutils import face_utils
from time import time
import imutils
import dlib
import cv2
import os
import cv2 as cv
# helper
def eye_aspect_ratio(eye):
	A = euclidean(eye[1], eye[5])
	B = euclidean(eye[2], eye[4])
	C = euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# variable initisialisation
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("static/shape_predictor_68_face_landmarks.dat")

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

path = "static/test/"
for i in os.listdir(path): 
    path_image = path + i
    frame = cv.imread(path_image)
    # frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        print(f"Name : {i.split('.')[0]}\nTotal Eye Spect Ratio : {ear}\nThresholding : {ear * (65/100)}")