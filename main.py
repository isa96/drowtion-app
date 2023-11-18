# load library
from scipy.spatial.distance import euclidean
from datetime import datetime as dt
from imutils import face_utils
from time import time
import imutils
import dlib
import cv2

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

cap=cv2.VideoCapture(0)
flag=0
fps = 0
count = 0
FPS = []
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
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
		if ear < thresh:
			flag += 1
			# print (flag)
			print(ear)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cur = str(dt.now()).split(".")[0].split()
				current_time = cur[0]+"_"+cur[1]
				print(current_time)
				cv2.imwrite(
					f"database/detected/alif_{''.join([i for i in str(dt.now()) if i.isdigit()])}.jpg",
					frame)
		else:
			flag = 0

		# FPS Count
		new_frame_time = time()
		fps = 1 / (new_frame_time - prev_frame_time)
		count += 1
		FPS.append(int(fps))
		prev_frame_time = new_frame_time
		# converting the fps into integer
		fps = str(int(fps))

	cv2.putText(frame, str(flag), (200, 70), cv2.FONT_HERSHEY_SIMPLEX,
	            3, (100, 100, 255), 2, cv2.LINE_AA)
	# putting the FPS count on the frame
	cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
	            3, (100, 255, 0), 3, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		print(f"Mean FPS : {int(sum(FPS) / count)}")
		cv2.destroyAllWindows()
		cap.release()
		break