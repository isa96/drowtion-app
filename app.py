# interface
import streamlit as st
import pandas as pd

# computer vision
from scipy.spatial.distance import euclidean
from imutils import face_utils
from time import time
import cv2 as cv
import imutils
import dlib

# system integration
from datetime import datetime as dt
from warnings import filterwarnings
import sqlite3
import os
filterwarnings("ignore")

# static integration (local DB, logs, trained model)

conn = sqlite3.connect('database/data.db')
c = conn.cursor()

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

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


# menu feature
mode = st.sidebar.selectbox("Choose Mode", ["User", "Detection","Admin"])

# user feature
if mode == "User":
	# login feature
	col_1, col_2, col_3 = st.columns([10, 30, 10])
	col_2.title("Drowtion Page")

	username = st.text_input("Username :")
	password = st.text_input("Password :")
	col1, col2, col3 = st.columns([18, 10, 30])
	click_login = col2.button("Sign Up")
	click_create = col3.button("Create Account")

	if click_login:
		create_usertable()
		result = login_user(username, password)
		print(result)
		if result:
			st.success(f"Hi {username}...")
		else:
			st.error("Please Click 'Create Account' Button")

	if click_create:
		create_usertable()
		add_userdata(username, password)
		st.success("Please Click 'Sign Up' Button")

if mode == "Detection":
	# drowtion start here
	col__1, col__2, col__3 = st.columns([10, 30, 10])
	col__2.title("Drowtion Webcam")
	username = st.text_input("Username")
	selection_position = st.selectbox("Select Your Position",
	                                  ['', 'Line Production',
	                                   'Warehouse', 'Car'])
	start = st.checkbox("Start / Stop")
	frames = st.image([])
	camera = cv.VideoCapture(0)
	flag = 0
	fps = 0
	if username != "":
		while start:
			_, frame = camera.read()
			frame = imutils.resize(frame, width=600)
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			subjects = detect(gray, 0)
			for subject in subjects:
				shape = predict(gray, subject)
				shape = face_utils.shape_to_np(shape)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				leftEyeHull = cv.convexHull(leftEye)
				rightEyeHull = cv.convexHull(rightEye)
				cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				if ear < thresh:
					flag += 1
					# if flag == frame_check:
						# actuator here
					cv.putText(frame,
								"Drowsy Detected",
								(160, 30),
								cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
					current_time = str(dt.now()).split(".")[0]
					# save image to current database
					# print(f"/database/detected/{username}_{current_time}.jpg")
					cv.imwrite(f"database/detected/{username.lower()}_{''.join([i for i in str(dt.now()) if i.isdigit()])}.jpg", frame)
						# print(frame.shape)
					# elif flag > 20:
					# 	flag = 0
				else:
					flag = 0

				# FPS Count
				new_frame_time = time()
				fps = 1 / (new_frame_time - prev_frame_time)
				prev_frame_time = new_frame_time
				# converting the fps into integer
				fps = str(int(fps))

			cv.putText(frame, str(flag), (490, 70), cv.FONT_HERSHEY_SIMPLEX,
			            3, (100, 100, 255), 2, cv.LINE_AA)
			# putting the FPS count on the frame
			cv.putText(frame, str(fps), (7, 70), cv.FONT_HERSHEY_SIMPLEX,
			            3, (100, 255, 0), 3, cv.LINE_AA)
			frames.image(frame[:, :, ::-1])
	else:
		st.error("Please Input Username First")

# admin feature (overview user detected)
if mode == "Admin":
	# login feature
	col_1, col_2, col_3 = st.columns([10, 30, 10])
	col_2.title("Drowtion Page")

	username = st.text_input("Username :")
	password = st.text_input("Password :")
	col1, col2, col3 = st.columns([18, 10, 30])
	click_login = col2.button("Sign Up")
	if username == "admin" and password == "admin":
		st.success("Hello Admin")
		click_show = col3.button("Show ID Detected")
		if click_show:
			# show image
			# st.image(["database/detected/" + i for i in os.listdir("database/detected") if i.endswith(".jpg")])
			# show logs
			list_logs = [i for i in os.listdir("database/detected") if i.endswith(".jpg")]
			# name  : list_logs[0].split('_')[0]
			# year  : list_logs[0].split('_')[1][:4]
			# month : list_logs[0].split('_')[1][4:6]
			# day   : list_logs[0].split('_')[1][6:8]
			# hour  : list_logs[0].split('_')[1][8:10]
			# minute: list_logs[0].split('_')[1][10:12]
			data_logs = {'name':[list_logs[i].split('_')[0] for i in range(len(list_logs))],
			             'year':[list_logs[i].split('_')[1][:4] for i in range(len(list_logs))],
			             'month':[list_logs[i].split('_')[1][4:6] for i in range(len(list_logs))],
			             'day':[list_logs[i].split('_')[1][6:8] for i in range(len(list_logs))],
			             'hour':[list_logs[i].split('_')[1][8:10] for i in range(len(list_logs))],
			             'minute':[list_logs[i].split('_')[1][10:12] for i in range(len(list_logs))]}
			data = pd.DataFrame(data_logs)
			st.dataframe(data)


