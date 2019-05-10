# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import datetime
import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def draw_eye_contours(eye):
	eyeHull = cv2.convexHull(eye)
	cv2.drawContours(frame, [eyeHull], -1, (0, 255, 0), 1)

def result_to_text_file():
	# open or create file eye_blink_results.txt
	f = open("eye_blink_results.txt", "a+")

	# add time stamp to result
	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H:%M:%S')

	# write result into eye_blink_results.txt
	f.write(timestamp + " | " + format(args["video"]) + " | number of blinks: " + format(total_blinks) 
		+ " | ear_average: " + format(get_ear_average()) + "\r\n\r\n")

def draw_values_on_video():
	# view blink counter on the top left of video
	cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# view current EAR on the top right of video
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (310, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "Treshold: {}".format(eye_ar_treshold), (310, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, "Frames: {}".format(total_frame_counter), (10, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def get_ear_average():
	return ear_total/face_detection_frame_counter

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
# COMES WITH 0.3
eye_ar_treshold = 0.23
# COMES WITH 3 -> 2 might be better with lower frame rate 
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters for consecutive frames under 
# blink treshold and the total number of blinks
counter = 0
total_blinks = 0

# sum of ear in all frames
ear_total = 0
# distance between ear_average and blink treshold
ear_treshold_difference = 0.06
# number of frames when face was detected
face_detection_frame_counter = 0
# total frames of video
total_frame_counter = 0
# all the arrays are for plotting
frame_count_array = []
ear_array = []
threshold_array = []
blink_count_array = []
blink_count_ear_array = []

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale channels)
	# doesn't work when video finishes
	try:
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	except AttributeError:

		plt.figure(figsize=(15,8))
		plt.scatter(blink_count_array, blink_count_ear_array, zorder = 1, color = "red", marker = "x")
		plt.plot(frame_count_array, ear_array, color = "blue")
		plt.plot(frame_count_array, threshold_array, color = "green")
		plt.savefig(format(args["video"] + ".png"),figsize = (100, 40), bbox_inches = 'tight')

		result_to_text_file()

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	print(format(rects))
	total_frame_counter += 1

	# loop over the face detections
	for rect in rects:
		#face was detected so increase counter by 1
		face_detection_frame_counter += 1
		frame_count_array.append(face_detection_frame_counter)

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		ear_total += ear
		ear_array.append(ear)
		threshold_array.append(eye_ar_treshold)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		draw_eye_contours(leftEye)
		draw_eye_contours(rightEye)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < eye_ar_treshold:
			counter += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# frames then increment the total number of blinks
			if counter >= EYE_AR_CONSEC_FRAMES:
				total_blinks += 1
				blink_count_array.append(face_detection_frame_counter)
				blink_count_ear_array.append(ear)
			# reset the eye frame counter
			counter = 0

		draw_values_on_video()

		#update ear_ar_treshold every 10 frames
		if face_detection_frame_counter % 20 == 0:
			eye_ar_treshold = get_ear_average() - ear_treshold_difference

			#plt.plot(frame_count_array, ear_array, color = "blue")
			#plt.scatter(blink_count_array, blink_count_ear_array, color = "red", marker = "x")
			#plt.pause(0.00001)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
