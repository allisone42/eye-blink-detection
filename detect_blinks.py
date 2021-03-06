# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
# from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import datetime
import dlib
import cv2
import matplotlib.pyplot as plt


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


def create_plot():
    # set size of resulting graphic
    plt.figure(figsize=(40, 5))

    # add_face_detection_plot()
    add_ear_plot()
    add_treshold_plot()
    add_auto_blink_detection_scatter_plot()
    add_manual_blink_detection_scatter_plot()
    add_labels_and_legend()

    save_plot_as_png()


def add_face_detection_plot():
    plt.plot(total_frame_count_array, face_detection_array, '--',
             label='face detection')


def add_ear_plot():
    plt.plot(total_frame_count_array, ear_array, color="blue", label='EAR',
             alpha=.5, linewidth=1.0)


def add_treshold_plot():
    final_treshold_string = format(round(eye_ar_treshold, 2))
    treshold_label = 'treshold (final = ' + final_treshold_string + ')'
    plt.plot(total_frame_count_array, threshold_array, color='green',
             label=treshold_label, alpha=.5, linewidth=1.0)


def add_auto_blink_detection_scatter_plot():
    blinks_label = 'blinks = ' + format(total_blinks)
    plt.scatter(blink_frame_array, blink_count_ear_array,
                color='red', marker='x', label=blinks_label)


def add_manual_blink_detection_scatter_plot():
    manual_blink_counter = len(manual_blink_detection_array)
    manual_blink_counter_array = np.zeros(manual_blink_counter)
    plt.scatter(manual_blink_detection_array, manual_blink_counter_array,
                color='orange', marker='|', label='blinks manually = ' +
                format(manual_blink_counter))


def add_labels_and_legend():
    face_detection_rate_string = format(round(get_face_detection_rate(), 1))
    plt.xlabel('Frame count (' + format(total_frame_counter) +
               ', detection rate: ' + face_detection_rate_string + '%)')
    plt.ylabel('EAR (average: ' + format(get_ear_average(2)) + ')')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               borderaxespad=0.)


def save_plot_as_png():
    plt.savefig(format(args["video"] + '|' + get_current_time_string() +
                ".png"), bbox_inches='tight')


def get_current_time_string():
    time_format = '%d-%m-%Y_%H:%M:%S'
    timestamp = datetime.datetime.fromtimestamp(time.time())
    timestamp = timestamp.strftime(time_format)
    return timestamp


def result_to_text_file():
    f = open("eye_blink_results.txt", "a+")

    # add time stamp to result
    timestamp = get_current_time_string()

    # write result into eye_blink_results.txt
    f.write(timestamp + " | " + format(args["video"]) +
            " | number of blinks: " + format(total_blinks) +
            " | ear_average: " + format(get_ear_average(2)) + "\r\n\r\n")


def draw_values_on_video():

    # write blink counter on the top left of video
    draw_text("Blinks: {}".format(total_blinks), 10, 30, 0.7)
    draw_text("EAR: {:.2f}".format(ear), 310, 30, 0.7)
    draw_text("Treshold: {}".format(eye_ar_treshold), 310, 50, 0.5)
    draw_text("Frames: {}".format(total_frame_counter), 10, 50, 0.5)
    manual_blink_counter = len(manual_blink_detection_array)
    if manual_blink_counter > 0:
        draw_text("Manual blink counter: {}".format(manual_blink_counter),
                  10, 65, 0.5)


def draw_text(text, xposition, yposition, text_size):
    cv2.putText(frame, text, (xposition, yposition),
                cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), 2)


def get_ear_average(digits):
    return round((ear_total / face_detection_frame_counter), digits)


def get_ear(shape):
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # draw_eye_contours(leftEye)
    # draw_eye_contours(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = np.mean([leftEAR, rightEAR])
    return ear


def get_current_ear_average():
    if face_detection_frame_counter >= 100:
        number_of_array_elements = 100
    else:
        number_of_array_elements = face_detection_frame_counter
    average = ear_array[-number_of_array_elements:]
    av = np.mean(average)
    return av


def get_face_detection_rate():
    face_detection_rate = (float(face_detection_frame_counter) /
                           total_frame_counter) * 100
    return face_detection_rate


def add_blink_on_button_press():
    manual_blink_detection_array.append(total_frame_counter)


def update_ear_treshold():
    global eye_ar_treshold
    # eye_ar_treshold = get_ear_average(4) - ear_treshold_difference
    eye_ar_treshold = get_current_ear_average() - ear_treshold_difference


# TODO: calculate to make it more adjustable to different eyes
# def update_ear_treshold_difference():
    # global ear_treshold_difference
    # ear_treshold_difference =


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# treshold for detecting a blink
eye_ar_treshold = 0.2
# how many frames under treshold before counting a blink
EYE_AR_CONSEC_FRAMES = 2

# consecutive frames under blink treshold and the total number of blinks
consec_frames_counter = 0
total_blinks = 0

ear_total = 0
# distance between ear_average and blink treshold
ear_treshold_difference = 0.06

face_detection_frame_counter = 0
total_frame_counter = 0

# all the arrays are for plotting
total_frame_count_array = []
ear_array = []
threshold_array = []
blink_frame_array = []
blink_count_ear_array = []
face_detection_array = []
manual_blink_detection_array = []

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
video_stream = FileVideoStream(args["video"]).start()
fileStream = True
# use if input is direct webcam
# video_stream = VideoStream(src=0).start()
# video_stream = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from video file stream
    frame = video_stream.read()
    try:
        # resize frame
        frame = imutils.resize(frame, width=450)
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # finish loop over frames when video stream is finished
    except AttributeError:
        if total_frame_counter > 0:
            create_plot()
            result_to_text_file()
        break

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    if len(rects) == 0:
        face_detection_array.append(0)
        ear_array.append(0)
        threshold_array.append(0)
    else:
        face_detection_array.append(0.5)

    total_frame_counter += 1
    total_frame_count_array.append(total_frame_counter)

    # loop over the face detections
    for rect in rects:
        face_detection_frame_counter += 1
        # determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ear = get_ear(shape)

        ear_total += ear
        ear_array.append(ear)
        threshold_array.append(eye_ar_treshold)

        # if ear is below blink threshold, increment consec_frames_counter
        if ear < eye_ar_treshold:
            consec_frames_counter += 1

        else:
            # if the eyes were closed for a sufficient number of
            # frames then increment the total number of blinks
            if consec_frames_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
                blink_frame_array.append(total_frame_counter)
                blink_count_ear_array.append(ear)
            # reset the consec_frames_counter
            consec_frames_counter = 0

        draw_values_on_video()

        # update ear_ar_treshold every 10 frames
        if face_detection_frame_counter % 20 == 0:
            update_ear_treshold()

    # show the frame
    cv2.imshow(format(args["video"]), frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("b"):
        add_blink_on_button_press()

    if key == ord("p"):
        raw_input("Press Enter to continue...")

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        create_plot()
        break

cv2.destroyAllWindows()
video_stream.stop()
