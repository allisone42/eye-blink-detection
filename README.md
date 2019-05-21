# eye-blink-detection

Preparation to use eye blink detection:
 install dlib: pip3 install --user dlib

Run blink detection:
 - go to folder blink-detection (containing files "detect_blinks.py" and "shape_predictor_68_face_landmarks.dat")
 - run this in terminal:
  python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video path_to_video_file
 - if it doesn't exist file "eye_blink_results.txt" will be created/ enhanced when finished

Path to Videos on Eye tracking computer:
 C:\Program Files\SMI\Experiment Suite 360\Experiment Center 2\Results\Camera_testing

Cut video:
 template:
  ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
 example:
  ffmpeg -ss 1:30 -i video_original.mkv -t 5:00 -c copy video_cut.mkv
-> for start time use time from .txt file from the Experiment Center 2 results folders

crop video:
 template:
  ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
 example:
  ffmpeg -i video_original.mkv -filter:v "crop=500:500:400:200" video_cropped.mkv

 Where the options are as follows:
  out_w is the width of the output rectangle
  out_h is the height of the output rectangle
  x and y specify the top left corner of the output rectangle

Hints for the videos:
 - whole face has to be visible
 - laughing and talking can change the result
 