# eye-blink-detection

An algorithm to count, visualize and safe the number of eye blinks in a video file or video stream.

## Getting started

### Prerequisites

```
pip3 install --user dlib
```

### How to run the code

```
python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video path_to_video_file
```

## What will be done

* (if it doesn't exist) file "eye_blink_results.txt" will be (created) enhanced when finished
* for each video a .png file will be created that shows the graph of the ear including all important results

### Hints for the videos

* whole face has to be visible and evenly-lit
* laughing and talking can change the result
* glasses can make problems

The goal was to have good results on 5 min long videos. To cut your video you can use this:

template
```
ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
```
example:
```
ffmpeg -ss 1:30 -i video_original.mkv -t 5:00 -c copy video_cut.mkv
```