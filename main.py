import cv2
import dlib
import keyboard
import numpy as np

landmark_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
viewmode = 1
polygons = 0

vid_capture = cv2.VideoCapture(0)
if not vid_capture.isOpened():
    print("[ERROR] Cannot open camera")
    exit()
vid, frame = vid_capture.read()
if not vid:
    print("[ERROR] Can't receive frame. Exiting.")
    exit()
rows, cols, _vid = frame.shape

while True:
    vid, frame = vid_capture.read()
    if not vid:
        print("[ERROR] Can't receive frame. Exiting.")
        continue
    faces = landmark_detector(frame)

    if keyboard.is_pressed('q'):
        viewmode = (viewmode + 1) % 2
    if keyboard.is_pressed('w'):
        polygons = (polygons + 1) % 2

    for face in faces:
        landmarks = landmark_predictor(frame, face)

        if polygons:

            left = [(landmarks.part(16).x, landmarks.part(16).y),
                    (landmarks.part(24).x, landmarks.part(24).y),
                    (landmarks.part(22).x, landmarks.part(22).y),
                    (landmarks.part(27).x, landmarks.part(27).y),
                    (landmarks.part(8).x, landmarks.part(8).y),
                    (landmarks.part(12).x, landmarks.part(12).y)]
            right = [(landmarks.part(0).x, landmarks.part(0).y),
                     (landmarks.part(19).x, landmarks.part(19).y),
                     (landmarks.part(21).x, landmarks.part(21).y),
                     (landmarks.part(27).x, landmarks.part(27).y),
                     (landmarks.part(8).x, landmarks.part(8).y),
                     (landmarks.part(4).x, landmarks.part(4).y)]

            cv2.fillPoly(frame, np.array([left]), (0, 0, 255))
            cv2.fillPoly(frame, np.array([right]), (51, 255, 153))

        for i in range(68):
            if viewmode:
                frame = cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=2, color=(0, 0, 255),
                                   thickness=1)
            else:
                frame = cv2.putText(frame, str(i), (landmarks.part(i).x, landmarks.part(i).y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.2, (211, 211, 211), 1, cv2.LINE_AA, False)

    cv2.imshow("main", frame)
    cv2.waitKey(1)
