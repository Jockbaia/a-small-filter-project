import cv2
import dlib
import keyboard

landmark_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
viewmode = 1

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

    for face in faces:
        landmarks = landmark_predictor(frame, face)



        for i in range(68):
            if viewmode:
                frame = cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=2, color=(0, 0, 255),
                                   thickness=1)
            else:
                frame = cv2.putText(frame, str(i), (landmarks.part(i).x, landmarks.part(i).y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.2, (211, 211, 211), 1, cv2.LINE_AA, False)

    cv2.imshow("main", frame)
    cv2.waitKey(1)
