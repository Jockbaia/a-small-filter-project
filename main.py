import cv2
import dlib
import keyboard
import numpy as np

def change_lips(img, lm):
    upper_lip = [lm.part(48).x, lm.part(48).y,
                 lm.part(49).x, lm.part(49).y,
                 lm.part(50).x, lm.part(50).y,
                 lm.part(51).x, lm.part(51).y,
                 lm.part(52).x, lm.part(52).y,
                 lm.part(53).x, lm.part(53).y,
                 lm.part(54).x, lm.part(54).y,
                 lm.part(64).x, lm.part(64).y,
                 lm.part(63).x, lm.part(63).y,
                 lm.part(62).x, lm.part(62).y,
                 lm.part(61).x, lm.part(61).y,
                 lm.part(60).x, lm.part(60).y]
    lower_lip = [lm.part(45).x, lm.part(45).y,
                 lm.part(59).x, lm.part(59).y,
                 lm.part(58).x, lm.part(58).y,
                 lm.part(57).x, lm.part(57).y,
                 lm.part(56).x, lm.part(56).y,
                 lm.part(55).x, lm.part(55).y,
                 lm.part(54).x, lm.part(54).y,
                 lm.part(64).x, lm.part(64).y,
                 lm.part(65).x, lm.part(65).y,
                 lm.part(66).x, lm.part(66).y,
                 lm.part(67).x, lm.part(67).y,
                 lm.part(60).x, lm.part(60).y]
    mask = np.zeros(img.shape)
    cv2.fillPoly(mask, np.array([upper_lip]), (0, 0, 255))
    cv2.fillPoly(mask, np.array([lower_lip]), (0, 0, 255))
    return cv2.addWeighted(img, 1, mask, 0.5, 0.0, img)


def apply_mask(flag, image, landmarks):   #flag per identificare la parte del viso da modificare (es. 1 - naso, 2 - bocca, ecc...)
    if flag == 2: #esempio stupido per cambiare colore labbra
        res = change_lips(image, landmarks)
    return res


def main():
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
    final_frame = np.copy(frame)

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

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #final_frame = apply_mask(2, frame, landmarks)

        cv2.imshow("main", frame)
        #cv2.imshow("main", final_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(1)