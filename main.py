import cv2
import dlib
import keyboard
import numpy as np


def change_lips(img, lm):
    upper_lip = [(lm.part(48).x, lm.part(48).y),
                 (lm.part(49).x, lm.part(49).y),
                 (lm.part(50).x, lm.part(50).y),
                 (lm.part(51).x, lm.part(51).y),
                 (lm.part(52).x, lm.part(52).y),
                 (lm.part(53).x, lm.part(53).y),
                 (lm.part(54).x, lm.part(54).y),
                 (lm.part(64).x, lm.part(64).y),
                 (lm.part(63).x, lm.part(63).y),
                 (lm.part(62).x, lm.part(62).y),
                 (lm.part(61).x, lm.part(61).y),
                 (lm.part(60).x, lm.part(60).y)]
    lower_lip = [(lm.part(48).x, lm.part(48).y),
                 (lm.part(59).x, lm.part(59).y),
                 (lm.part(58).x, lm.part(58).y),
                 (lm.part(57).x, lm.part(57).y),
                 (lm.part(56).x, lm.part(56).y),
                 (lm.part(55).x, lm.part(55).y),
                 (lm.part(54).x, lm.part(54).y),
                 (lm.part(64).x, lm.part(64).y),
                 (lm.part(65).x, lm.part(65).y),
                 (lm.part(66).x, lm.part(66).y),
                 (lm.part(67).x, lm.part(67).y),
                 (lm.part(60).x, lm.part(60).y)]
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([upper_lip]), (0, 0, 80))
    cv2.fillPoly(mask, np.array([lower_lip]), (0, 0, 80))
    return cv2.addWeighted(img, 1, mask, 0.2, 0.0, img)


def put_glasses(img, lm):
    left_stick = [(lm.part(16).x, lm.part(16).y),
                  (int((lm.part(16).x + lm.part(15).x) / 2), int((lm.part(16).y + lm.part(15).y) / 2)),
                  (lm.part(26).x, lm.part(26).y),
                  (lm.part(26).x, lm.part(24).y)]
    right_stick = [(lm.part(0).x, lm.part(0).y),
                   (int((lm.part(0).x + lm.part(1).x) / 2), int((lm.part(0).y + lm.part(1).y) / 2)),
                   (lm.part(17).x, lm.part(17).y),
                   (lm.part(17).x, lm.part(19).y)]
    glass = [(lm.part(26).x, lm.part(24).y),
             (lm.part(17).x, lm.part(19).y),
             (lm.part(17).x, lm.part(1).y), # da rivedere
             (lm.part(26).x, lm.part(15).y)] # da rivedere
    nose = [(lm.part(27).x, lm.part(27).y),
            (lm.part(31).x, lm.part(31).y),
            (lm.part(35).x, lm.part(35).y)]
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([left_stick]), (51, 255, 153))
    cv2.fillPoly(mask, np.array([right_stick]), (51, 255, 153))
    cv2.fillPoly(mask, np.array([glass]), (51, 255, 153))
    cv2.fillPoly(mask, np.array([nose]), (0, 0, 0))
    return cv2.addWeighted(img, 1, mask, 0.9, 0.0, img)


def apply_mask(flag, image,
               landmarks):  # flag per identificare la parte del viso da modificare (es. 1 - naso, 2 - bocca, ecc...)
    if flag == 2:
        res = put_glasses(image, landmarks)
    if flag == 1:
        res = change_lips(image, landmarks)
    else:
        res = image
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
            viewmode = (viewmode + 1) % 3
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
                if viewmode == 1:
                    frame = cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=2, color=(0, 0, 255),
                                       thickness=1)
                if viewmode == 2:
                    frame = cv2.putText(frame, str(i), (landmarks.part(i).x, landmarks.part(i).y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.2, (211, 211, 211), 1, cv2.LINE_AA, False)
                else:
                    continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            final_frame = apply_mask(2, frame, landmarks)

        # cv2.imshow("main", frame)
        cv2.imshow("main", final_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(1)
