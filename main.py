import cv2
import dlib
import keyboard
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


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
    cv2.fillPoly(mask, np.array([upper_lip]), (0, 0, 120))
    cv2.fillPoly(mask, np.array([lower_lip]), (0, 0, 120))
    return cv2.addWeighted(img, 1, mask, 0.2, 0.0, img)


def perspective_image(img, foreground_path, myPoints):

    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    background = img
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)

    input_pts = np.float32([[0, 0], [foreground.shape[1], 0], [0, foreground.shape[0]], [foreground.shape[1], foreground.shape[0]]])
    output_pts = np.float32(myPoints)

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    foreground = cv2.warpPerspective(foreground, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = foreground[:, :, 3] / 255.0

    for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_foreground)

    background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return background


def add_eyebrow_piercing(img, lm):
    piercing = cv2.imread("piercing.png")
    mask = np.zeros(img.shape, dtype=np.uint8)
    eyebrow_up = (lm.part(25).x, lm.part(25).y)
    ref_pt = (int((lm.part(45).x + lm.part(44).x) / 2), int((lm.part(44).y + lm.part(45).y) / 2))
    height = int((ref_pt[1] - eyebrow_up[1]) / 2)
    width = height
    half_width = int(width/2)
    center = (int((eyebrow_up[0] + ref_pt[0]) / 2), int((eyebrow_up[1] + ref_pt[1]) / 2))
    piercing = cv2.resize(piercing, (width, height))
    roi = mask[center[1] - height:center[1], center[0] + half_width:center[0] + width, :]
    roi = cv2.add(roi, piercing[:, half_width:, :])
    mask[center[1] - height:center[1], center[0] + half_width:center[0] + width, :] = roi
    res = cv2.add(img, mask)
    return res


def add_septum(img, lm):
    piercing = cv2.imread("piercing.png")
    mask = np.zeros(img.shape, dtype=np.uint8)
    midnose_left = (lm.part(32).x, lm.part(32).y)
    midnose_right = (lm.part(34).x, lm.part(34).y)
    nosetip = (lm.part(30).x, lm.part(30).y)
    width = midnose_right[0] - midnose_left[0]
    half_width = int(width / 2)
    center_upper_lip = (lm.part(51).x, lm.part(51).y)
    height = int((center_upper_lip[1] - nosetip[1]) / 2)
    one_third_height = int(height / 3)
    center = (midnose_left[0] + half_width, nosetip[1] + (height - one_third_height))
    piercing = cv2.resize(piercing, (width, height))
    roi = mask[center[1] - one_third_height:center[1] + (height - one_third_height),
          center[0] - half_width:center[0] + (width - half_width), :]
    roi = cv2.add(roi, piercing)
    mask[center[1]:center[1] + (height - one_third_height), center[0] - half_width:center[0] + (width - half_width),
    :] = roi[one_third_height:, :, :]
    res = cv2.add(img, mask)
    return res


def cheek_filter(img, lm, filter_path):
    left_cheekbone = (lm.part(0).x, lm.part(0).y)
    right_cheekbone = (lm.part(16).x, lm.part(16).y)
    down_left = (int((lm.part(3).x + lm.part(0).x) / 2), lm.part(3).y)
    down_right = (int((lm.part(13).x + lm.part(16).x) / 2), lm.part(13).y)
    output_pts = np.float32([left_cheekbone, right_cheekbone, down_left, down_right])
    res = perspective_image(img, filter_path, output_pts)
    #transform = cv2.getPerspectiveTransform(input_pts, output_pts)
    #out = cv2.warpPerspective(freckles, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    #res = cv2.addWeighted(img, 1, out, 0.7, 0)
    return res


def add_beard(img, lm):
    left_cheekbone = (lm.part(1).x, lm.part(1).y)
    right_cheekbone = (lm.part(15).x, lm.part(15).y)
    down_left = (lm.part(2).x, int(lm.part(8).y * 1.05))
    down_right = (lm.part(14).x, down_left[1])
    output_pts = np.float32([left_cheekbone, right_cheekbone, down_left, down_right])
    res = perspective_image(img, "beard.png", output_pts)
    return res


def put_glasses(img, lm):
    lens = [(lm.part(17).x, lm.part(19).y), (lm.part(26).x, lm.part(24).y), (lm.part(17).x, lm.part(1).y),
            (lm.part(26).x, lm.part(15).y)]
    left_stick = [(lm.part(15).x, lm.part(15).y),
                  (lm.part(16).x, lm.part(16).y),
                  (lm.part(26).x, lm.part(15).y),
                  (lm.part(26).x, lm.part(24).y)]
    right_stick = [(lm.part(1).x, lm.part(1).y),
                   (lm.part(0).x, lm.part(0).y),
                   (lm.part(17).x, lm.part(1).y),
                   (lm.part(17).x, lm.part(19).y)]

    res = img
    res = perspective_image(res, "glasses.png", lens)
    res = perspective_image(res, "stick.png", left_stick)
    res = perspective_image(res, "stick.png", right_stick)
    return res


def apply_mask(flag, image,
               lm):  # flag per identificare il filtro da applicare
    if flag == 1:
        res = change_lips(image, lm)
    elif flag == 2:  # add glasses
        res = put_glasses(image, lm)
        #lens = [(lm.part(17).x, lm.part(19).y), (lm.part(26).x, lm.part(24).y), (lm.part(17).x, lm.part(1).y),
        #        (lm.part(26).x, lm.part(15).y)]
        #left_stick = [(lm.part(15).x, lm.part(15).y),
        #              (lm.part(16).x, lm.part(16).y),
        #              (lm.part(26).x, lm.part(15).y),
        #              (lm.part(26).x, lm.part(24).y)]
        #right_stick = [(lm.part(1).x, lm.part(1).y),
        #               (lm.part(0).x, lm.part(0).y),
        #               (lm.part(17).x, lm.part(1).y),
        #               (lm.part(17).x, lm.part(19).y)]

        #res = image
        #res = perspective_image(res, "glasses.png", lens)
        #res = perspective_image(res, "stick.png", left_stick)
        #res = perspective_image(res, "stick.png", right_stick)

    elif flag == 3:
        res = add_eyebrow_piercing(image, lm)
    elif flag == 4:
        res = add_septum(image, lm)
    elif flag == 5:
        res = cheek_filter(image, lm, "lentiggini3_cut.png")
    elif flag == 6:
        res = cheek_filter(image, lm, "blush.png")
    elif flag == 7:
        res = cheek_filter(image, lm, "anime blush cut.png")
    elif flag == 8:
        res = add_beard(image, lm)
    elif flag == 9:
        res = cheek_filter(image, lm, "blush.png")
        res = cheek_filter(res, lm, "lentiggini3_cut.png")
    elif flag == 10:
        res = change_lips(image, lm)
        res = cheek_filter(res, lm, "lentiggini3_cut.png")
    elif flag == 11:
        res = change_lips(image, lm)
        res = cheek_filter(res, lm, "anime blush cut.png")
    elif flag == 12:
        res = add_eyebrow_piercing(image, lm)
        res = add_septum(res, lm)
    else:
        res = image
    return res


def main():
    landmark_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    viewmode = 1
    polygons = 0
    image_filter = 1

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
        if keyboard.is_pressed('+'):
            image_filter = (image_filter + 1) % 13

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

            final_frame = apply_mask(image_filter, frame, landmarks)

        # cv2.imshow("main", frame)
        cv2.imshow("main", final_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(1)
