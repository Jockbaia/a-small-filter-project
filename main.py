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


def put_glasses(img, lm):
    filter_mask = cv2.imread("gla.png", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    pt_A = [lm.part(17).x, lm.part(19).y]
    pt_B = [lm.part(26).x, lm.part(24).y]
    pt_C = [lm.part(17).x, lm.part(1).y]
    pt_D = [lm.part(26).x, lm.part(15).y]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([[0, 0], [filter_mask.shape[1], 0], [0, filter_mask.shape[0]], [filter_mask.shape[1], filter_mask.shape[0]]])
    output_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(filter_mask, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    res = cv2.addWeighted(img, 1, out, 1, 0)
    return res


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


def add_freckles(img, lm):
    freckles = cv2.imread("lentiggini3_cut.png")
    #freckles = cv2.cvtColor(freckles, cv2.COLOR_BGR2RGB)
    left_cheekbone = (lm.part(0).x, lm.part(0).y)
    right_cheekbone = (lm.part(16).x, lm.part(16).y)
    #upper_pt = (int((left_cheekbone[0] + right_cheekbone[0])/2), int((left_cheekbone[1] + right_cheekbone[1])/2))
    down_left = (lm.part(3).x, lm.part(3).y)
    down_right = (lm.part(13).x, lm.part(13).y)
    #lower_pt = (int((down_left[0] + down_right[0])/2), int((down_left[1] + down_right[1])/2))
    input_pts = np.float32([[0, 0], [freckles.shape[1], 0], [0, freckles.shape[0]], [freckles.shape[1], freckles.shape[0]]])
    output_pts = np.float32([left_cheekbone, right_cheekbone, down_left, down_right])
    transform = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(freckles, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    res = cv2.addWeighted(img, 1, out, 0.7, 0)
    #mask = np.zeros(img.shape, dtype=np.uint8)
    #width = right_cheekbone[0] - left_cheekbone[0]
    #height = lower_pt[1] - upper_pt[1]
    #freckles = cv2.resize(freckles, (width, height))
    #roi = mask[upper_pt[1]:lower_pt[1], left_cheekbone[0]:right_cheekbone[0], :]
    #roi = cv2.add(roi, freckles)
    #mask[upper_pt[1]:lower_pt[1], left_cheekbone[0]:right_cheekbone[0], :] = roi
    #res = cv2.add(img, mask)
    return res


def apply_mask(flag, image,
               landmarks):  # flag per identificare il filtro da applicare
    if flag == 1:
        res = change_lips(image, landmarks)
    elif flag == 2:
        res = put_glasses(image, landmarks)
    elif flag == 3:
        res = add_eyebrow_piercing(image, landmarks)
    elif flag == 4:
        res = add_septum(image, landmarks)
    elif flag == 5:
        res = add_freckles(image, landmarks)
    else:
        res = image
    return res


def main():
    landmark_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    viewmode = 1
    polygons = 0
    image_filter = 5

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
            image_filter = (image_filter + 1) % 7

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
            final_frame = apply_mask(image_filter, frame, landmarks)

        # cv2.imshow("main", frame)
        cv2.imshow("main", final_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(1)
