import cv2
import dlib
import keyboard
import numpy as np
import string
import random
from datetime import datetime


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

    input_pts = np.float32(
        [[0, 0], [foreground.shape[1], 0], [0, foreground.shape[0]], [foreground.shape[1], foreground.shape[0]]])
    output_pts = np.float32(myPoints)

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    foreground = cv2.warpPerspective(foreground, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return overlay_png(background, foreground)


def add_eyebrow_piercing(img, lm):
    piercing = cv2.imread("filters/piercing.png", cv2.IMREAD_UNCHANGED)
    mask = np.zeros(img.shape, dtype=np.uint8)
    eyebrow_up = (lm.part(25).x, lm.part(25).y)
    ref_pt = (int((lm.part(45).x + lm.part(44).x) / 2), int((lm.part(44).y + lm.part(45).y) / 2))
    height = int((ref_pt[1] - eyebrow_up[1]) / 2)
    width = height
    half_width = int(width / 2)
    center = (int((eyebrow_up[0] + ref_pt[0]) / 2), int((eyebrow_up[1] + ref_pt[1]) / 2))
    piercing = cv2.resize(piercing, (width, height))
    roi = mask[center[1] - height:center[1], center[0] + half_width:center[0] + width, :]
    roi = cv2.add(roi, piercing[:, half_width:, :])
    mask[center[1] - height:center[1], center[0] + half_width:center[0] + width, :] = roi
    res = cv2.add(img, mask)
    return res


def add_septum(img, lm):
    img = img[:, :, :3]
    piercing = cv2.imread("filters/piercing2.png")
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
    return res


def add_beard(img, lm):
    sx_points = [(lm.part(1).x, lm.part(1).y), (lm.part(33).x, lm.part(33).y),
                 (lm.part(1).x, lm.part(8).y), (lm.part(8).x, lm.part(8).y)]
    dx_points = [(lm.part(33).x, lm.part(33).y), (lm.part(15).x, lm.part(15).y),
                 (lm.part(8).x, lm.part(8).y), (lm.part(15).x, lm.part(8).y)]

    res = perspective_image(img, "filters/beard_sx.png", sx_points)
    res = perspective_image(res, "filters/beard_dx.png", dx_points)
    return res


def glasses_filter(image, lens_image, sl_image, sr_image, lm):

    # sx e dx dal punto di vista dell'osservatore

    res = image

    dx_L = [(lm.part(26).x + lm.part(45).x + lm.part(13).x + lm.part(14).x) * 1/4, (lm.part(26).y + lm.part(45).y + lm.part(13).y + lm.part(14).y) * 1/4]
    sx_L = [(lm.part(17).x + lm.part(36).x + lm.part(2).x + lm.part(3).x) * 1/4, (lm.part(17).y + lm.part(36).y + lm.part(2).y + lm.part(3).y) * 1/4]

    lens_points = [(lm.part(17).x, lm.part(19).y), (lm.part(26).x, lm.part(24).y), sx_L, dx_L]
    left_stick = [
        (lm.part(16).x, lm.part(16).y),
        (lm.part(26).x, lm.part(24).y),
        (lm.part(15).x, lm.part(15).y),
        (lm.part(26).x, lm.part(15).y)]
    right_stick = [
        (lm.part(17).x, lm.part(19).y),
        (lm.part(0).x, lm.part(0).y),
        (lm.part(17).x, lm.part(1).y),
        (lm.part(1).x, lm.part(1).y)
    ]

    # Applying images to webcam feed

    res = perspective_image(res, lens_image, lens_points)
    if lm.part(16).x - lm.part(26).x > 5:
        res = perspective_image(res, sl_image, left_stick)
    if lm.part(17).x - lm.part(0).x > 5:
        res = perspective_image(res, sr_image, right_stick)

    return res


def overlay_png(bg, fg):
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)
    alpha_background = bg[:, :, 3] / 255.0
    alpha_foreground = fg[:, :, 3] / 255.0

    for color in range(0, 3):
        bg[:, :, color] = alpha_foreground * fg[:, :, color] + \
                          alpha_background * bg[:, :, color] * (1 - alpha_foreground)

    bg[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return bg


def nose_overlay(frame, image, lm):

    # sx e dx dal punto di vista dell'osservatore

    sx1 = [lm.part(31).x + ((int(lm.part(29).x) - int(lm.part(31).x)) * (1 / 2)), lm.part(29).y]
    sx2 = [lm.part(31).x + ((int(lm.part(28).x) - int(lm.part(31).x)) * (6 / 7)), lm.part(28).y]

    dx1 = [lm.part(35).x - ((int(lm.part(35).x) - int(lm.part(29).x)) * (1 / 2)), lm.part(29).y]
    dx2 = [lm.part(35).x - ((int(lm.part(35).x) - int(lm.part(28).x)) * (6 / 7)), lm.part(28).y]

    centernose = [(int(lm.part(39).x) + int(lm.part(42).x)) / 2, (int(lm.part(39).y) + int(lm.part(42).y)) / 2]

    nose1 = np.array(
        [centernose, dx2, dx1, [lm.part(35).x, lm.part(35).y], [lm.part(33).x, lm.part(33).y],
         [lm.part(31).x, lm.part(31).y], sx1, sx2], dtype=np.int32)
    nose2 = np.array([sx1, sx2, centernose, [lm.part(35).x, lm.part(35).y]], dtype=np.int32)
    nose3 = np.array([dx1, dx2, centernose, [lm.part(31).x, lm.part(31).y]], dtype=np.int32)

    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, nose1, 1)
    cv2.fillConvexPoly(mask, nose2, 1)
    cv2.fillConvexPoly(mask, nose3, 1)
    mask = mask.astype(bool)

    nose_mask = np.zeros_like(image)
    nose_mask[mask] = image[mask]

    tmp = cv2.cvtColor(nose_mask, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    b, g, r, alpha = cv2.split(nose_mask)

    rgba = [b, g, r, alpha]
    nose_png = cv2.merge(rgba, 4)

    return overlay_png(frame, nose_png)


def mouth_overlay(frame, image, lm):
    mouth = np.array(
        [[lm.part(48).x, lm.part(48).y], [lm.part(49).x, lm.part(49).y], [lm.part(53).x, lm.part(53).y],
         [lm.part(54).x, lm.part(54).y], [lm.part(55).x, lm.part(55).y], [lm.part(56).x, lm.part(56).y],
         [lm.part(57).x, lm.part(57).y], [lm.part(58).x, lm.part(58).y], [lm.part(59).x, lm.part(59).y]],
        dtype=np.int32)

    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, mouth, 1)

    mask = mask.astype(bool)

    mouth_mask = np.zeros_like(image)
    mouth_mask[mask] = image[mask]

    tmp = cv2.cvtColor(mouth_mask, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    b, g, r, alpha = cv2.split(mouth_mask)

    rgba = [b, g, r, alpha]
    mouth_png = cv2.merge(rgba, 4)

    return overlay_png(frame, mouth_png)


def apply_mask(flag, image,
               lm):  # flag per identificare il filtro da applicare

    res = image

    if flag == 1:
        res = change_lips(image, lm)
    elif flag == 2:
        res = glasses_filter(image, "glasses/heart_lens.png", "glasses/heart_L.png", "glasses/heart_R.png", lm)
        res = nose_overlay(res, image, lm)
    elif flag == 3:
        res = glasses_filter(image, "glasses/basic_lens.png", "glasses/basic_L.png", "glasses/basic_R.png", lm)
        res = nose_overlay(res, image, lm)
    elif flag == 4:
        res = add_eyebrow_piercing(image, lm)
    elif flag == 5:
        res = add_septum(image, lm)
    elif flag == 6:
        res = cheek_filter(image, lm, "filters/lentiggini.png")
    elif flag == 7:
        res = cheek_filter(image, lm, "filters/blush.png")
    elif flag == 8:
        res = cheek_filter(image, lm, "filters/anime_blush.png")
        res = nose_overlay(res, image, lm)
    elif flag == 9:
        res = add_beard(image, lm)
        res = mouth_overlay(res, image, lm)
    elif flag == 10:
        res = cheek_filter(image, lm, "filters/blush.png")
        res = cheek_filter(res, lm, "filters/lentiggini.png")
    elif flag == 11:
        res = change_lips(image, lm)
        res = cheek_filter(res, lm, "filters/lentiggini.png")
    elif flag == 12:
        res = change_lips(image, lm)
        res = cheek_filter(res, lm, "filters/anime_blush.png")
        res = nose_overlay(res, image, lm)
    elif flag == 13:
        res = add_eyebrow_piercing(res, lm)
        res = add_septum(res, lm)
    else:
        pass

    return res


def apply_frame(flag, frame):
    res = frame

    if flag == 1:
        pass
    elif flag == 2:
        res = overlay_png(frame, cv2.imread("frames/hearts.png", cv2.IMREAD_UNCHANGED))
    elif flag == 3:
        res = overlay_png(frame, cv2.imread("frames/rainbow.png", cv2.IMREAD_UNCHANGED))
    elif flag == 4:
        res = overlay_png(frame, cv2.imread("frames/tv.png", cv2.IMREAD_UNCHANGED))
    elif flag == 5:
        res = overlay_png(frame, cv2.imread("frames/museum.png", cv2.IMREAD_UNCHANGED))

    return res


def GUI_text(frame, viewmode, image_frame, image_filter):
    GUI_desc_color = (145, 85, 29)
    GUI_filter_color = (0, 84, 163)

    # General

    frame = cv2.putText(frame, "Premi + o - per cambiare effetto", (30, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, GUI_desc_color, 1, cv2.LINE_AA, False)

    frame = cv2.putText(frame, "DEBUG MODE (premi x)", (30, 460),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, GUI_desc_color, 1, cv2.LINE_AA, False)

    # Viewmodes

    if viewmode == 0:
        frame = cv2.putText(frame, "disattivata", (230, 460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, GUI_filter_color, 1, cv2.LINE_AA, False)
    if viewmode == 1:
        frame = cv2.putText(frame, "indicatori", (230, 460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, GUI_filter_color, 1, cv2.LINE_AA, False)
    if viewmode == 2:
        frame = cv2.putText(frame, "numerati", (230, 460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, GUI_filter_color, 1, cv2.LINE_AA, False)

    # Filters

    if image_filter == 1:
        frame = cv2.putText(frame, "Rossetto", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 2:
        frame = cv2.putText(frame, "Occhiali (1/2)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 3:
        frame = cv2.putText(frame, "Occhiali (2/2)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 4:
        frame = cv2.putText(frame, "Piercing", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 5:
        frame = cv2.putText(frame, "Septum", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 6:
        frame = cv2.putText(frame, "Lentiggini", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 7:
        frame = cv2.putText(frame, "Blush (1/2)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 8:
        frame = cv2.putText(frame, "Blush (2/2)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 9:
        frame = cv2.putText(frame, "Barba", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 10:
        frame = cv2.putText(frame, "Combo: blush + lentiggini", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 11:
        frame = cv2.putText(frame, "Combo: rossetto + lentiggini", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 12:
        frame = cv2.putText(frame, "Combo: blush + rossetto", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    elif image_filter == 13:
        frame = cv2.putText(frame, "Combo: piercing + septum", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)
    else:
        frame = cv2.putText(frame, "Disattivato", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, GUI_filter_color, 1, cv2.LINE_AA, False)

    return frame


def main():
    saving_msg = 0
    landmark_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Menù

    viewmode = 0
    image_filter = 0
    image_frame = 0

    vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        faces = landmark_detector(frame)

        if keyboard.is_pressed('x'):
            viewmode = (viewmode + 1) % 3
        if keyboard.is_pressed('+'):
            image_filter = (image_filter + 1) % 14
        if keyboard.is_pressed('-'):
            image_filter = (image_filter - 1) % 14
        if keyboard.is_pressed('f'):
            image_frame = (image_frame + 1) % 6
        if keyboard.is_pressed('s'):
            letters = string.ascii_lowercase
            now = datetime.now()
            timeDate = now.strftime("%d%m%y_%H%M%S")
            filename = "saved/" + timeDate + "_" + ''.join(random.choice(letters) for i in range(3)) + ".png"
            cv2.imwrite(filename, saving_frame)
            saving_msg = 1
        if keyboard.is_pressed('q'):
            break

        final_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        for face in faces:
            landmarks = landmark_predictor(frame, face)
            final_frame = apply_mask(image_filter, final_frame, landmarks)

        final_frame = apply_frame(image_frame, final_frame)
        saving_frame = final_frame

        final_frame = overlay_png(final_frame, cv2.imread("GUI/GUI.png", cv2.IMREAD_UNCHANGED))

        if saving_msg:
            final_frame = overlay_png(final_frame, cv2.imread("GUI/SAVE_overlay.png", cv2.IMREAD_UNCHANGED))
            final_frame = cv2.putText(final_frame, "Immagine salvata in " + filename, (60, 410),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (0, 255, 255), 1, cv2.LINE_AA, False)

        final_frame = GUI_text(final_frame, viewmode, image_frame, image_filter)

        for face in faces:
            for i in range(68):
                if viewmode == 0:
                    pass
                if viewmode == 1:
                    final_frame = cv2.circle(final_frame, (landmarks.part(i).x, landmarks.part(i).y), radius=2,
                                             color=(0, 0, 255),
                                             thickness=1)
                if viewmode == 2:
                    final_frame = cv2.putText(final_frame, str(i), (landmarks.part(i).x, landmarks.part(i).y),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.2, (211, 211, 211), 1, cv2.LINE_AA, False)
                else:
                    continue

        cv2.imshow("main", final_frame)
        cv2.waitKey(1)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        exit(1)
