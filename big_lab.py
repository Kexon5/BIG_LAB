import cv2
import numpy as np
import csv

WIDTH = 1920
HEIGHT = 1080


def showAndSaveImage(img, scale_percent=30, pic_name=None):
    if pic_name is not None:
        cv2.imwrite("results\\" + pic_name, img)
    else:
        dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img_resize)


def prepareImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1.6)
    img_canny = cv2.Canny(img_blur, 10, 120)
    kernel = np.ones((5, 5), np.uint8)
    img_closed = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    return img_closed


def det2(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]


def intersection(point1, point2, point3, point4):
    d1 = point3 - point1
    d2 = point4 - point2

    det = det2(d1, d2)
    if det == 0:
        return None
    t = det2(point2 - point1, d2) / det

    return point1 + t * d1


def checkEllipseInRect(rect, ellipse):
    v1 = [rect[1][0][0] - rect[0][0][0], rect[1][0][1] - rect[0][0][1]]
    v2 = [rect[3][0][0] - rect[0][0][0], rect[3][0][1] - rect[0][0][1]]

    v_check = [ellipse[0][0] - rect[0][0][0], ellipse[0][1] - rect[0][0][1]]

    det = det2(v1, v2)

    if abs(det) < 1e-6:
        return False

    k = (v_check[0] * v2[1] - v_check[1] * v2[0]) / det
    f = (v1[0] * v_check[1] - v_check[0] * v1[1]) / det

    if 0 <= k <= 1 and 0 <= f <= 1:
        return True
    else:
        return False


def getContours(img, cam_mode=False):
    imgC = prepareImage(img)
    contours, h = cv2.findContours(imgC, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    P_min = 1E+15
    contours_to_draw = 2 * [None]
    fig_in_contour = 2 * [False]
    ellipse = None

    for i in range(len(contours)):
        if len(contours[i]) > 4 and h[0][i][3] != -1 and h[0][i][2] != -1:
            P_ell = cv2.arcLength(contours[i], True)
            if P_ell < P_min:
                P_min = P_ell
                ellipse = cv2.fitEllipse(contours[i])

        S_int = cv2.contourArea(contours[i])
        if S_int > 100:
            P_int = cv2.arcLength(contours[i], True)
            points_internal_fig = cv2.approxPolyDP(contours[i], 0.02 * P_int, True)
            if len(points_internal_fig) == 4 and h[0][i][3] != -1:
                S_ext = cv2.contourArea(contours[h[0][i][3]])
                P_ext = cv2.arcLength(contours[h[0][i][3]], True)
                points_external_fig = cv2.approxPolyDP(contours[h[0][i][3]], 0.02 * P_ext, True)

                if len(points_external_fig) == 4 and 2.2 * S_int > S_ext > 1.8 * S_int:
                    contours_to_draw[0] = points_internal_fig
                    contours_to_draw[1] = points_external_fig

    for i in range(len(contours_to_draw)):
        if contours_to_draw[i] is not None:
            if ellipse is not None:
                fig_in_contour[i] = checkEllipseInRect(contours_to_draw[i], ellipse)
            cv2.drawContours(img, [contours_to_draw[i]], -1, (255, 0, 0), 5)

    if ellipse is not None:
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)

    if contours_to_draw[0] is not None:
        M = cv2.moments(contours_to_draw[0])

        cv2.line(img, (contours_to_draw[0][0][0][0], contours_to_draw[0][0][0][1]),
                 (contours_to_draw[0][2][0][0], contours_to_draw[0][2][0][1]), (255, 0, 0), 1)
        cv2.line(img, (contours_to_draw[0][1][0][0], contours_to_draw[0][1][0][1]),
                 (contours_to_draw[0][3][0][0], contours_to_draw[0][3][0][1]), (255, 0, 0), 1)

        point = intersection(contours_to_draw[0][0][0], contours_to_draw[0][1][0], contours_to_draw[0][2][0],
                             contours_to_draw[0][3][0])
        if point is not None:
            cv2.arrowedLine(img, (int(point[0]), int(point[1])), (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])),
                            (0, 0, 255), 5)

    if cam_mode:
        showAndSaveImage(img, scale_percent=100)
    else:
        showAndSaveImage(img, pic_name=img_name)
    return [[img_name, "?", contours_to_draw[0] is not None, ellipse is not None, fig_in_contour[1], fig_in_contour[0],
            "Full ellipse" if ellipse is not None else "Nothing"]]


cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    while True:
        suc, img = cap.read()
        getContours(img, cam_mode=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    answer = [["ImageName", "DirectionToCamera", "FrameExist", "FigureExist", "FigureInPlaneSheet", "FigureInFrame",
              "FigureShape"]]
    for i in range(1, 31):
        img_name = str(i) + ".jpg"
        img = cv2.imread("Data_jpg\\" + img_name)
        answer += getContours(img)

    with open("results\\answers.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=";", lineterminator="\r")
        for ans in answer:
            file_writer.writerow(ans)
