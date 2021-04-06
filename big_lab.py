import csv
import os
import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080


def showAndSaveImage(img, scale_percent=30, pic_name=None):
    if pic_name is not None:
        cv2.imwrite("results\\" + pic_name, img)
    else:
        dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img_resize)


def prepareImage(img, flag_viewing):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2.)
    img_canny = cv2.Canny(img_blur, 10, 120)
    kernel = np.ones((5, 5), np.uint8)
    img_closed = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    img_show = np.hstack((cv2.resize(img_gray, (0, 0), None, .15, .15), cv2.resize(img_blur, (0, 0), None, .15, .15),
                          cv2.resize(img_canny, (0, 0), None, .15, .15), cv2.resize(img_closed, (0, 0), None, .15, .15)))
    if flag_viewing:
        cv2.imshow("Current image", cv2.resize(img, (0, 0), None, .25, .25))
        cv2.waitKey(0)
        cv2.imshow("Preprocessing", img_show)
        cv2.waitKey(0)
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


def myFindHomography(img_standard, img_target, kp_standard, des_standard):
    detector = cv2.ORB_create()

    if kp_standard is None and des_standard is None:
        kp1, des1 = detector.detectAndCompute(img_standard, None)
        kp_standard = kp1
        des_standard = des1
    else:
        kp1 = kp_standard
        des1 = des_standard

    kp2, des2 = detector.detectAndCompute(img_target, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    coef_ratio = 0.95
    good_matches = []
    for m, n in knn_matches:
        if m.distance < coef_ratio * n.distance:
            good_matches.append(m)

    H = None

    if len(good_matches) >= 4:
        src = np.empty((len(good_matches), 2), dtype=np.float32)
        dst = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            src[i] = kp1[good_matches[i].queryIdx].pt
            dst[i] = kp2[good_matches[i].trainIdx].pt
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H


def getContours(img, flag_viewing):
    img_work = img.copy()
    # Предобработка изображения
    img_prepare = prepareImage(img, flag_viewing)

    contours, h = cv2.findContours(img_prepare, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_to_draw = 2 * [None]
    # Ищем контуры рамки и листа
    for i in range(len(contours)):
        # Потенциальная площадь контура рамки
        S_int = cv2.contourArea(contours[i])
        if S_int > 100:

            # Аппроксимируем теущий контур
            P_int = cv2.arcLength(contours[i], True)
            points_internal_fig = cv2.approxPolyDP(contours[i], 0.02 * P_int, True)
            # Если у апр. контура 4 вершины и в иерархии у контура есть предок

            if len(points_internal_fig) == 4 and h[0][i][3] != -1:

                # Значит, предок может быть контуром листа
                # В структуре иерархии под h[0][i][3] лежит индекс контура предка

                S_ext = cv2.contourArea(contours[h[0][i][3]])
                P_ext = cv2.arcLength(contours[h[0][i][3]], True)
                points_external_fig = cv2.approxPolyDP(contours[h[0][i][3]], 0.02 * P_ext, True)

                # Если у апр. контура предка 4 вершины и площадь внешнего подчиняется данным неравенствам (подобраны эмпирически)
                if len(points_external_fig) == 4 and 2.2 * S_int > S_ext > 1.8 * S_int:
                    # Сохраняем контуры
                    contours_to_draw[0] = points_internal_fig
                    contours_to_draw[1] = points_external_fig

    for i in range(len(contours_to_draw)):
        if contours_to_draw[i] is not None:
            cv2.drawContours(img_work, [contours_to_draw[i]], -1, (255, 0, 0), 2)

    res_img = np.hstack((cv2.resize(img, (0, 0), None, .25, .25), cv2.resize(img_work, (0, 0), None, .25, .25)))
    cv2.imshow("Found contours [ATTENTION TO THE CONSOLE]", res_img)
    cv2.waitKey(0)

    ans = input('На данном изображении корректно найдены контуры рамки и листа? (Y/N): ')
    if ans.upper().find("Y") != -1 or ans.upper().find("Н") != -1:
        return contours_to_draw
    else:
        return None


def getMask(img, contour):
    mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    mask_img_standard = cv2.bitwise_and(img, img, mask=mask)
    return mask_img_standard


def searchImageStandard(images, path_name, flag_viewing=False):
    for img_name in images:
        img = cv2.imread(path_name + "\\" + img_name)
        contours = getContours(img, flag_viewing)
        if contours is not None:
            img_standard_name = img_name
            # Сделаем маску, чтобы искать гомографию именно по особым точкам листа
            mask_img_standard = getMask(img, contours[1])
            if flag_viewing:
                cv2.imshow('Mask along the resulting contour of the sheet', cv2.resize(mask_img_standard, (0, 0), None, .25, .25))
                cv2.waitKey(0)
            return img_standard_name, mask_img_standard, contours
    return None


def searchEllipse(contours):
    ellipse = None
    for cnt in contours:
        if len(cnt) > 4:
            # Используем соображении о возможной площади необходимого эллипса
            if 8500 > cv2.contourArea(cnt) > 7000:
                ellipse = cv2.fitEllipse(cnt)
                # Соображение о соотношении длины большой оси к меньшей
                if not 0.675 <= ellipse[1][1] / ellipse[1][0] <= 1.6:
                    ellipse = None
    return ellipse


def drawAndSave(img, contours_list, ellipse, img_name=None):
    if ellipse is not None:
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)
    # Отрисовка контуров рамки и листа
    cv2.drawContours(img, [contours_list[0]], -1, (255, 0, 0), 2)
    cv2.drawContours(img, [contours_list[1]], -1, (255, 0, 0), 2)

    # Построение направления на камеру из центра листа
    M = cv2.moments(contours_list[0])

    point = intersection(contours_list[0][0][0], contours_list[0][1][0], contours_list[0][2][0],
                         contours_list[0][3][0])
    if point is not None:
        cv2.arrowedLine(img, (int(point[0]), int(point[1])),
                        (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])),
                        (0, 255, 0), 10)
    if img_name is not None:
        showAndSaveImage(img, pic_name=img_name)


def processImage(images, path_name, img_standard_name, mask_img_standard, contours, flag_viewing=False):
    # Счётчик найденных гомографий
    count_found_H = 0
    ans = []
    kp_standard, des_standard = None, None
    for img_name in images:
        # Не строим гомографию для эталона
        if img_standard_name not in img_name:
            img = cv2.imread(path_name + "\\" + img_name)
            H = myFindHomography(mask_img_standard, img, kp_standard, des_standard)
            if H is not None:
                count_found_H += 1
                # Если матрица гомографии не None, то можем найти координаты контуров листа и рамки на новом изображении с помощью преобразования
                contours_list = []
                for contour in contours:
                    points_list = np.zeros([4, 1, 2])
                    for i in range(len(contour)):
                        c = np.dot(H, np.append(contour[i], 1))
                        c /= c[2]
                        points_list[i] = [c[:2]]
                    contours_list.append(points_list.astype(int))

                # Будем искать эллипс в зоне листа
                ellipse = None
                mask = getMask(img, contours_list[1])
                img_prepare = prepareImage(mask, False)
                cont, h = cv2.findContours(img_prepare, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(cont) != 1:
                    ellipse = searchEllipse(cont)
                    drawAndSave(img, contours_list, ellipse,  img_name=img_name)
                    if ellipse is not None:
                        if checkEllipseInRect(contours_list[0], ellipse):
                            ans += [[img_name, "Yes", "Yes", "Yes", "Full"]]
                        else:
                            ans += [[img_name, "Yes", "Yes", "No", "Full"]]
                    else:
                        ans += [[img_name, "No", "No", "No", "-"]]
                else:
                    ans += [[img_name, "No", "No", "No", "-"]]
            else:
                ans += [[img_name, "No", "No", "No", "-"]]

        else:
            ellipse = None
            img_prepare = prepareImage(mask_img_standard, False)
            cont, h = cv2.findContours(img_prepare, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ellipse = searchEllipse(cont)
            img = cv2.imread(path_name + "\\" + img_standard_name)
            drawAndSave(img, contours, ellipse, img_name=img_standard_name)
            if ellipse is not None:
                if checkEllipseInRect(contours[0], ellipse):
                    ans += [[img_name, "Yes", "Yes", "Yes", "Full"]]
                else:
                    ans += [[img_name, "Yes", "Yes", "No", "Full"]]
            else:
                ans += [[img_name, "No", "No", "No", "-"]]
    return ans, count_found_H


def liveMode():
    # В случае live режима стоит закрепить разрешение выдаваемой картинки
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    path_name = "Data_jpg"
    files = os.listdir(path=path_name)
    images = [i for i in filter(lambda x: x.endswith('.jpg'), files)]

    # Определим изображение эталон, с помощью которого в дальнейшем будем строить гомографию
    # Потенциально "хорошее" изображение специально стоит первым
    img_standard_name, mask_img_standard, contours = searchImageStandard(images, path_name)

    while True:
        suc, img = cap.read()
        H = myFindHomography(mask_img_standard, img)
        if H is not None:
            # Если матрица гомографии не None, то можем найти координаты контуров листа и рамки на новом изображении с помощью преобразования
            contours_list = []
            for contour in contours:
                points_list = np.zeros([4, 1, 2])
                for i in range(len(contour)):
                    c = np.dot(H, np.append(contour[i], 1))
                    c /= c[2]
                    points_list[i] = [c[:2]]
                contours_list.append(points_list.astype(int))

            # Будем искать эллипс в зоне листа
            ellipse = None
            mask = getMask(img, contours_list[1])
            img_prepare = prepareImage(mask, False)
            cont, h = cv2.findContours(img_prepare, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ellipse = searchEllipse(cont)
            drawAndSave(img, contours_list, ellipse)
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def defaultMode(flag_viewing):
    # Для проверки корректности работы создадим список с названиями столбцов будущей таблицы csv формата
    answer = [["ImageName", "FigureExist", "FigureInPlaneSheet", "FigureInFrame",
               "FigureShape"]]
    path_name = "Data_jpg"
    files = os.listdir(path=path_name)
    images = [i for i in filter(lambda x: x.endswith('.jpg'), files)]

    # Определим изображение эталон, с помощью которого в дальнейшем будем строить гомографию
    # Потенциально "хорошее" изображение специально стоит первым
    img_standard_name, mask_img_standard, contours = searchImageStandard(images, path_name, flag_viewing)

    if contours is None:
        print("Из множества изображений не нашлось эталона")
        return

    # Теперь приступим к обработке всех изображений
    answ, count_found_H = processImage(images, path_name, img_standard_name, mask_img_standard, contours, flag_viewing)
    answer += answ
    print("В " + str(count_found_H) + "/" + str(len(images) - 1) + " изображений найдена гомография")

    with open("results\\answers.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=";", lineterminator="\r")
        for ans in answer:
            file_writer.writerow(ans)


if __name__ == '__main__':
    # Определим есть ли в Вашем устройстве камера
    cap = cv2.VideoCapture(0)
    flagMode = False
    if cap.isOpened():
        # Если да, то Вам будет предложено использовать live режим
        s = input("У Вас обнаружена камера.\nЖелаете использовать live режим? (Y/N): ")
        if s.upper().find("Y") != -1 or s.upper().find("Н") != -1:  # Поддержка русской "Н" на случай не той включённой раскладки
            liveMode()
        else:
            flagMode = True
    else:
        flagMode = True
    if flagMode:
        s = input("Желаете ли Вы смотреть на предобработки изображения? (Y/N): ")
        flag = False
        if s.upper().find("Y") != -1 or s.upper().find("Н") != -1:
            flag = True
        defaultMode(flag)
