from PyQt5.QtGui import QPixmap
import numpy as np
from window import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from hand import HandDetector
# from predict import predict
import time
import copy
import argparse
import cv2
# import numpy as np
import mediapipe as mp
# import csv
import pandas as pd
import pickle
# from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh* min_detection_confidence',
                        type=float,
                        default=0.45)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh* min_tracking_confidence',
                        type=int,
                        default=0.33)

    parser.add_argument('--use_brect', type=bool, default=False)
    parser.add_argument('--plot_world_landmark', type=bool, default=False)
    
    args = parser.parse_args()

    return args

def draw_hands_landmarks(
        draw,
        image,
        cx,
        cy,
        landmarks,
        # upper_body_only,
        handedness_str='R'):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        draw_color = (204,255,102)
        draw_with_detection = draw


        if index == 0:  # 手首1
            cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 1:  # 手首2
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 2:  # 親指：付け根
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 3:  # 親指：第1関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 4:  # 親指：指先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        #     cv2.circle(image, (landmark_x, landmark_y), 12, draw_color, 2)
        # if index == 5:  # 人差指：付け根
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 6:  # 人差指：第2関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 7:  # 人差指：第1関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 8:  # 人差指：指先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        #     cv2.circle(image, (landmark_x, landmark_y), 12, draw_color, 2)
        # if index == 9:  # 中指：付け根
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 10:  # 中指：第2関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 11:  # 中指：第1関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, draw_with_detection, 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, draw_with_detection, 2)
        # if index == 13:  # 薬指：付け根
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 14:  # 薬指：第2関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 15:  # 薬指：第1関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 16:  # 薬指：指先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        #     cv2.circle(image, (landmark_x, landmark_y), 12, draw_color, 2)
        # if index == 17:  # 小指：付け根
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 18:  # 小指：第2関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 19:  # 小指：第1関節
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        # if index == 20:  # 小指：指先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        #     cv2.circle(image, (landmark_x, landmark_y), 12, draw_color, 2)

        # if not upper_body_only:
        if index == 12:
            cv2.putText(image, "y:" + str(round(landmark_y, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,250), 1,
                       cv2.LINE_AA)

    # 接続線
    # if len(landmark_point) > 0:
    #     # 親指
    #     cv2.line(image, landmark_point[2], landmark_point[3], draw_color, 2)
    #     cv2.line(image, landmark_point[3], landmark_point[4], draw_color, 2)

    #     # 人差指
    #     cv2.line(image, landmark_point[5], landmark_point[6], draw_color, 2)
    #     cv2.line(image, landmark_point[6], landmark_point[7], draw_color, 2)
    #     cv2.line(image, landmark_point[7], landmark_point[8], draw_color, 2)

    #     # 中指
    #     cv2.line(image, landmark_point[9], landmark_point[10], draw_color, 2)
    #     cv2.line(image, landmark_point[10], landmark_point[11], draw_color, 2)
    #     cv2.line(image, landmark_point[11], landmark_point[12], draw_color, 2)

    #     # 薬指
    #     cv2.line(image, landmark_point[13], landmark_point[14], draw_color, 2)
    #     cv2.line(image, landmark_point[14], landmark_point[15], draw_color, 2)
    #     cv2.line(image, landmark_point[15], landmark_point[16], draw_color, 2)

    #     # 小指
    #     cv2.line(image, landmark_point[17], landmark_point[18], draw_color, 2)
    #     cv2.line(image, landmark_point[18], landmark_point[19], draw_color, 2)
    #     cv2.line(image, landmark_point[19], landmark_point[20], draw_color, 2)

    #     # 手の平
    #     cv2.line(image, landmark_point[0], landmark_point[1], draw_color, 2)
    #     cv2.line(image, landmark_point[1], landmark_point[2], draw_color, 2)
    #     cv2.line(image, landmark_point[2], landmark_point[5], draw_color, 2)
    #     cv2.line(image, landmark_point[5], landmark_point[9], draw_color, 2)
    #     cv2.line(image, landmark_point[9], landmark_point[13], draw_color, 2)
    #     cv2.line(image, landmark_point[13], landmark_point[17], draw_color, 2)
    #     cv2.line(image, landmark_point[17], landmark_point[0], draw_color, 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        cv2.circle(image, (cx, cy), 12, draw_color, 2)
        cv2.putText(image, handedness_str, (cx - 6, cy + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 250), 2, cv2.LINE_AA)

    return image

def draw_face_landmarks(draw, image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))
        target_num_p = [5 ,168 ,200]
        draw_color = draw

        if index in target_num_p:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,250), 1,
                       cv2.LINE_AA)

        # cv2.circle(image, (landmark_x, landmark_y), 1, draw_color, 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg        
        
        # 中線(168、5、200)
        cv2.line(image, landmark_point[10], landmark_point[151], draw_color, 2)
        cv2.line(image, landmark_point[151], landmark_point[9], draw_color, 2)
        cv2.line(image, landmark_point[9], landmark_point[8], draw_color, 2)
        cv2.line(image, landmark_point[8], landmark_point[168], draw_color, 2)
        cv2.line(image, landmark_point[168], landmark_point[6], draw_color, 2)
        cv2.line(image, landmark_point[6], landmark_point[197], draw_color, 2)
        cv2.line(image, landmark_point[197], landmark_point[195], draw_color, 2)
        cv2.line(image, landmark_point[195], landmark_point[5], draw_color, 2)
        cv2.line(image, landmark_point[5], landmark_point[4], draw_color, 2)
        cv2.line(image, landmark_point[4], landmark_point[1], draw_color, 2)
        cv2.line(image, landmark_point[1], landmark_point[19], draw_color, 2)
        cv2.line(image, landmark_point[19], landmark_point[94], draw_color, 2)
        cv2.line(image, landmark_point[94], landmark_point[2], draw_color, 2)
        cv2.line(image, landmark_point[2], landmark_point[164], draw_color, 2)
        cv2.line(image, landmark_point[164], landmark_point[0], draw_color, 2)
        cv2.line(image, landmark_point[0], landmark_point[11], draw_color, 2)
        cv2.line(image, landmark_point[11], landmark_point[12], draw_color, 2)
        cv2.line(image, landmark_point[12], landmark_point[13], draw_color, 2)
        cv2.line(image, landmark_point[13], landmark_point[14], draw_color, 2)
        cv2.line(image, landmark_point[14], landmark_point[15], draw_color, 2)
        cv2.line(image, landmark_point[15], landmark_point[16], draw_color, 2)
        cv2.line(image, landmark_point[16], landmark_point[17], draw_color, 2)
        cv2.line(image, landmark_point[17], landmark_point[18], draw_color, 2)
        cv2.line(image, landmark_point[18], landmark_point[200], draw_color, 2)
        cv2.line(image, landmark_point[200], landmark_point[199], draw_color, 2)
        cv2.line(image, landmark_point[199], landmark_point[175], draw_color, 2)
        cv2.line(image, landmark_point[175], landmark_point[152], draw_color, 2)

        # Cross Line
        cv2.line(image, landmark_point[234], landmark_point[227], draw_color, 2)
        cv2.line(image, landmark_point[116], landmark_point[117], draw_color, 2)
        cv2.line(image, landmark_point[118], landmark_point[100], draw_color, 2)
        cv2.line(image, landmark_point[126], landmark_point[198], draw_color, 2)
        cv2.line(image, landmark_point[134], landmark_point[51], draw_color, 2)
        cv2.line(image, landmark_point[281], landmark_point[363], draw_color, 2)
        cv2.line(image, landmark_point[420], landmark_point[355], draw_color, 2)
        cv2.line(image, landmark_point[329], landmark_point[347], draw_color, 2)
        cv2.line(image, landmark_point[346], landmark_point[345], draw_color, 2)
        cv2.line(image, landmark_point[447], landmark_point[454], draw_color, 2)
        
        
        # # 左眉毛(55：内側、46：外側)
        # cv2.line(image, landmark_point[55], landmark_point[65], draw_color, 2)
        # cv2.line(image, landmark_point[65], landmark_point[52], draw_color, 2)
        # cv2.line(image, landmark_point[52], landmark_point[53], draw_color, 2)
        # cv2.line(image, landmark_point[53], landmark_point[46], draw_color, 2)

        # # 右眉毛(285：内側、276：外側)
        # cv2.line(image, landmark_point[285], landmark_point[295], draw_color,
        #         2)
        # cv2.line(image, landmark_point[295], landmark_point[282], draw_color,
        #         2)
        # cv2.line(image, landmark_point[282], landmark_point[283], draw_color,
        #         2)
        # cv2.line(image, landmark_point[283], landmark_point[276], draw_color,
        #         2)

        # # 左目 (133：目頭、246：目尻)
        # cv2.line(image, landmark_point[133], landmark_point[173], draw_color,
        #         2)
        # cv2.line(image, landmark_point[173], landmark_point[157], draw_color,
        #         2)
        # cv2.line(image, landmark_point[157], landmark_point[158], draw_color,
        #         2)
        # cv2.line(image, landmark_point[158], landmark_point[159], draw_color,
        #         2)
        # cv2.line(image, landmark_point[159], landmark_point[160], draw_color,
        #         2)
        # cv2.line(image, landmark_point[160], landmark_point[161], draw_color,
        #         2)
        # cv2.line(image, landmark_point[161], landmark_point[246], draw_color,
        #         2)

        # cv2.line(image, landmark_point[246], landmark_point[163], draw_color,
        #         2)
        # cv2.line(image, landmark_point[163], landmark_point[144], draw_color,
        #         2)
        # cv2.line(image, landmark_point[144], landmark_point[145], draw_color,
        #         2)
        # cv2.line(image, landmark_point[145], landmark_point[153], draw_color,
        #         2)
        # cv2.line(image, landmark_point[153], landmark_point[154], draw_color,
        #         2)
        # cv2.line(image, landmark_point[154], landmark_point[155], draw_color,
        #         2)
        # cv2.line(image, landmark_point[155], landmark_point[133], draw_color,
        #         2)

        # # 右目 (362：目頭、466：目尻)
        # cv2.line(image, landmark_point[362], landmark_point[398], draw_color,
        #         2)
        # cv2.line(image, landmark_point[398], landmark_point[384], draw_color,
        #         2)
        # cv2.line(image, landmark_point[384], landmark_point[385], draw_color,
        #         2)
        # cv2.line(image, landmark_point[385], landmark_point[386], draw_color,
        #         2)
        # cv2.line(image, landmark_point[386], landmark_point[387], draw_color,
        #         2)
        # cv2.line(image, landmark_point[387], landmark_point[388], draw_color,
        #         2)
        # cv2.line(image, landmark_point[388], landmark_point[466], draw_color,
        #         2)

        # cv2.line(image, landmark_point[466], landmark_point[390], draw_color,
        #         2)
        # cv2.line(image, landmark_point[390], landmark_point[373], draw_color,
        #         2)
        # cv2.line(image, landmark_point[373], landmark_point[374], draw_color,
        #         2)
        # cv2.line(image, landmark_point[374], landmark_point[380], draw_color,
        #         2)
        # cv2.line(image, landmark_point[380], landmark_point[381], draw_color,
        #         2)
        # cv2.line(image, landmark_point[381], landmark_point[382], draw_color,
        #         2)
        # cv2.line(image, landmark_point[382], landmark_point[362], draw_color,
        #         2)

        # # 口 (308：右端、78：左端)
        # cv2.line(image, landmark_point[308], landmark_point[415], draw_color,
        #         2)
        # cv2.line(image, landmark_point[415], landmark_point[310], draw_color,
        #         2)
        # cv2.line(image, landmark_point[310], landmark_point[311], draw_color,
        #         2)
        # cv2.line(image, landmark_point[311], landmark_point[312], draw_color,
        #         2)
        # cv2.line(image, landmark_point[312], landmark_point[13], draw_color, 2)
        # cv2.line(image, landmark_point[13], landmark_point[82], draw_color, 2)
        # cv2.line(image, landmark_point[82], landmark_point[81], draw_color, 2)
        # cv2.line(image, landmark_point[81], landmark_point[80], draw_color, 2)
        # cv2.line(image, landmark_point[80], landmark_point[191], draw_color, 2)
        # cv2.line(image, landmark_point[191], landmark_point[78], draw_color, 2)

        # cv2.line(image, landmark_point[78], landmark_point[95], draw_color, 2)
        # cv2.line(image, landmark_point[95], landmark_point[88], draw_color, 2)
        # cv2.line(image, landmark_point[88], landmark_point[178], draw_color, 2)
        # cv2.line(image, landmark_point[178], landmark_point[87], draw_color, 2)
        # cv2.line(image, landmark_point[87], landmark_point[14], draw_color, 2)
        # cv2.line(image, landmark_point[14], landmark_point[317], draw_color, 2)
        # cv2.line(image, landmark_point[317], landmark_point[402], draw_color,
        #         2)
        # cv2.line(image, landmark_point[402], landmark_point[318], draw_color,
        #         2)
        # cv2.line(image, landmark_point[318], landmark_point[324], draw_color,
        #         2)
        # cv2.line(image, landmark_point[324], landmark_point[308], draw_color,
        #         2)
            
    return image

def draw_pose_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
    ):
    image_width, image_height = image.shape[1], image.shape[0]

    
    landmark_color = (204,255,102)
    link_color = (0,255,0)
    landmark_point = []
    # indicate cover region
    cv2.line(image,(200, 0),(200, 691),(204, 255, 102), 1, cv2.LINE_AA)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        # if index == 0:  # 鼻
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 1:  # 右目：目頭
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 2:  # 右目：瞳
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 3:  # 右目：目尻
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 4:  # 左目：目頭
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 5:  # 左目：瞳
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 6:  # 左目：目尻
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 7:  # 右耳
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 8:  # 左耳
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 9:  # 口：左端
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 10:  # 口：左端
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 11:  # 右肩
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 12:  # 左肩
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 13:  # 右肘
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 14:  # 左肘
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 15:  # 右手首
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        if index == 16:  # 左手首
            cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 17:  # 右手1(外側端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 18:  # 左手1(外側端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 19:  # 右手2(先端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 20:  # 左手2(先端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 21:  # 右手3(内側端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 22:  # 左手3(内側端)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 23:  # 腰(右側)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 24:  # 腰(左側)
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 25:  # 右ひざ
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 26:  # 左ひざ
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 27:  # 右足首
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 28:  # 左足首
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 29:  # 右かかと
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 30:  # 左かかと
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 31:  # 右つま先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)
        # if index == 32:  # 左つま先
        #     cv2.circle(image, (landmark_x, landmark_y), 5, landmark_color, 2)

        # if not upper_body_only:
        # if index >= 11 and index <= 16:
        #     cv2.putText(image, "z:" + str(round(landmark_z, 3)),
        #             (landmark_x - 10, landmark_y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1,
        #             cv2.LINE_AA)

    # if len(landmark_point) > 0:
    #     # 右目
    #     if landmark_point[1][0] > visibility_th and landmark_point[2][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[1][1], landmark_point[2][1],
    #                 link_color, 2)
    #     if landmark_point[2][0] > visibility_th and landmark_point[3][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[2][1], landmark_point[3][1],
    #                 link_color, 2)

    #     # 左目
    #     if landmark_point[4][0] > visibility_th and landmark_point[5][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[4][1], landmark_point[5][1],
    #                 link_color, 2)
    #     if landmark_point[5][0] > visibility_th and landmark_point[6][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[5][1], landmark_point[6][1],
    #                 link_color, 2)

    #     # 口
    #     if landmark_point[9][0] > visibility_th and landmark_point[10][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[9][1], landmark_point[10][1],
    #                 link_color, 2)

    #     # 肩
    #     if landmark_point[11][0] > visibility_th and landmark_point[12][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[11][1], landmark_point[12][1],
    #                 link_color, 2)

    #     # 右腕
    #     if landmark_point[11][0] > visibility_th and landmark_point[13][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[11][1], landmark_point[13][1],
    #                 link_color, 2)
    #     if landmark_point[13][0] > visibility_th and landmark_point[15][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[13][1], landmark_point[15][1],
    #                 link_color, 2)

    #     # 左腕
    #     if landmark_point[12][0] > visibility_th and landmark_point[14][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[12][1], landmark_point[14][1],
    #                 link_color, 2)
    #     if landmark_point[14][0] > visibility_th and landmark_point[16][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[14][1], landmark_point[16][1],
    #                 link_color, 2)

    #     # 右手
    #     if landmark_point[15][0] > visibility_th and landmark_point[17][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[15][1], landmark_point[17][1],
    #                 link_color, 2)
    #     if landmark_point[17][0] > visibility_th and landmark_point[19][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[17][1], landmark_point[19][1],
    #                 link_color, 2)
    #     if landmark_point[19][0] > visibility_th and landmark_point[21][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[19][1], landmark_point[21][1],
    #                 link_color, 2)
    #     if landmark_point[21][0] > visibility_th and landmark_point[15][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[21][1], landmark_point[15][1],
    #                 link_color, 2)

    #     # 左手
    #     if landmark_point[16][0] > visibility_th and landmark_point[18][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[16][1], landmark_point[18][1],
    #                 link_color, 2)
    #     if landmark_point[18][0] > visibility_th and landmark_point[20][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[18][1], landmark_point[20][1],
    #                 link_color, 2)
    #     if landmark_point[20][0] > visibility_th and landmark_point[22][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[20][1], landmark_point[22][1],
    #                 link_color, 2)
    #     if landmark_point[22][0] > visibility_th and landmark_point[16][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[22][1], landmark_point[16][1],
    #                 link_color, 2)

    #     # 胴体
    #     if landmark_point[11][0] > visibility_th and landmark_point[23][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[11][1], landmark_point[23][1],
    #                 link_color, 2)
    #     if landmark_point[12][0] > visibility_th and landmark_point[24][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[12][1], landmark_point[24][1],
    #                 link_color, 2)
    #     if landmark_point[23][0] > visibility_th and landmark_point[24][
    #             0] > visibility_th:
    #         cv2.line(image, landmark_point[23][1], landmark_point[24][1],
    #                 link_color, 2)

    #     if len(landmark_point) > 25:
    #         # 右足
    #         if landmark_point[23][0] > visibility_th and landmark_point[25][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[23][1], landmark_point[25][1],
    #                     link_color, 2)
    #         if landmark_point[25][0] > visibility_th and landmark_point[27][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[25][1], landmark_point[27][1],
    #                     link_color, 2)
    #         if landmark_point[27][0] > visibility_th and landmark_point[29][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[27][1], landmark_point[29][1],
    #                     link_color, 2)
    #         if landmark_point[29][0] > visibility_th and landmark_point[31][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[29][1], landmark_point[31][1],
    #                     link_color, 2)

    #         # 左足
    #         if landmark_point[24][0] > visibility_th and landmark_point[26][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[24][1], landmark_point[26][1],
    #                     link_color, 2)
    #         if landmark_point[26][0] > visibility_th and landmark_point[28][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[26][1], landmark_point[28][1],
    #                     link_color, 2)
    #         if landmark_point[28][0] > visibility_th and landmark_point[30][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[28][1], landmark_point[30][1],
    #                     link_color, 2)
    #         if landmark_point[30][0] > visibility_th and landmark_point[32][
    #                 0] > visibility_th:
    #             cv2.line(image, landmark_point[30][1], landmark_point[32][1],
    #                     link_color, 2)
    
    return image

def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy

def detection(self,image,lh_l, fa_l, po_l):

    with open('body_language_007_1.pkl', 'rb') as f:
        model = pickle.load(f)

    lh_list = [0,12]
    f_list = [10,151,9,8,168,6,197,195,5,4,1,19,94,2,164,0,11,12,13,14,15,16,17,18,200,199,175,152,
    234,227,116,117,118,100,126,198,134,51,281,363,420,355,329,347,346,345,447,454]
    p_list = [11,12,13,14,15,16,19,20]

    # Extract Pose landmarks  
    pose = po_l.landmark
    pose_row = list(np.array(
        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    pose_row_d = list(range(24))
    j = 0

    for i in range(21):
        if i in p_list:
            pose_row_d[j] = 5*pose_row[4*i]
            pose_row_d[j+1] = 5*pose_row[4*i+1]
            pose_row_d[j+2] = 5*pose_row[4*i+2]
            # pose_row_d[j+3] = pose_row[4*i+3]
            j = j+3


    # Extract Face landmarks
    face = fa_l.landmark
    face_row = list(np.array(
        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
    face_row_d = list(range(144))
    k = 0

    for n in range(455):
        if n in f_list:
            face_row_d[k] = 5*face_row[4*n]
            face_row_d[k+1] = 5*face_row[4*n+1]
            face_row_d[k+2] = 5*face_row[4*n+2]
            # face_row_d[k+3] = face_row[4*n+3]
            k = k+3

    # Extract Left Hand landmarks
    l_hand = lh_l.landmark
    l_hand_row = list(np.array(
        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in l_hand]).flatten())
    l_hand_row_d = list(range(6))
    l = 0

    for m in range(13):
        if m in lh_list:
            l_hand_row_d[l] = 5*l_hand_row[4*m]
            l_hand_row_d[l+1] = 5*l_hand_row[4*m+1]
            l_hand_row_d[l+2] = 5*l_hand_row[4*m+2]
            # l_hand_row_d[l+3] = l_hand_row[4*m+3]
            l = l+3
    # Concate rows
    row = l_hand_row_d + pose_row_d + face_row_d

    # Make Detections
    X = pd.DataFrame([row])
    body_language_class = model.predict(X)[0]
    body_language_prob = model.predict_proba(X)[0]
    MainWindow.print_log(log_words=' Result: [' + body_language_class + ']\n'+str(body_language_prob))
   
    # Grab ear coords
    coords = tuple(np.multiply(
                    np.array(
                        (po_l.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].x, 
                            po_l.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].y))
                , [640,500]).astype(int))
    
    # indicate cover region
    # cv2.line(image,(200, 0),(200, 691),(204, 255, 102), 1, cv2.LINE_AA)
    # cv2.rectangle(image, (0, 0), (200, 691), (204, 255, 102), -1) # cover for user
    cv2.rectangle(image, 
                    (coords[0]-2, coords[1]+5), 
                    (coords[0]+len(body_language_class)*16, coords[1]-20), 
                    (245, 117, 16), -1)
    cv2.putText(image, body_language_class, coords, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    

    # shape = (111, 241, 3) # y, x, RGB

    # # 第一種方法，直接建立全白圖片 100*100
    # img_s = np.full(shape, 255).astype(np.uint8)

    # # 第二種方法，一樣先建立全黑的圖片，再將全部用白色填滿。
    # origin_img = np.zeros(shape, np.uint8)
    # origin_img.fill(255)
    img_s = cv2.imread('image/result_default.png')

    # Get status box
    cv2.rectangle(img_s, (0,0), (242, 112), (255, 255, 255), -1)
    
    if (body_language_class is not None) and (body_language_prob is not None):
        # Display Class
        cv2.putText(img_s, 'CLASS'
                    , (15,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img_s, body_language_class.split(' ')[0]
                    , (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Display Probability
        cv2.putText(img_s, 'PROB'
                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img_s, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                    , (10,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    seconds = time.time()
    de_sec = seconds*10
    count = (de_sec//1)%5
    if count == 0:
        label_show(label=self.labelResult, image=img_s)

    return (image)

def plot_world_landmarks(
    self,
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
    ):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append([landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [2, 5]
    right_arm_index_list = [11, 13, 15]
    left_arm_index_list = [12, 14, 16]
    right_body_side_index_list = [11, 23]
    left_body_side_index_list = [12, 24]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    seconds = time.time()
    de_sec = seconds*10
    count = (de_sec//1)%5
    if count == 0:

        # 顔
        face_x, face_y, face_z = [], [], []
        for index in face_index_list:
            point = landmark_point[index][1]
            face_x.append(point[0])
            face_y.append(point[2])
            face_z.append(point[1] * (-1))

        # 右腕
        right_arm_x, right_arm_y, right_arm_z = [], [], []
        for index in right_arm_index_list:
            point = landmark_point[index][1]
            right_arm_x.append(point[0])
            right_arm_y.append(point[2])
            right_arm_z.append(point[1] * (-1))

        # 左腕
        left_arm_x, left_arm_y, left_arm_z = [], [], []
        for index in left_arm_index_list:
            point = landmark_point[index][1]
            left_arm_x.append(point[0])
            left_arm_y.append(point[2])
            left_arm_z.append(point[1] * (-1))

        # 右半身
        right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
        for index in right_body_side_index_list:
            point = landmark_point[index][1]
            right_body_side_x.append(point[0])
            right_body_side_y.append(point[2])
            right_body_side_z.append(point[1] * (-1))

        # 左半身
        left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
        for index in left_body_side_index_list:
            point = landmark_point[index][1]
            left_body_side_x.append(point[0])
            left_body_side_y.append(point[2])
            left_body_side_z.append(point[1] * (-1))

        # 肩
        shoulder_x, shoulder_y, shoulder_z = [], [], []
        for index in shoulder_index_list:
            point = landmark_point[index][1]
            shoulder_x.append(point[0])
            shoulder_y.append(point[2])
            shoulder_z.append(point[1] * (-1))

        # 腰
        waist_x, waist_y, waist_z = [], [], []
        for index in waist_index_list:
            point = landmark_point[index][1]
            waist_x.append(point[0])
            waist_y.append(point[2])
            waist_z.append(point[1] * (-1))
                
        ax.cla()
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        ax.scatter(face_x, face_y, face_z)
        ax.plot(right_arm_x, right_arm_y, right_arm_z)
        ax.plot(left_arm_x, left_arm_y, left_arm_z)
        ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
        ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
        ax.plot(shoulder_x, shoulder_y, shoulder_z)
        ax.plot(waist_x, waist_y, waist_z)

        plt.savefig.jpeg_quality = 50
        plt.savefig('coords_fig/coords.jpg')
            
        image_3d = cv2.imread('coords_fig/coords.jpg')
        image_3d = image_3d[470:940, 241:482]
        label_show(label=self.labelMAYA, image=image_3d)

    return

def label_show(label, image):
    qt_img_buf = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    qt_img = QtGui.QImage(qt_img_buf.data, qt_img_buf.shape[1], qt_img_buf.shape[0], QtGui.QImage.Format_RGB32)
    image = QPixmap.fromImage(qt_img).scaled(label.width(), label.height())
    label.setPixmap(image)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.image_file = False
        self.setupUi(self)
        self.initUI()
        self.hand_size = 0.6
        self.detector = HandDetector()
        self.videoFPS = 24
        self.image = False
        self.video = False
        self.camera = False
        self.camera_selected = 0
        self.detect = False
        # self.model = predict(model='models/inference_model1')
        self.cap = None
        self.predict = False
        self.mode_key = 0
        self.results = None

    def maya(self, word=None, speed=200, action=True):
        gif = QtGui.QMovie(f'src/{word}.gif')
        self.labelMAYA.setMovie(gif)
        gif.setSpeed(speed)
        if action:
            gif.start()
        else:
            gif.stop()

    def print_log(self, log_words):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.textLog.append(current_time + ': ' + log_words)  # 在指定的区域显示提示信息
        cursor = self.textLog.textCursor()
        self.textLog.moveCursor(cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()

    def initUI(self):
        self.setWindowTitle('Python_Mediapipe_Palmer_Drama')

        self.actionLoadImage.triggered.connect(self.load_image)
        self.actionStart.triggered.connect(self.start_detect)
        self.actionLoadVideo.triggered.connect(self.load_video)
        self.actionVideoFPS.triggered.connect(self.change_VideoFPS)
        self.actionOpenCamera.triggered.connect(self.open_camera)
        self.actionCloseCamera.triggered.connect(self.close_camera)
        self.actionPredict.triggered.connect(self.predict)
        self.actionSelectedCamera.triggered.connect(self.select_camera)
        self.actionHandSize.triggered.connect(self.changeHandSize)
        self.actionModelFile.triggered.connect(self.changeModelFile)
        self.actionMinDetectionConfidence.triggered.connect(self.changeMinDetectionConfidence)
        self.actionMinTrackingConfidence.triggered.connect(self.changeMinTrackingConfidence)
        self.actionExit.triggered.connect(self.exit)

    def exit(self):
        if self.camera == True :
            self.cap.release()
 
        self.maya(action=False)
        label_show(label=self.labelResult, image=cv2.imread('src/result_default.png'))
        label_show(label=self.labelMAYA, image=cv2.imread('src/MAYA_default.png'))
        label_show(label=self.labelImageOrVideo, image=cv2.imread('src/default.png'))
        app = QApplication.instance()
        # 退出应用程序
        app.quit()

    def load_image(self):
        image_name, image_type = QFileDialog.getOpenFileName(self, '打开图片', r'./', '图片 (*.png *.jpg *.jpeg)')
        cv2_image = cv2.imread(image_name)
        label_show(label=self.labelImageOrVideo, image=cv2_image)
        self.image = True
        self.print_log(log_words=f'load image:{image_name}')
        self.image_file = cv2_image

    def load_video(self):
        self.video_name, video_type = QFileDialog.getOpenFileName(self, '打开视频', r'./', '视频 (*.avi *.mp4)')
        self.cap = cv2.VideoCapture(self.video_name)
        self.video = True
        self.print_log(log_words=f'load video:{self.video_name}')
        while True:
            ret, cv2_image = self.cap.read()
            if not ret:
                break
            label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(int(1000 / self.videoFPS))

    def start_detect(self): 
 
        self.camera = True
        # 引数解析 #################################################################
        args = get_args()
       
        model_complexity = args.model_complexity
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        # モデルロード #############################################################
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        camera = cv2.VideoCapture(self.camera_selected)
        self.cap = camera
        self.print_log(log_words=f' [Start Detection] ')
        status = False

        while self.mode_key == 0 :
            ret, image = camera.read()
            if not ret:
                break
            self.camera = True
            # カメラキャプチャ #####################################################
            
            image = cv2.flip(image, 1)  # ミラー表示
            # image = image[0:720, 720:1280] # not functioning with cvtColor
            debug_image = copy.deepcopy(image)
            image_width, image_height = 640, 480

            # 検出実施 #############################################################
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            self.results = holistic.process(image)
            image.flags.writeable = True

            # Face Mesh ###########################################################
            face_landmarks = self.results.face_landmarks
            if face_landmarks is not None:
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, face_landmarks)

                # キーポイント
                for index, landmark in enumerate(face_landmarks.landmark):
                    if landmark.visibility < 0 or landmark.presence < 0:
                        continue

                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    if index == 234:  
                        coords_re = landmark_y

                    if index == 454:  
                        coords_le = landmark_y

                # Check the Height of Left Hand
                if (coords_re > coords_le + 10) or (coords_re < coords_le - 10) :
                    draw_c = (0,0,255)
                
                    cv2.putText(debug_image, 'Try To Keep The Head Straight !'
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2, cv2.LINE_AA)

                else :
                    draw_c = (255,204,102)

                debug_image = draw_face_landmarks(draw_c, debug_image, face_landmarks)
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                

            # Pose ###############################################################
            pose_landmarks = self.results.pose_landmarks
            if pose_landmarks is not None:
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, pose_landmarks)
                # 描画
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # Grab Left shoulder coords
                coords_s = tuple(np.multiply(
                            np.array(
                                (pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].x, 
                                    pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].y))
                        , [image_height,image_width]).astype(int))

                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].x, 
                                        pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].y))
                            , [image_height,image_width]).astype(int))
            
            # Hands ###############################################################
            left_hand_landmarks = self.results.left_hand_landmarks
            right_hand_landmarks = self.results.right_hand_landmarks

            coords_mf_y = 0

            # 左手
            if left_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                # 描画
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 右手
            if right_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # # キーポイント
                for index, landmark in enumerate(right_hand_landmarks.landmark):
                    if landmark.visibility < 0 or landmark.presence < 0:
                        continue

                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    if index == 12:  
                        coords_mf_y = landmark_y

                #  Check the Height of Left Hand
                if coords_mf_y > coords_s[1]+ 100 :
                    draw_c = (255,0,0)
                
                    cv2.rectangle(debug_image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len('Too Low')*20, coords[1]-30), 
                                    (255,0,0), -1)
                    cv2.putText(debug_image, 'Too Low', coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                else :
                    if coords_mf_y < coords_s[1] :
                        draw_c = (0,0,255)
                        
                        cv2.rectangle(debug_image, 
                                        (coords[0], coords[1]+5), 
                                        (coords[0]+len('Too High')*20, coords[1]-30), 
                                        (0,0,255), -1)
                        cv2.putText(debug_image, 'Too High', coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    else :
                        draw_c = (255,204,102)

                        cv2.rectangle(debug_image, 
                                        (coords[0], coords[1]+5), 
                                        (coords[0]+len('Looking Good')*20, coords[1]-30), 
                                        (255,204,102), -1)
                        cv2.putText(debug_image, 'Looking Good', coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # draw_c = (255,204,102) # Default
                debug_image = draw_hands_landmarks(
                    draw_c,
                    debug_image,
                    cx,
                    cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            if (face_landmarks is None) or (pose_landmarks is None) or (right_hand_landmarks is None):
                img_r = cv2.imread('image/result_default.png')
                label_show(label=self.labelResult, image=img_r)
                seconds = time.time()
                de_sec = seconds*10
                count = (de_sec//1)%10
                if count == 0:
                    self.print_log(log_words=f' [Searching for Landmarks] ')
                status = False
            
            else:
                if status == False :
                    img_r = cv2.imread('image/result_default.png')
                    cv2.putText(img_r, 'Ready...'
                    , (15,72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    label_show(label=self.labelResult, image=img_r)
                    self.print_log(log_words=f' [Check!] ')
                    status = True

            label_show(label=self.labelImageOrVideo, image=debug_image)
            # label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(0)
            cv2.waitKey(int(1000 / self.videoFPS))      

        # maya = cv2.imread('src/MAYA_default.png')
        # self.detect = True
        # if self.image:
        #     hand, rectangle, success = self.detector.rectangle(self.image_file, hand_size=self.hand_size)
        #     if success:
        #         label_show(label=self.labelImageOrVideo, image=rectangle)
        #     else:
        #         self.print_log(log_words=f'cant find hand')
        # if self.video:
        #     cap = cv2.VideoCapture(self.video_name)
        #     while True:
        #         ret, image = cap.read()
        #         if not ret:
        #             break
        #         hand, rectangle, success = self.detector.rectangle(image, hand_size=self.hand_size)
        #         label_show(label=self.labelImageOrVideo, image=rectangle)
        #         if success:
        #             label_show(label=self.labelMAYA, image=hand)
        #         else:
        #             label_show(label=self.labelMAYA, image=maya)
        #         cv2.waitKey(int(1000 / self.videoFPS))
        # if self.camera:
        #     self.print_log(log_words=f'using camera detecting')

        #     while True:
        #         ret, image = self.cap.read()
        #         if not ret:
        #             break
        #         hand, rectangle, success = self.detector.rectangle(image, hand_size=self.hand_size)
        #         label_show(label=self.labelImageOrVideo, image=rectangle)
        #         if success:
        #             label_show(label=self.labelMAYA, image=hand)
        #         else:
        #             label_show(label=self.labelMAYA, image=maya)
        #         cv2.waitKey(int(1000 / self.videoFPS))

    def predict(self):
        self.detect = True
        self.predict = True
        self.camera = True
        # 引数解析 #################################################################
        args = get_args()
       
        model_complexity = args.model_complexity
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        # モデルロード #############################################################
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        camera = cv2.VideoCapture(self.camera_selected)
        self.cap = camera
        self.print_log(log_words=f' [Start Prediction] ')
        status = False

        # World座標プロット ########################################################
        import matplotlib.pyplot as plt        
        fig = plt.figure(figsize=(7.32,14.1))
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
                

        while self.mode_key == 0 :
            ret, image = camera.read()
            if not ret:
                break
            self.camera = True
            # カメラキャプチャ #####################################################
            
            image = cv2.flip(image, 1)  # ミラー表示
            # image = image[0:720, 720:1280] # not functioning with cvtColor
            debug_image = copy.deepcopy(image)
            image_width, image_height = 640, 480
            

            # 検出実施 #############################################################
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            cv2.rectangle(image, (0, 0), (200, 691), (204, 255, 102), -1) # cover for user
            self.results = holistic.process(image)
            image.flags.writeable = True

            # Face Mesh ###########################################################
            face_landmarks = self.results.face_landmarks
            if face_landmarks is not None:
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, face_landmarks)

                # キーポイント
                for index, landmark in enumerate(face_landmarks.landmark):
                    if landmark.visibility < 0 or landmark.presence < 0:
                        continue

                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    if index == 234:  
                        coords_re = landmark_y

                    if index == 454:  
                        coords_le = landmark_y

                # Check the Height of Left Hand
                if (coords_re > coords_le + 10) or (coords_re < coords_le - 10) :
                    draw_c = (0,0,255)
                
                    cv2.putText(debug_image, 'Try To Keep The Head Straight !'
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2, cv2.LINE_AA)

                else :
                    draw_c = (255,204,102)

                debug_image = draw_face_landmarks(draw_c, debug_image, face_landmarks)
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                

            # Pose ###############################################################
            pose_landmarks = self.results.pose_landmarks
            if pose_landmarks is not None:
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, pose_landmarks)
                # 描画
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # Grab Left shoulder coords
                coords_s = tuple(np.multiply(
                            np.array(
                                (pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].x, 
                                    pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].y))
                        , [image_height,image_width]).astype(int))

                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].x, 
                                        pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_EAR].y))
                            , [image_height,image_width]).astype(int))
            
            # Hands ###############################################################
            left_hand_landmarks = self.results.left_hand_landmarks
            right_hand_landmarks = self.results.right_hand_landmarks

            coords_mf_y = 0

            # 左手
            if left_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                # 描画
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 右手
            if right_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # # キーポイント
                for index, landmark in enumerate(right_hand_landmarks.landmark):
                    if landmark.visibility < 0 or landmark.presence < 0:
                        continue

                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    if index == 12:  
                        coords_mf_y = landmark_y

                #  Check the Height of Left Hand
                if coords_mf_y > coords_s[1]+ 100 :
                    draw_c = (255,0,0)
                
                    cv2.rectangle(debug_image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len('Too Low')*20, coords[1]-30), 
                                    (255,0,0), -1)
                    cv2.putText(debug_image, 'Too Low', coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                else :
                    if coords_mf_y < coords_s[1] :
                        draw_c = (0,0,255)
                        
                        cv2.rectangle(debug_image, 
                                        (coords[0], coords[1]+5), 
                                        (coords[0]+len('Too High')*20, coords[1]-30), 
                                        (0,0,255), -1)
                        cv2.putText(debug_image, 'Too High', coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    else :
                        draw_c = (255,204,102)

                        cv2.rectangle(debug_image, 
                                        (coords[0], coords[1]+5), 
                                        (coords[0]+len('Looking Good')*20, coords[1]-30), 
                                        (255,204,102), -1)
                        cv2.putText(debug_image, 'Looking Good', coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                draw_c = (255,204,102) # Default
                debug_image = draw_hands_landmarks(
                    draw_c,
                    debug_image,
                    cx,
                    cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            
            if (face_landmarks is None) or (pose_landmarks is None) or (right_hand_landmarks is None):
                img_r = cv2.imread('image/result_default.png')
                label_show(label=self.labelResult, image=img_r)
                seconds = time.time()
                de_sec = seconds*10
                count = (de_sec//1)%10
                if count == 0:
                    self.print_log(log_words=f' [Searching for Landmarks] ')
                status = False

                       
            else : 
                if status == False:
                    self.print_log(log_words=f' [Detecting!] ')
                    status = True

                if (right_hand_landmarks is not None) and (face_landmarks is not None) and (pose_landmarks is not None):
                    detection(self, debug_image, right_hand_landmarks, face_landmarks, pose_landmarks)
                    # image has been flipped <swap R and L>

            if self.results.pose_world_landmarks is not None:
                plot_world_landmarks(
                    self,
                    plt,
                    ax,
                    self.results.pose_world_landmarks,)
            else :  
                img_3d = cv2.imread('image/3d_default.png')
                label_show(label=self.labelMAYA, image=img_3d) 
                seconds = time.time()
                de_sec = seconds*10
                count = (de_sec//1)%10
                if count == 0:             
                    self.print_log(log_words=f' [Searching for 3D_Landmarks] ')

            label_show(label=self.labelImageOrVideo, image=debug_image)
            # label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(0)
            cv2.waitKey(int(1000 / self.videoFPS))
                
        # self.predict = True
        # if self.image:
        #     hand, rectangle, success = self.detector.rectangle(self.image_file, hand_size=self.hand_size)
        #     if success:
        #         word = self.model.result(hand)['category']
        #         score = self.model.result(hand)['score']
        #         result_image = QPixmap(f'src/{word}.png')
        #         label_show(label=self.labelImageOrVideo, image=rectangle)
        #         self.maya(word=word)
        #         self.labelResult.setPixmap(result_image)
        #         self.print_log(log_words='result:' + word + ' score: ' + str(score))
        #         self.maya(word=word, speed=100)
        #     else:
        #         label_show(label=self.labelImageOrVideo, image=rectangle)
        #         self.print_log(log_words='cant find hand')

        # if self.camera:
        #     flag = 0
        #     dic_count = {'bu': 0, 'hao': 0, 'yi': 0, 'si': 0}
        #     while True:
        #         ret, cv2_image = self.cap.read()
        #         if not ret:
        #             break
        #         hand, rectangle, success = self.detector.rectangle(cv2_image, hand_size=self.hand_size)
        #         label_show(label=self.labelImageOrVideo, image=rectangle)

        #         if success:

        #             hand = cv2.resize(hand, (128, 128))
        #             word = self.model.result(hand)['category']
        #             score = self.model.result(hand)['score']
        #             self.print_log(log_words='result:' + word + ' score: ' + str(score))
        #             image = QPixmap(f'src/{word}.png')
        #             self.labelResult.setPixmap(image)
        #             dic_count[word] += 1
        #             if sum(dic_count.values()) < 8:
        #                 word_show = max(dic_count, key=dic_count.get)
        #                 if flag == 0:
        #                     self.maya(word=word_show)
        #                     flag = 1
        #             else:
        #                 dic_count = {'bu': 0, 'hao': 0, 'yi': 0, 'si': 0}
        #                 flag = 0
        #         else:
        #             label_show(label=self.labelMAYA, image=cv2.imread('src/MAYA_default.png'))
        #             label_show(label=self.labelResult, image=cv2.imread('src/result_default.png'))
        #         cv2.waitKey(int(1000 / self.videoFPS))

    def open_camera(self):
        
        self.camera = True
        camera = cv2.VideoCapture(self.camera_selected)
        self.cap = camera
        self.print_log(log_words=f' [Opening Camera] ')
        
        while self.mode_key == 0 :
            ret, image = camera.read()
            if not ret:
                break
            self.camera = True
            # カメラキャプチャ #####################################################
            
            image = cv2.flip(image, 1)  # ミラー表示
            # image = image[0:720, 720:1280] # not functioning with cvtColor
            debug_image = copy.deepcopy(image)
            

            # 検出実施 #############################################################
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # image.flags.writeable = False
            # self.results = holistic.process(image)
            # image.flags.writeable = True

            label_show(label=self.labelImageOrVideo, image=debug_image)
            # label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(0)
            cv2.waitKey(int(1000 / self.videoFPS))

    def change_VideoFPS(self):
        number, ok = QInputDialog.getInt(self, "input video fps", "(10-60)")
        self.videoFPS = number
        self.print_log(log_words=f'set video fps: {str(number)}')

    def close_camera(self):
        if self.camera == True :        
            self.camera = False
            self.cap.release()
            self.print_log(log_words='camera has been closed')
            self.maya(action=False)
            label_show(label=self.labelResult, image=cv2.imread('src/result_default.png'))
            label_show(label=self.labelMAYA, image=cv2.imread('src/MAYA_default.png'))
            label_show(label=self.labelImageOrVideo, image=cv2.imread('src/default.png'))

    def select_camera(self):
        if self.camera_selected == 0:
            self.camera_selected = 1
            self.actionSelectedCamera.setText('SelectedCamera 1')
            self.print_log(log_words='camera 1 has been selected')
        else:
            self.camera_selected = 0
            self.actionSelectedCamera.setText('SelectedCamera 0')
            self.print_log(log_words='camera 0 has been selected')

    def changeHandSize(self):
        number, ok = QInputDialog.getDouble(self, "input hand_size", "(0-1)")
        self.hand_size = number
        self.print_log(log_words=f'set hand size: {str(number)}')

    def changeModelFile(self):
        path = QFileDialog.getExistingDirectory(self, "choose model filepath", '/')
        # self.model = predict(model=path)
        self.print_log(log_words=f'set model: {path}')

    def changeMinDetectionConfidence(self):
        number, ok = QInputDialog.getDouble(self, "MinDetectionConfidence", "(0-1)")
        self.detector.min_detection_confidence = number
        self.print_log(log_words=f'MinDetectionConfidence: {str(number)}')

    def changeMinTrackingConfidence(self):
        number, ok = QInputDialog.getDouble(self, "MinTrackingConfidence", "(0-1)")
        self.detector.min_tracking_confidence = number
        self.print_log(log_words=f'MinTrackingConfidence: {str(number)}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
