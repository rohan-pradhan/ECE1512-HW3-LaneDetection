import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from statistics import mean
import math


posList = []


def test_images(directory, labels_file_path):
    labels_df = pd.read_csv(labels_file_path)
    labels_df.set_index('name',inplace=True)
    acc_list = []

    for filename in os.listdir(directory):
        global posList
        posList=[]

        #bottom left corner
        x_bL = labels_df.loc[filename, 'x_bL']
        y_bL = labels_df.loc[filename, 'y_bL']

        #top left corner
        x_tL = labels_df.loc[filename, 'x_tL']
        y_tL = labels_df.loc[filename, 'y_tL']

        #top right corner
        x_tR = labels_df.loc[filename, 'x_tR']
        y_tR = labels_df.loc[filename, 'y_tR']

        #bottom right corner
        x_bR = labels_df.loc[filename, 'x_bR']
        y_bR = labels_df.loc[filename, 'y_bR']

        points_gt = np.array([[(x_bL, y_bL), (x_tL, y_tL), (x_tR, y_tR), (x_bR, y_bR)]])

        filename = directory + '/' + filename


        cap = cv.imread(filename, -1)
        cap_copy = np.copy(cap)




        ### UNCOMMENT WHEN USING PERSONALIZED / MANUAL REGION OF INTEREST

        # cv.putText(cap_copy,"Click on three points to define region of interest triangle", (10,100), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
        # cv.imshow('first',cap_copy)
        #
        #
        # cv.setMouseCallback('first', onMouse)
        # # posNp = np.array(posList)
        #
        # if cv.waitKey(0) & 0xFF== ord('q'):
        #     cv.waitKey(1)
        #     cv.destroyAllWindows()
        #     for i in range (1,5):
        #         cv2.waitKey(1)








        canny = canny_function(cap)

        segment = region_of_interest_function(canny, True)

        hough = cv.HoughLinesP(segment, 1, np.pi / 180, 70, np.array([]), minLineLength = 50, maxLineGap = 100)
        # cv.imshow("output2", segment)

        lines = display_lines(cap, hough)
        if (lines is None):
            continue

        # lines_visualize, cross_track_lines = overlay_lines(cap, lines)
        bounding_box = generate_bounding_box(lines)
        output = cv.addWeighted(cap, 0.9, lines_visualize, 1, 1)
        output = cv.addWeighted(output, 0.9, cross_track_lines, 1, 1)
        output = cv.addWeighted(output, 0.5, bounding_box, 1, 1)
        points = generate_bounding_box(lines)


        acc = calculate_IOU(points,points_gt)
        print (acc)
        acc_list.append(acc)

        cv.polylines(output, points, True, (255,0,0))
        cv.imshow("output", output)

        if cv.waitKey(0) & 0xFF== ord('q'):
            cap.release()
            cv.destroyAllWindows()

        posList.clear()
    print ("Average accuracy ", mean(acc_list))



def calculate_IOU(predicted_bounding_box, ground_truth_bounding_box):

    #Top left corner of predicted bounding box
    predicted_x1 = predicted_bounding_box[0][1][0]
    predicted_y1 = predicted_bounding_box[0][1][1]

    #Bottom Right corner of predicted bounding box
    predicted_x2 = predicted_bounding_box[0][3][0]
    predicted_y2 = predicted_bounding_box[0][3][1]

    #Top left corner of ground truth bounding box
    gt_x1 = ground_truth_bounding_box[0][1][0]
    gt_y1 = ground_truth_bounding_box[0][1][1]

    #Bottom Right corner of ground truth bounding box
    gt_x2 = ground_truth_bounding_box[0][3][0]
    gt_y2 = ground_truth_bounding_box[0][3][1]


    x_left = max(predicted_x1, gt_x1)

    y_top = max(predicted_y1, gt_y1)

    x_right = min(predicted_x2, gt_x2)

    y_bottom = min(predicted_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0


    intersection_area = (x_right - x_left) * (y_bottom - y_top)


    bb1_area = (predicted_x2-predicted_x1)*(predicted_y2-predicted_y1)
    bb2_area = (gt_x2-gt_x1)*(gt_y2-gt_y1)

    bb1_area = abs(bb1_area)
    bb2_area = abs(bb2_area)


    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    if (iou < 0):
        iou=0
    return iou


def canny_function(frame, low_threshold=50, high_threshold=150):
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)

    blur = cv.GaussianBlur(gray,(5,5), 0)

    canny = cv.Canny(blur, low_threshold , high_threshold)

    return canny

def generate_bounding_box(lines):
    x_a_1 = lines[0][0]
    y_a_1 = lines[0][1]
    x_a_2 = lines[0][2]
    y_a_2 = lines[0][3]

    x_b_1 = lines[1][0]
    y_b_1 = lines[1][1]
    x_b_2 = lines[1][2]
    y_b_2 = lines[1][3]


    polygons = np.array([[(x_a_1, y_a_1), (x_a_2, y_a_2),(x_b_2, y_b_2), (x_b_1, y_b_1)]])

    return polygons


def onMouse(event, x, y, flags, param):
   global posList
   if event == cv.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        print (posList)


def sobel_function(frame):
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray,(5,5), 0)
    sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)
    sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)


    # canny = cv.Canny(blur, 80,150)

    return sobelx, sobely

def region_of_interest_function(frame, posList):

    height = frame.shape[0]

    polygons = np.array([[(200, 650),(1080,650), (640,360)]])

    mask = np.zeros_like(frame)

    cv.fillPoly(mask, polygons, 255)

    roi = cv.bitwise_and(frame, mask)

    return roi

def display_lines(frame, lines):
    left_lines = []
    right_lines = []

    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            parameters = np.polyfit((x1,x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]

            if slope<0:
                left.append((slope, y_intercept))
            else:
                right.append((slope,y_intercept))

        left_avg = np.average(left, axis=0)
        right_avg = np.average(right, axis=0)

        print (left_avg)
        print (right_avg)
        left_line = get_cords(frame, left_avg)
        right_line = get_cords(frame, right_avg)

        return (np.array([left_line, right_line]))
    except:
        print ("No Lines have been detected")
        return None

def get_cords(frame, parameters):
    slope, intercept = parameters
    print("slope ", slope )

    y1 = frame.shape[0]

    y2 = int(y1-200)

    x1 = int((y1-intercept)/slope)

    x2 = int((y2-intercept)/slope)

    return np.array([x1, y1, x2, y2])

def overlay_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    cross_track_lines = np.zeros_like(frame)

    print (lines)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 10)

        cv.line(cross_track_lines, (lines[0][0], lines[1][1]), (lines[1][0], lines[0][1]), (0,0,255), 5)

    else:
        print ("No lines to visualize")

    # TODO
    return lines_visualize, cross_track_lines

test_images('../EvaluationData/Night', '../EvaluationData/NightLabels.csv')

'''
cap = cv.imread("10.jpg", -1)
cap_copy = np.copy(cap)

cv.putText(cap_copy,"Click on three points to define region of interest triangle", (10,100), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
cv.imshow('first',cap_copy)


cv.setMouseCallback('first', onMouse)
posNp = np.array(posList)

if cv.waitKey(0) & 0xFF== ord('q'):
    cv.waitKey(1)
    cv.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)



canny = canny_function(cap)
kernel = np.ones((5,5),np.uint8)


# cv.imshow("canny", canny)


segment = region_of_interest_function(canny, posList)

hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)


# slopes = []
#
# for line in hough:
#     for x1,y1,x2,y2 in line:
#
#         slope = (y2 - y1) / (x2 - x1)
#         slope = abs(slope)
#         if (abs(slope)>0.5):
#             cv.line(cap,(x1,y1),(x2,y2),(0,255,0),2)
#         # print (line, " slope: ", slope)
#         slopes.append((slope, line))
#         # cv.line(cap,(x1,y1),(x2,y2),(0,255,0),2)
#
# slopes = sorted(slopes, key=lambda x: x[0], reverse=True)
# line_1 = slopes[0]
# line_2 = slopes[1]
#
#
# for x1,y1,x2,y2 in line_1[1]:
#     cv.line(cap,(x1,y1),(x2,y2),(0,255,0),2)
#
# for x1,y1,x2,y2 in line_2[1]:
#     cv.line(cap,(x1,y1),(x2,y2),(0,255,0),2)
#
#
# print(line_1)
# print(line_2)


# cv.imshow("output2", cap)
# cv.imshow("output3", segment)
# # slopes = sorted(map(abs, slopes))
# print("slopes sorted: ", slopes)
lines = display_lines(cap, hough)
lines_visualize, cross_track_lines = overlay_lines(cap, lines)
# bounding_box = generate_bounding_box(lines)
output = cv.addWeighted(cap, 0.9, lines_visualize, 1, 1)
output = cv.addWeighted(output, 0.9, cross_track_lines, 1, 1)
# # output = cv.addWeighted(output, 0.5, bounding_box, 1, 1)

# points = generate_bounding_box(lines)

# print(calculate_IOU(points,points))
#
# cv.polylines(output, points, True, (255,0,0))
cv.imshow("output", output)
# cv.waitKey(0)
if cv.waitKey(0) & 0xFF== ord('q'):
    cap.release()
    cv.destroyAllWindows()

print (posList)

'''
