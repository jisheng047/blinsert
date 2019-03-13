"code originally in PCN.h"
import numpy as np
import cv2



class Window:
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score

def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry

def draw_line(img, pointlist, color):
    thick = 2
    cyan = (255, 255, 0)
    blue = (255, 0, 0)
    cv2.line(img, pointlist[0], pointlist[1], color, thick)
    cv2.line(img, pointlist[1], pointlist[2], color, thick)
    cv2.line(img, pointlist[2], pointlist[3], color, thick)
    cv2.line(img, pointlist[3], pointlist[0], blue, thick)

def draw_face(img, face:Window):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x -1
    y2 = face.width + face.y -1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2

    x1_b = face.x - 8
    y1_b = face.y - 8
    x2_b = face.width + x1_b + 15
    y2_b = face.width + y1_b + 15

    # TopLeft - BottomLeft - BottomRight - TopRight
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    lst_b = (x1_b, y1_b), (x1_b, y2_b), (x2_b, y2_b), (x2_b, y1_b)

    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for (x, y) in lst]
    pointlist2 = [rotate_point(x, y, centerX, centerY, face.angle) for (x, y) in lst_b]

    draw_line(img, pointlist, (255,255,0))
    draw_line(img, pointlist2, (0,0,255))

    return lst_b