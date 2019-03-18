import os
import numpy as np
import cv2
import torch
import math
#import selectivesearch
from models import load_model
from utils import Window, draw_face
from PIL import ImageFont, ImageDraw, Image
import kdtree
import random
# global settings
EPS = 1e-5
minFace_ = 20 * 1.4
scale_ = 1.414
stride_ = 8
classThreshold_ = [0.37, 0.43, 0.97]
nmsThreshold_ = [0.8, 0.8, 0.3]
angleRange_ = 45
stable_ = 0


class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


def preprocess_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim))
    return img - np.array([104, 117, 123])

def resize_img(img, scale):
    h, w = img.shape[:2]
    h_, w_ = int(h / scale), int(w / scale)
    img = img.astype(np.float32) # fix opencv type error
    ret = cv2.resize(img, (w_, h_))
    return ret

def pad_img(img):
    row = min(int(img.shape[0] * 0.2), 100)
    col = min(int(img.shape[1] * 0.2), 100)
    ret = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)
    return ret

def legal(x, y, img):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return True
    else:
        return False

def inside(x, y, rect):
    if rect.x <= x < (rect.x + rect.w) and rect.y <= y < (rect.y + rect.h):
        return True
    else:
        return False

def smooth_angle(a, b):
    if a > b:
        a, b = b, a
    diff = (b - a) % 360
    if diff < 180:
        return a + diff // 2
    else:
        return b + (360 - diff) // 2

# Prelist global variable
prelist = []
def smooth_window(winlist):
    global prelist
    for win in winlist:
        for pwin in prelist:
            if IoU(win, pwin) > 0.9:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = pwin.x
                win.y = pwin.y
                win.w = pwin.w
                win.h = pwin.h
                win.angle = pwin.angle
            elif IoU(win, pwin) > 0.6:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = (win.x + pwin.x) // 2
                win.y = (win.y + pwin.y) // 2
                win.w = (win.w + pwin.w) // 2
                win.h = (win.h + pwin.h) // 2
                win.angle = smooth_angle(win.angle, pwin.angle)
    prelist = winlist
    return winlist

def IoU(w1, w2):
    xOverlap = max(0, min(w1.x + w1.w - 1, w2.x + w2.w - 1) - max(w1.x, w2.x) + 1)
    yOverlap = max(0, min(w1.y + w1.h - 1, w2.y + w2.h - 1) - max(w1.y, w2.y) + 1)
    intersection = xOverlap * yOverlap
    unio = w1.w * w1.h + w2.w * w2.h - intersection
    return intersection / unio

# Non-maximum suppression
def NMS(winlist, local, threshold):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                continue
            if IoU(winlist[i], winlist[j]) > threshold:
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret

def deleteFP(winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            win = winlist[j]
            if inside(win.x, win.y, winlist[i]) and inside(win.x + win.w - 1, win.y + win.h - 1, winlist[i]):
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret


# Mimic method
def set_input(img):
    if type(img) == list:
        img = np.stack(img, axis=0)
    else:
        img = img[np.newaxis, :, :, :]
    img = img.transpose((0, 3, 1, 2))
    return torch.FloatTensor(img)

# Trans window
def trans_window(img, imgPad, winlist):
    """transfer Window2 to Window1 in winlist"""
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    ret = list()
    for win in winlist:
        if win.w > 0 and win.h > 0:
            ret.append(Window(win.x-col, win.y-row, win.w, win.angle, win.conf))
    return ret

# Stage 1
def stage1(img, imgPad, net, thres):
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    winlist = []
    netSize = 24
    curScale = minFace_ / netSize
    img_resized = resize_img(img, curScale)
    while min(img_resized.shape[:2]) >= netSize:
        img_resized = preprocess_img(img_resized)
        # net forward
        net_input = set_input(img_resized)
        with torch.no_grad():
            net.eval()
            cls_prob, rotate, bbox = net(net_input)

        w = netSize * curScale
        for i in range(cls_prob.shape[2]): # cls_prob[2]->height
            for j in range(cls_prob.shape[3]): # cls_prob[3]->width
                if cls_prob[0, 1, i, j].item() > thres:
                    sn = bbox[0, 0, i, j].item()
                    xn = bbox[0, 1, i, j].item()
                    yn = bbox[0, 2, i, j].item()
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, imgPad) and legal(rx + rw - 1, ry + rw -1, imgPad):
                        if rotate[0, 1, i, j].item() > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j].item()))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j].item()))
        img_resized = resize_img(img_resized, scale_)
        curScale = img.shape[0] / img_resized.shape[0]
    return winlist

# Stage 2
def stage2(img, img180, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height = img.shape[0]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(img[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        else:
            y2 = win.y + win.h -1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w, :], dim))
    # net forward
    net_input = set_input(datalist)
    with torch.no_grad():
        net.eval()
        cls_prob, rotate, bbox = net(net_input)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            if abs(winlist[i].angle) > EPS:
                cropY = height - 1 - (cropY + cropW - 1)
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            maxRotateScore = 0
            maxRotateIndex = 0
            for j in range(3):
                if rotate[i, j].item() > maxRotateScore:
                    maxRotateScore = rotate[i, j].item()
                    maxRotateIndex = j
            if legal(x, y, img) and legal(x+w-1, y+w-1, img):
                angle = 0
                if abs(winlist[i].angle) < EPS:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 0
                    else:
                        angle = -90
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append(Window2(x, height-1-(y+w-1), w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret

# Stage 3
def stage3(imgPad, img180, img90, imgNeg90, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist

    datalist = []
    height, width = imgPad.shape[:2]

    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(imgPad[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        elif abs(win.angle - 90) < EPS:
            datalist.append(preprocess_img(img90[win.x:win.x+win.w, win.y:win.y+win.h, :], dim))
        elif abs(win.angle + 90) < EPS:
            x = win.y
            y = width - 1 - (win.x + win.w -1)
            datalist.append(preprocess_img(imgNeg90[y:y+win.h, x:x+win.w, :], dim))
        else:
            y2 = win.y + win.h - 1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w], dim))
    # network forward
    net_input = set_input(datalist)
    with torch.no_grad():
        net.eval()
        cls_prob, rotate, bbox = net(net_input)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            img_tmp = imgPad
            if abs(winlist[i].angle - 180) < EPS:
                cropY = height - 1 - (cropY + cropW -1)
                img_tmp = img180
            elif abs(winlist[i].angle - 90) < EPS:
                cropX, cropY = cropY, cropX
                img_tmp = img90
            elif abs(winlist[i].angle + 90) < EPS:
                cropX = winlist[i].y
                cropY = width -1 - (winlist[i].x + winlist[i].w - 1)
                img_tmp = imgNeg90

            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            angle = angleRange_ * rotate[i, 0].item()
            if legal(x, y, img_tmp) and legal(x+w-1, y+w-1, img_tmp):
                if abs(winlist[i].angle) < EPS:
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 180) < EPS:
                    ret.append(Window2(x, height-1-(y+w-1), w, w, 180-angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 90) < EPS:
                    ret.append(Window2(y, x, w, w, 90-angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    ret.append(Window2(width-y-w, x, w, w, -90+angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret

# Detect face
def detect(img, imgPad, nets):
    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)

    winlist = stage1(img, imgPad, nets[0], classThreshold_[0])
    winlist = NMS(winlist, True, nmsThreshold_[0])
    winlist = stage2(imgPad, img180, nets[1], classThreshold_[1], 24, winlist)
    winlist = NMS(winlist, True, nmsThreshold_[1])
    winlist = stage3(imgPad, img180, img90, imgNeg90, nets[2], classThreshold_[2], 48, winlist)
    winlist = NMS(winlist, False, nmsThreshold_[2])
    winlist = deleteFP(winlist)
    return winlist

# Pcn Detect
def pcn_detect(img, nets):
    imgPad = pad_img(img)
    winlist = detect(img, imgPad, nets)
    if stable_:
        winlist = smooth_window(winlist)
    return trans_window(img, imgPad, winlist)

# Draw region
def draw_rpn(img, regions_index):
    # Tuple (mp, region)
    for index in range(len(regions_index)):
        midpoint_x, mid_point_y = regions_index[index][0]
        # Head is regions_index[index][1]
        rect_x, rect_y, rect_w, rect_h = regions_index[index][2] # Get coordinate
        cv2.circle(img, (midpoint_x, mid_point_y), 2, (0,255,255), -1)
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0,255,0), 2)

# Check area resize image follow by ratio
def check_area(img):
    height, width, depth = img.shape
    # print("Width {0} - Height {1} - Depth {2}".format(width, height, depth))
    if width > 1000 :
        ratio = height / width
        width = 500
        height = ratio * width
        img = cv2.resize(img, (int(width) , int(height)))
    elif height > 1000:
        ratio = height / width
        height = 500
        width = height / ratio
        img = cv2.resize(img, (int(width) , int(height)))
    return img

# Check colide swept aabb
def check_collide(head, region):
    if region == False:
        return False
    else:

        head_x = head[0][0]
        head_y = head[0][1]
        head_w = head[2][0] - head[0][0]

        left = region[0] - (head_x + head_w)
        top = (region[1] + region[3]) - head_y
        right = (region[0] + region[2]) - head_x
        bottom = region[1] - (head_y + head_w)

        if (left > 0 or right < 0 or top < 0 or bottom > 0): 
            return region
        else:
            return False  # Not get False

# Check colide swept aabb
def swept_aabb(heads, regions):
    regions_index = regions
    for head in heads:
        regions_index = list(map(lambda region: check_collide(head, region), regions_index))
    return regions_index

# Mid point of head
def head_mid_point(head):
    # Head (TopLeft, BottomLeft, BottomRight, TopRight)
    mid_point_x = ((head[3][0] - head[0][0]) // 2) + head[0][0]
    mid_point_y = ((head[1][1] - head[0][1]) // 4) + head[0][1]  
    return (mid_point_x, mid_point_y)

# Mid point of region
def region_mid_point(region):
    # # region (x,y,w,h)
    mid_point_x = (region[0] + region[2] // 2)
    mid_point_y = (region[1] + region[3] // 2)
    return (mid_point_x, mid_point_y)

# Find midpoint tuple in list of midpoint box
def getIndexTuple(tupCheck, listTuple):
    index = [x for x, y in enumerate(listTuple) if (y[0] == tupCheck[0] and \
                                                    y[1] == tupCheck[1])]
    return index[0]

# KdTree to find closest box of the object
def head_region(heads, regions_index, img):
    heads_mp = list(map(lambda head: head_mid_point(head), heads))
    regions_mp = list(map(lambda region: region_mid_point(region), regions_index))
    kdtree_region = kdtree.create(regions_mp)
    region_nearest = []
    index_head = 0
    
    for head_mp in heads_mp:
        nearestMPPoint = kdtree_region.search_nn(head_mp)
        nearestMP = nearestMPPoint[0].data # Get nearest point 
        indexNearestMP = getIndexTuple(nearestMP, regions_mp)
        region_nearest.append((nearestMP, heads[index_head], regions_index[indexNearestMP]))
        index_head += 1

    return region_nearest

def getFontScale(img):
    height, width, depth = img.shape
    fontScale = (height * width) / (500 * 500)
    return fontScale

def randomBoundingBox(img, total):
    random.seed(30)
    padding_left = padding_right = 20
    padding_top = padding_bottom = 20
    height_img, width_img, depth_img = img.shape
    # width = int(width_img * 0.5)
    width = int(width_img * 0.3)
    height = int(height_img * 0.23)

    # Get width and height of text insert
    random_point_x = [random.randint(0 + padding_left, width_img - width - padding_right) for _ in range(total)]
    random_point_y = [random.randint(0 + padding_top, height_img - height - padding_bottom) for _ in range(total)]
    random_point = zip(random_point_x, random_point_y)
    random_bb = list(map(lambda point: (point[0], point[1], width, height \
                            ), random_point))
    return random_bb

def checkSafeZone(img, mp_region, head, side):
    head_point_1, head_point_2, head_point_3, head_point_4 = head
    offset_height_head = (head_point_3[1] - head_point_1[1]) / 12
    offset_width_head = (head_point_3[1] - head_point_1[1]) / 12
    if side == "HORIZONTAL":
        sz_hor_b1 = int(head_point_1[1] + 4 * offset_height_head)
        sz_hor_b2 = int(head_point_1[1] + 8 * offset_height_head)
        if mp_region[1] < sz_hor_b1:
            return 0
        elif mp_region[1] > sz_hor_b2:
            return 2
        else:
            return 1
    elif side == "VERTICAL":
        sz_ver_b1 = int(head_point_1[0] + 4 * offset_width_head)
        sz_ver_b2 = int(head_point_1[0] + 8 * offset_width_head)
        if mp_region[0] < sz_ver_b1:
            return 0
        elif mp_region[0] > sz_ver_b2:
            return 2
        else:
            return 1

def findBalloonTail(img, region, head):
    x, y, width_r, height_r = region

    region_point_1 = (x,y)
    region_point_2 = (x + width_r, y)
    region_point_3 = (x + width_r, y + height_r)
    region_point_4 = (x, y + height_r)

    head_point_1, head_point_2, head_point_3, head_point_4 = head
    tail_firstb_x, tail_firstb_y, tail_lastb_x, tail_lastb_y = (0,0,0,0)
    # balloon_tail = cv2.imread("./sample/tail/left_0.png")
    balloon_tail = cv2.imread("./sample/tail2/top_0.png")
    mp_region = region_mid_point(region)
    mp_head = head_mid_point(head)
    # cv2.circle(img, mp_region, 2, (0,0,255), -1)
    # cv2.circle(img, mp_head, 2, (0,0,255), -1)

    # check right - x  region < left head => LEFT:
    # check bottom - y region < top head => TOP:
    # check top - y region > bottom head => BOTTOM:
    # check left - x region > right head => RIGHT
    if region_point_2[0]  < head_point_1[0]: # LEFT

        sz_index = checkSafeZone(img, mp_region, head, "HORIZONTAL")
        # balloon_tail = cv2.imread("./sample/tail/left_{0}.png".format(str(sz_index)))
        balloon_tail = cv2.imread("./sample/tail2/left_{0}.png".format(str(sz_index)))
        height_t, width_t, depth_t = balloon_tail.shape
        aspect_ratio_t = height_t / width_t

        width_t = abs(head_point_1[0] - region_point_3[0])
        height_t = int(width_t * aspect_ratio_t)



        if sz_index == 1:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2))
        elif sz_index == 0:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2) + height_t * 0.3)
        elif sz_index == 2:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2) - height_t * 0.3)

        tail_firstb_x = int(region_point_2[0])
        tail_lastb_x = int(region_point_2[0] + width_t)
        tail_lastb_y = int(tail_firstb_y + height_t)
        print("LEFT")
        
    
    elif region_point_1[0] > head_point_3[0]: # RIGHT

        sz_index = checkSafeZone(img, mp_region, head, "HORIZONTAL")
        # balloon_tail = cv2.imread("./sample/tail/right_{0}.png".format(str(sz_index)))
        balloon_tail = cv2.imread("./sample/tail2/right_{0}.png".format(str(sz_index)))
        height_t, width_t, depth_t = balloon_tail.shape
        aspect_ratio_t = height_t / width_t

        width_t = abs(head_point_3[0] - region_point_1[0])
        height_t = int(width_t * aspect_ratio_t)

        if sz_index == 1:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2))
        elif sz_index == 0:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2) + height_t * 0.3)
        elif sz_index == 2:
            tail_firstb_y = int(mp_region[1] - int(height_t / 2) - height_t * 0.3)



        tail_firstb_x = int(head_point_3[0])
        tail_lastb_x = int(head_point_3[0] + width_t)
        tail_lastb_y = int(tail_firstb_y + height_t)

    elif region_point_3[1] < head_point_1[1]: # TOP
        
        sz_index = checkSafeZone(img, mp_region, head, "VERTICAL")
        # balloon_tail = cv2.imread("./sample/tail/top_{0}.png".format(str(sz_index)))
        balloon_tail = cv2.imread("./sample/tail2/top_{0}.png".format(str(sz_index)))
        height_t, width_t, depth_t = balloon_tail.shape
        aspect_ratio_t = height_t / width_t

        height_t = abs(region_point_3[1] - head_point_1[1])
        width_t = int(height_t / aspect_ratio_t)

        if sz_index == 1:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2))
        elif sz_index == 0:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2) + width_t * 0.3)
        elif sz_index == 2:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2) - width_t * 0.3)

        tail_firstb_y = int(region_point_3[1])
        tail_lastb_x = int(tail_firstb_x + width_t)
        tail_lastb_y = int(region_point_3[1] + height_t)
        print("TOP")


    elif region_point_2[1] > head_point_4[1]: # BOTTOM
        
        sz_index = checkSafeZone(img, mp_region, head, "VERTICAL")
        # balloon_tail = cv2.imread("./sample/tail/bottom_{0}.png".format(str(sz_index)))
        balloon_tail = cv2.imread("./sample/tail2/bottom_{0}.png".format(str(sz_index)))
        height_t, width_t, depth_t = balloon_tail.shape
        aspect_ratio_t = height_t / width_t

        height_t = abs(region_point_1[1] - head_point_3[1])
        width_t = int(height_t / aspect_ratio_t)


        if sz_index == 1:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2))
        elif sz_index == 0:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2) + width_t * 0.3)
        elif sz_index == 2:
            tail_firstb_x = int(mp_region[0] - int(width_t / 2) - width_t * 0.3)

        tail_firstb_y = int(head_point_3[1])
        tail_lastb_x = int(tail_firstb_x + width_t)
        tail_lastb_y = int(head_point_3[1] + height_t)
        print("BOTTOM")

    balloon_tail = cv2.resize(balloon_tail, (int(width_t) , int(height_t)))
    balloon2gray = cv2.cvtColor(balloon_tail, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(balloon2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi = img[tail_firstb_y:tail_lastb_y, tail_firstb_x:tail_lastb_x]

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    balloon_fg = cv2.bitwise_and(balloon_tail, balloon_tail, mask=mask)

    dst = cv2.add(img_bg, balloon_fg)
    img[tail_firstb_y:tail_lastb_y, tail_firstb_x:tail_lastb_x] = dst

    return img

def assignBalloon(img, regions_index):
    mp_point = regions_index[0]
    head = regions_index[1]
    region = regions_index[2]
    head_mp_point = head_mid_point(head)

    x_mp_point, y_mp_point = mp_point
    x_head_mp_point, y_head_mp_point = head_mp_point

    x, y, width_r, height_r = region

    balloon_img = cv2.imread("./sample/cloud.png")
    
    balloon_img = cv2.resize(balloon_img, (int(width_r) , int(height_r)))
    balloon2gray = cv2.cvtColor(balloon_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(balloon2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = img[y:y+height_r,x:x+width_r]

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    balloon_fg = cv2.bitwise_and(balloon_img, balloon_img, mask=mask)

    dst = cv2.add(img_bg, balloon_fg)
    img[y:y+height_r,x:x+width_r] = dst

    img = findBalloonTail(img, region, head)

    return img

def calculateHeadArea(head):
    point1, point2, point3, point4 = head
    width_head = int(point2[0] - point1[0])
    height_head = int(point3[1] - point2[1])
    area = width_head * height_head
    return area

def findLargeHead(lst_head):
    head_area = list(map(lambda head: calculateHeadArea(head),lst_head))
    index = head_area.index(max(head_area))
    return index
    # Return index of head

if __name__ == '__main__':
    nets = load_model()

    # Load image and check area for large image
    files = os.listdir("./balloon_ta/")
    img_paths = list(map(lambda name_file: "./balloon_ta/" + name_file, files))
    for imgpath in img_paths:
        img = cv2.imread(imgpath)
        img = check_area(img)

        # Detect faces
        faces = pcn_detect(img, nets)

        # Get head
        lst_head = []
        lst_safezone = []
        for face in faces:
            lst_b, lst_c = draw_face(img, face)
            lst_head.append(lst_b)
            lst_safezone.append(lst_c)

        if len(lst_head) > 0:   
            print(imgpath) 
            # Get text and text_size
            regions = randomBoundingBox(img, 2000)
            # regions_index = swept_aabb(lst_head, regions)
            regions_index = swept_aabb(lst_safezone, regions)
            regions_index = list(filter(lambda region: region != False, regions_index))
            if len(regions_index) > 0:
                # KdTree
                regions_index = head_region(lst_head, regions_index, img)
                # Only 1 head - largest head
                index_head = findLargeHead(lst_head)
                regions_index = list((regions_index[index_head],))
                assignBalloon(img, regions_index[0]) # Only 1 balloon now
                # draw_rpn(img, regions_index)

        name = os.path.basename(imgpath)
        cv2.imwrite('result2/ret_{}'.format(name), img)

