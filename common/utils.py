# coding: utf-8

import cv2
import numpy as np
import copy

from scipy.signal import *
from scipy.sparse import spdiags, eye


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))     
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_

def drawLandmark(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    img_ = img.copy()
    cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_

def drawLandmark_multiple(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def drawLandmark_Attribute(img, bbox, landmark,gender,age):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
        if gender.argmax()==0:
                # -1->female, 1->male; -1->old, 1->young
                cv2.putText(img, 'female', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
                cv2.putText(img, 'male', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),3)
        if age.argmax()==0:
                cv2.putText(img, 'old', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
        else:
                cv2.putText(img, 'young', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
    return img


def drawLandmark_only(img, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    img_=img.copy()
    #cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_


def processImage(imgs):
    '''
    Subtract mean and normalize, imgs [N, 1, W, H]
    '''
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img-m)/s
    return imgs

def flip(face, landmark):
    '''
    flip a face and its landmark
    '''
    face_ = cv2.flip(face, 1) # 1 means flip horizontal
    landmark_flip = np.asarray(np.zeros(landmark.shape))
    for i, point in enumerate(landmark):
        landmark_flip[i] = (1-point[0], point[1])
    # for 5-point flip
        landmark_flip[[0,1]] = landmark_flip[[1,0]]
    landmark_flip[[3,4]] = landmark_flip[[4,3]]
    # for 19-point flip
        #landmark_flip[[0,9]] = landmark_flip[[9,0]]
    #landmark_flip[[1,8]] = landmark_flip[[8,1]]
    #landmark_flip[[2,7]] = landmark_flip[[7,2]]
    #landmark_flip[[3,6]] = landmark_flip[[6,3]]
    #landmark_flip[[4,11]] = landmark_flip[[11,4]]
    #landmark_flip[[5,10]] = landmark_flip[[10,5]]
    #landmark_flip[[12,14]] = landmark_flip[[14,12]]
    #landmark_flip[[15,17]] = landmark_flip[[17,15]]
    return (face_, landmark_flip)

def scale(landmark):
    '''
    scale the landmark from [0,1] to [-1,1]
    '''
    landmark_ = np.asarray(np.zeros(landmark.shape))
    lanmark_=(landmark-0.5)*2
    return landmark_

def check_bbox(img, bbox):
    '''
    Check whether bbox is out of the range of the image
    '''
    img_w, img_h = img.shape
    if bbox.x > 0 and bbox.y > 0 and bbox.right < img_w and bbox.bottom < img_h:
        return True
    else:
        return False

def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)

def drawLandmark_butterfly(img, bbox, landmark):  # 나비모양 랜드마크
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    m_img = copy.copy(img)
    cv2.rectangle(m_img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)

    n = 0
    for x, y in landmark:
        if n in [1, 2, 14, 15, 29, 30, 31, 32, 34, 35]:  # 나비모양 빨간점
            cv2.circle(m_img, (int(x), int(y)), 2, (0, 0, 255), -1)
        elif n in [0, 3, 13, 16, 28, 33]:  # 꼭짓점 연두점
            cv2.circle(m_img, (int(x), int(y)), 2, (0, 255, 0), -1)
        else:  # 그 외 파란점
            cv2.circle(m_img, (int(x), int(y)), 2, (255, 0, 0), -1)
        n += 1

    return m_img

def getPoint(landmark):
    n = 0
    Point = []
    for x, y in landmark:
        if n == 0:
            Point.insert(0, (int(x), int(y)))
        elif n == 3:
            Point.insert(1, (int(x), int(y)))
        elif n == 33:
            Point.insert(2, (int(x), int(y)))
        elif n == 13:
            Point.insert(3, (int(x), int(y)))
        elif n == 16:
            Point.insert(4, (int(x), int(y)))
        elif n == 28:
            Point.insert(5, (int(x), int(y)))
        n += 1

    return Point

def drawPolyline(img, landmark):  # 다각형 그리기
    pnt = getPoint(landmark)
    bpt = np.array(pnt)  # numpy 형식으로 저장
    for i in range(6):  # 다각형 그림
        cv2.polylines(img, [bpt], True, (0, 255, 255))

    return img

def getROI(img, landmark):  # ROI 추출
    pnt = getPoint(landmark)

    bpt = np.array(pnt)  # numpy 형식으로 저장

    mask = np.zeros_like(img)  # frame 과 같은 사이즈의 0행렬
    cv2.fillPoly(mask, [bpt], (150, 150, 150, 64))
    ROI = np.zeros_like(img)
    ROI = cv2.copyTo(img, mask, ROI)  # ROI 추출

    return ROI

def warptoRect(img, landmark):
    w_img = copy.copy(img)
    point = getPoint(landmark)

    w = 130
    h = 100

    # 직사각형 모양으로 워핑하는 과정- 나비모양의 반을 쪼개서 각각 워핑 한 후 붙임
    bp = np.float32([point[0], point[1], point[2], point[5]])  # 반쪽
    pnt = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    trn_mask = cv2.getPerspectiveTransform(bp, pnt)  # 변환 행렬
    ROI_one = cv2.warpPerspective(w_img, trn_mask, (w, h))  # 어파인 변환

    bp = np.float32([point[2], point[3], point[4], point[5]])  # 나머지 반쪽
    pnt = np.float32([[0, h-1], [w-1, h-1], [w-1, 0], [0, 0]])
    trn_mask = cv2.getPerspectiveTransform(bp, pnt)
    ROI_two = cv2.warpPerspective(w_img, trn_mask, (w, h))  # 어파인 변환

    w_ROI = np.concatenate((ROI_one, ROI_two), axis=1)  # 붙임

    return w_ROI


def getXY(ROI, option):
    """
    b_ROI = ROI[:, :, 0]
    g_ROI = ROI[:, :, 1]
    r_ROI = ROI[:, :, 2]
    """
    b_ROI, g_ROI, r_ROI = cv2.split(ROI)

    b_mean = b_ROI.mean()
    g_mean = g_ROI.mean()
    r_mean = r_ROI.mean()

    b_std = b_ROI.std()
    g_std = g_ROI.std()
    r_std = r_ROI.std()

    # Formulae from the paper "Self-adaptive Matri
    # x Completion..."
    R_n = r_ROI * r_std / r_mean
    G_n = g_ROI * g_std / g_mean
    B_n = b_ROI * b_std / b_mean

    if option == "chrom":
        X = 3 * R_n - 2 * G_n
        Y = 1.5 * R_n + G_n - 1.5 * (b_ROI * b_std / b_mean)
    elif option == "pos":
        X = G_n - B_n
        Y = -2*R_n + G_n + B_n

    return (X, Y)

def butter_bandpass(lowcut, highcut, fs, order=5):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   # y = lfilter(b, a, data)
   y = filtfilt(b, a, data, method="pad")
   return y

def getChrom(X_f, Y_f, option):
    alpha = X_f.std() / Y_f.std()
    if option == "chrom":
        chrominance = X_f - alpha * Y_f
    elif option == "pos":
        chrominance = X_f + alpha*Y_f
    return chrominance

def normalization(signal):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    signal_proc = (signal - mean)/std_dev
    return signal_proc

def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, 'valid')
    return moving_avg

