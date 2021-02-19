# Face alignment demo
# Uses MTCNN or FaceBoxes as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Cunjian Chen (ccunjian@gmail.com), Aug 2020

from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
import scipy
from scipy.signal import savgol_filter
import time

from common.utils import *
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
from MTCNN import detect_faces
from pulse import *
import math

parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='MobileFaceNet', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def load_model():
    if args.backbone=='MobileNet':
        model = MobileNet_GDConv(136)
        model = torch.nn.DataParallel(model)
        # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
        checkpoint = torch.load('checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=map_location)
        print('Use MobileNet as backbone')
    elif args.backbone=='PFLD':
        model = PFLDInference()
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone')
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    if args.backbone=='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)
    else:
        print('Error: not suppored backbone')
    model.load_state_dict(checkpoint['state_dict'])
    return model

cap = cv2.VideoCapture("src/subject30.avi")  # "src/rohin_active.mov"1avvicinamento  3rotazione_dx2.webm
fps = 30  # int(cap.get(cv2.CAP_PROP_FPS))

f_low = 0.7
f_high = 4
fs = 100

option = "pos"  # "chrom" - pos로 할건지 chrom으로 할건지 결정

if __name__ == '__main__':
    heartrate = list()
    mean_chrom_signal = np.empty((1, 1), object)

    print("Gathering information....")
    f = open("results/retina_result.txt", 'w')

    n = 0
    while cap.isOpened():
        ret, frame = cap.read()  # ret 은 프레임 읽기를 성공하면 True 값 반환
        if not ret:
            print("End or Error")
            break
        if args.backbone == 'MobileNet':
            out_size = 224
        else:
            out_size = 112

            model = load_model()
            model = model.eval()

        if args.detector == 'MTCNN':
            # perform face detection using MTCNN
            image = Image.open(frame)
            faces, landmarks = detect_faces(frame)
        elif args.detector == 'FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(frame)
        elif args.detector == 'Retinaface':
            retinaface = Retinaface.Retinaface()
            faces = retinaface(frame)
        else:
            print('Error: not suppored detector')

        if len(faces) == 0:
            print('NO face is detected!')
            continue

        height, width, _ = frame.shape
        for k, face in enumerate(faces):  # project11 참고
            x1, y1, x2, y2 = face[0], face[1], face[2], face[3]

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(min([w, h]) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped = frame[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face / 255.0
            if args.backbone == 'MobileNet':
                test_face = (test_face - mean) / std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)

            # start = time.time()
            if args.backbone == 'MobileFaceNet':
                landmark = model(input)[0].cpu().data.numpy()
            else:
                landmark = model(input).cpu().data.numpy()
            # end = time.time()
            # print('Time: {:.6f}s.'.format(end - start))

            landmark = landmark.reshape(-1, 2)

            landmark = new_bbox.reprojectLandmark(landmark)

        mframe = drawLandmark_butterfly(frame, new_bbox, landmark)  # 나비모양 랜드마크

        pframe = drawPolyline(mframe, landmark)  # 다각형 그리기

        ROI = getROI(frame, landmark)  # ROI 추출

        w_ROI = warptoRect(frame, landmark)  # 직사각형으로 워핑

        X, Y = getXY(w_ROI, option)

        X_f = butter_bandpass_filter(X, f_low, f_high, fps)  # 대역 통과 필터
        Y_f = butter_bandpass_filter(Y, f_low, f_high, fps)

        chrominance = getChrom(X_f, Y_f, option)  # 채도 신호 추출

        if n == 0:
            mean_chrom_signal[0] = np.mean(chrominance)
        else:
            mean_chrom_signal = np.append(mean_chrom_signal, np.mean(chrominance))

        hr_frame = copy.copy(frame)
        if n >= 2*fps:  # fps 동안 정보 수집
            # project 7 참고
            m_signal = butter_bandpass_filter(mean_chrom_signal, f_low, f_high, fps)
            norm_signal = normalization(m_signal)  # 정규화

            detrend_signal = scipy.signal.detrend(norm_signal)  # 추세선 삭제

            avg_signal = moving_avg(detrend_signal, 6)  # moving average

            sav_signal = savgol_filter(avg_signal, 5, 2, mode="nearest")  # 아웃라이어 제거
            s_signal = butter_bandpass_filter(sav_signal, f_low, f_high, fps)

            hr = get_bpm(s_signal, fps)
            heartrate.append(hr)

            # print("frame: {}, estimated HR: {}".format(n, str(hr)))

            if n % (2*fps) == 0 and n != 2*fps:
                sum = 0
                for i in heartrate:
                    sum += i
                avg = sum / len(heartrate)

                avg = str(round(avg, 2))
                print("frame: {}, estimated HR: {}".format(n, avg))
                f.write(avg+'\n')
                # heartrate = list()

        cv2.imshow("frame", frame)
        cv2.imshow("mframe", mframe)
        cv2.imshow("ROI", ROI)
        cv2.imshow("warped ROI", w_ROI)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        n += 1

    plt.plot(heartrate)
    plt.title("frame: " + str(n) + " heartrate")
    plt.show()

    f.close()
cap.release()
cv2.destroyAllWindows()
