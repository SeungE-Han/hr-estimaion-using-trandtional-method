"""
This code performs a real-time face and landmark detections
1. Use a light-weight face detector (ONNX): https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
2. Use mobilefacenet as a light-weight landmark detector (OpenVINO: 10 times faster than ONNX)
Date: 09/27/2020 by Cunjian Chen (ccunjian@gmail.com)
"""
import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import os

# onnx runtime
import onnxruntime as ort
import onnx
import onnxruntime

# import libraries for landmark
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from PIL import Image
import torchvision.transforms as transforms

# import openvino 
from openvino.inference_engine import IENetwork, IECore
from common.utils import *
from pulse import *
import scipy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import math

ie = IECore()
model_bin = os.path.splitext("openvino/mobilefacenet.xml")[0] + ".bin"
net = ie.read_network(model="openvino/mobilefacenet.xml", weights=model_bin)
input_blob = next(iter(net.input_info))
#plugin = IEPlugin(device="CPU")
#exec_net = plugin.load(network=net)
exec_net = ie.load_network(network=net,device_name="CPU")

# setup the parameters
resize = transforms.Resize([112, 112])
to_tensor = transforms.ToTensor()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])


# face detection setting
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"

onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# perform face detection and alignment from camera
cap = cv2.VideoCapture("D:/dbs/UBFC-rPPG/UBFC_DATASET/DATASET_2/subject30/vid.avi")  # capture from camera  "src/3rotazione_dx2.webm"
fps = 30  # int(cap.get(cv2.CAP_PROP_FPS))
threshold = 0.7

f_low = 0.7
f_high = 4
option = "pos"  # "chrom" - pos로 할건지 chrom으로 할건지 결정

detection_time = 0
sum = 0
n = 0
mean_chrom_signal = np.empty((1, 1), object)
heartrate = list()
print("Gathering information....")
f = open("results/openvino_result.txt", 'w')

while True:
    ret, frame = cap.read()
    if frame is None:
        print("end")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    # print("cost time:{}".format(time.time() - time_time))
    boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, threshold)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        # perform landmark detection
        out_size = 112
        img = frame.copy()
        height, width, _ = img.shape
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(max([w, h]))
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
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
        cropped = img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
        cropped_face = cv2.resize(cropped, (out_size, out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            continue
        test_face = cropped_face.copy()   
        test_face = test_face/255.0
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)  
        # OpenVINO Inference
        start = time.time()
        outputs = exec_net.infer(inputs={input_blob: test_face})
        key = list(outputs.keys())[0]
        output = outputs[key]
        landmark = output[0].reshape(-1, 2)
        end = time.time()
        # print('Time: {:.6f}s.'.format(end - start))
        detection_time += end-start
        # print(detection_time)
        landmark = landmark.reshape(-1, 2)
        landmark = new_bbox.reprojectLandmark(landmark)

    mframe = drawLandmark_butterfly(frame, new_bbox, landmark)  # 나비모양 랜드마크

    pframe = drawPolyline(mframe, landmark)  # 다각형 그리기

    ROI = getROI(frame, landmark)  # ROI 추출

    w_ROI = warptoRect(frame, landmark)  # 직사각형으로 워핑

    X, Y = getXY(w_ROI, option)

    X_f = butter_bandpass_filter(X, f_low, f_high, fps)
    Y_f = butter_bandpass_filter(Y, f_low, f_high, fps)

    chrominance = getChrom(X_f, Y_f, option)  # 채도 신호 추출

    if n == 0:
        mean_chrom_signal[0] = np.mean(chrominance)
    else:
        mean_chrom_signal = np.append(mean_chrom_signal, np.mean(chrominance))

    hr_frame = copy.copy(frame)
    if n >= 2*fps:  # fps 동안 정보 수집
    # if int(detection_time) == 1:
        # project 7 참고
        m_signal = butter_bandpass_filter(mean_chrom_signal, f_low, f_high, fps)
        norm_signal = normalization(m_signal)  # 정규화

        detrend_signal = scipy.signal.detrend(norm_signal)  # 추세선 삭제

        avg_signal = moving_avg(detrend_signal, 6)  # moving average

        sav_signal = savgol_filter(avg_signal, 5, 2, mode="nearest")  # 아웃라이어 제거 17
        s_signal = butter_bandpass_filter(sav_signal, f_low, f_high, fps)

        hr = get_bpm(s_signal, fps)
        heartrate.append(hr)

        if n % (2*fps) == 0 and n != 2*fps:
        # if int(detection_time) % 2 == 0:
            su = 0
            for i in heartrate:
                su += i
            avg = su / len(heartrate)

            avg = str(round(avg, 2))
            print("frame: {}, estimated HR: {}".format(n, avg))
            f.write(avg + '\n')
            # heartrate = list()
            # detection_time = 0

    sum += boxes.shape[0]
    cv2.imshow("frame", frame)
    cv2.imshow("mframe", mframe)
    cv2.imshow("ROI", ROI)
    cv2.imshow("warped ROI", w_ROI)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    n += 1
"""
m_signal = butter_bandpass_filter(mean_chrom_signal, f_low, f_high, fps)

norm_signal = normalization(m_signal)  # 정규화
plt.subplot(221)
plt.title("normalization")
plt.plot(norm_signal)

detrend_signal = scipy.signal.detrend(norm_signal)  # 추세선 삭제
plt.subplot(222)
plt.title("detrend")
plt.plot(detrend_signal)

avg_signal = moving_avg(detrend_signal, 6)  # moving average
plt.subplot(223)
plt.title("moving average")
plt.plot(avg_signal)

sav_signal = savgol_filter(avg_signal, 5, 2, mode="nearest")  # 아웃라이어 제거 17
s_signal = butter_bandpass_filter(sav_signal, f_low, f_high, fps)
plt.subplot(224)
plt.title("savgol_filter")
plt.plot(s_signal)

plt.show()

# pulse = Pulse(fps, len(s_signal))
# HR = pulse.get_rfft_hr(s_signal)
# hr = round(HR, 2)
hr = get_bpm(s_signal, fps)
print(hr)

plt.plot(heartrate)
plt.title("frame: " + str(n) + " heartrate")
plt.show()

"""
f.close()
cap.release()
cv2.destroyAllWindows()

