from pulse import *
import numpy as np
import cv2

def CalGroundTruth(fps, groudntruth):
    heartrate = list()
    pulse = Pulse(fps, len(groudntruth))
    HR = pulse.get_rfft_hr(groudntruth)
    # print(HR)
    hr = round(HR, 2)
    heartrate.append(hr)

    return hr

if __name__ == '__main__':
    f = open("src/gt30.txt", 'r')
    f_w = open("results/ground_truth_HR.txt", 'w')
    groundtruth = list()

    gt = f.read().split('  ')  # '   ' or
    for i in gt:
        if len(i) == 0:
            continue
        i = float(i)
        groundtruth.append(i)

    groundtruth = np.asarray(groundtruth)

    cap = cv2.VideoCapture("src/subject30.avi")
    fps = 30 # int(cap.get(cv2.CAP_PROP_FPS))

    n = 0
    k = 0
    while cap.isOpened():
        ret, frame = cap.read()  # ret 은 프레임 읽기를 성공하면 True 값 반환
        if not ret:
            print("End or Error")
            break

        sum = 0

        if k % (2*fps) == 0 and k >= 4 * fps:
            tmp = groundtruth[:n + 2*fps + 1]
            for i in tmp:
                sum += i
            avg = sum / len(tmp)
            avg = str(round(avg, 2))
            print("frame: {}, estimated HR: {}".format(k, avg))
            f_w.write(avg + "\n")
            tmp = list()
            n += 2*fps
        k += 1

    f.close()
    f_w.close()