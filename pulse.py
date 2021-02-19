import numpy as np
from cdf import CDF
from asf import ASF
from numpy.linalg import inv
from common.utils import *
import pywt

from scipy.fftpack import rfftfreq, rfft
import cv2
from torchvision import transforms
import pdb
from PIL import Image

PRE_STEP_ASF = False  
PRE_STEP_CDF = False

class Pulse():
    def __init__(self, framerate, signal_size):
        self.framerate = framerate
        self.signal_size = signal_size
        self.minFreq = 0.7 #
        self.maxFreq = 4 #
        self.fft_spec = []
        
    def get_pulse(self, mean_rgb):
        seg_t = 3.2
        l = int(self.framerate * seg_t)
        H = np.zeros(self.signal_size)

        B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]
                
        for t in range(0, (self.signal_size - l + 1)):
            # pre processing steps
            C = mean_rgb[t:t+l,:].T

            if PRE_STEP_CDF:
                C = CDF(C, B)
           
            if PRE_STEP_ASF:
                C = ASF(C)
           
            # POS
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv,C)
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])
            S = np.matmul(projection_matrix,Cn)
            std = np.array([1, np.std(S[0,:])/np.std(S[1, :])])
            P = np.matmul(std, S)
            H[t:t+l] = H[t:t+l] + (P-np.mean(P))
        return H

    def get_rfft_hr(self, signal):
        signal_size = len(signal)
        signal = signal.flatten()
        fft_data = np.fft.rfft(signal)  # FFT
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1./self.framerate)  # Frequency data

        inds = np.where((freq < self.minFreq) | (freq > self.maxFreq))[0]
        fft_data[inds] = 0
        bps_freq = 60.0 * freq
        max_index = np.argmax(fft_data)
        fft_data[max_index] = fft_data[max_index]**2
        self.fft_spec.append(fft_data)
        HR = bps_freq[max_index]

        return HR

def get_bpm(Sig, rate= 30.0):
   sig = Sig.copy()
   n = len(sig)
   # print(n)
   fps = rate

   win = windows.hann(sig.size)
   sig = sig - np.expand_dims(np.mean(sig, -1), -1)
   sig = sig * win

   filtered_sig = butter_bandpass_filter(sig, 0.7, 4, fps, order=5)

   f, Pxx_den = welch(filtered_sig, fps, nperseg=n)
   index = np.argmax(Pxx_den)
   HR_estimate = round(f[index]*60.0, 2)

   return HR_estimate