

import numpy as np
from skimage.color import rgb2gray
import cv2
import os
from skimage.filters import (threshold_sauvola)

from time import time

from scipy.signal import convolve2d


import pickle
import sys


input_dir = None
output_dir = None

if(len(sys.argv) >2):
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
else:
    print("You need to put input and output dirs.")



def lpq(img, winSize=3, freqestim=1, mode='nh'):
    rho = 0.90

    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    '''
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)
    '''

    convmode = 'valid'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]
#         LPQdesc = np.histogram(LPQdesc.flatten(), range(512))[0]

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / LPQdesc.sum()
    return LPQdesc


def compute_LPQ(img,window_size = 25):
    thresh_sauvola = threshold_sauvola(img, window_size=window_size)
    image_binary_sauvola = img > thresh_sauvola
    return lpq(image_binary_sauvola)

def read_images(dir):
    
    filenames = os.listdir(dir)

    filenames.sort(key=len)
    imgs = []
    for filename in filenames:
        img = cv2.imread(os.path.join(dir,filename))
        if img is not None:
            imgs.append(rgb2gray(img))
    imgs = np.asarray(imgs)
    return imgs





if(input_dir and output_dir):
    filename = 'finalized_model_majority_vote.sav'
    loaded_model_svm = pickle.load(open(filename, 'rb'))
    test_images = read_images(input_dir)
    times = []
    predictions = []
    for img in test_images:
        t_start = time()
        lpq_result = np.asarray(compute_LPQ(img)).reshape((-1,255))
        predictions.append(loaded_model_svm.predict(lpq_result))
        times.append(time()-t_start)

    with open(output_dir+"/results.txt", 'w') as f:
        for i in range(len(predictions)):
            if i == (len(predictions)-1):
                f.write(str(predictions[i][0]))
            else:
                f.write(str(predictions[i][0])+"\n")

    with open(output_dir+"/times.txt", 'w') as f:
        for i in range(len(times)):
            if times[i] == 0.0:
                times[i] = 0.01
            if i == (len(times)-1):
                f.write('%0.3f' %times[i])
            else:
                f.write('%0.3f\n' %times[i])

