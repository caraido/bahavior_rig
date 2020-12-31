import numpy as np
import scipy.fft as fft
import scipy.signal as sig
from nptdms import TdmsFile
import os
import scipy.io as sio

sample_rate = 3e5

# turn tdms to np array
def read_audio(path, raw_data_flag=False):
    dir_list = os.listdir(path)
    item_list = [path + item for item in dir_list if '.tdms' in item and '.tdms_index' not in item]
    f = item_list[0]

    with TdmsFile.open(f) as file:
        group = file.groups()[0]
        channel = group.channels()[0]
        data = channel[:]

        if raw_data_flag:
            raw_data = channel.raw_data
        else:
            raw_data = None
        return data, raw_data

# resampling
def audio_resample(data:np.ndarray, old_samprate, new_samprate):
    new_sample_size = int(data.size * new_samprate/old_samprate)
    new_data = sig.resample(data, new_sample_size)
    return new_data

# pitch changing functions. reference: https://nbviewer.jupyter.org/github/prandoni/COM303-Py3/blob/master/VoiceTransformer/VoiceTransformer.ipynb
# DFT based pitch shift. doesn't preserve the position of formants.
def win_taper(N, a):
    R = int(N * a / 2)
    r = np.arange(0, R) / float(R)
    win = np.r_[r, np.ones(N - 2*R), r[::-1]]
    stride = N - R - 1
    return win, stride

def ms2smp(ms, Fs):
    return int(float(Fs) * float(ms) / 1000.0)

def DFT_rescale(x, f):
    X = np.fft.fft(x)
    # separate even and odd lengths
    parity = (len(X) % 2 == 0)
    N = int(len(X) / 2) + 1 if parity else (len(X) + 1) / 2
    Y = np.zeros(N, dtype=np.complex)
    # work only in the first half of the DFT vector since input is real
    for n in range(0, N):
        # accumulate original frequency bins into rescaled bins
        ix = int(n * f)
        if ix < N:
            Y[ix] += X[n]
    # now rebuild a Hermitian-symmetric DFT
    Y = np.r_[Y, np.conj(Y[-2:0:-1])] if parity else np.r_[Y, np.conj(Y[-1:0:-1])]
    return np.real(np.fft.ifft(Y))

def DFT_pshift(x, f, G, overlap=0):
    N = len(x)
    y = np.zeros(N)
    win, stride = win_taper(G, overlap)
    for n in range(0, len(x) - G, stride):
        w = DFT_rescale(x[n:n+G] * win, f)
        y[n:n+G] += w * win
    return y

if __name__ == '__main__':
    GregVoice_path = 'C:\\Users\\SchwartzLab\\Desktop\\2020-10-29-GregVoice\\'
    AngryMouse_path = 'C:\\Users\\SchwartzLab\\Desktop\\2020-10-29-angry-female-mouse-held-close\\'
    AlecMusic_path = 'C:\\Users\\SchwartzLab\\Desktop\\AlecMusic_grounded_01VPa_316\\'
    Testing_path = 'C:\\Users\\SchwartzLab\\Desktop\\2020-11-17-mom\\'

    # get path
    chosen_path = Testing_path
    # load data
    data, _ = read_audio(path=chosen_path)

    # downsampling: Fs from 300kHz to 32kHz
    #resampled_data = audio_resample(data=data,old_samprate=sample_rate,new_samprate=3.2e4)

    # tone shifting from 90kHz to 30kHz is to rescale the frequency 0.3 times
    #shifted_data = DFT_pshift(data, 0.3, ms2smp(40, Fs=sample_rate))
    # or
    #shifted_resampled_data = DFT_pshift(resampled_data, 0.3, ms2smp(40, Fs=3.2e4))

    # specify
    sio.savemat(chosen_path+'resampled.mat',{'momandpups': data})






