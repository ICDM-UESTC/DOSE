from scipy.signal import stft, get_window, correlate, resample
from scipy.linalg import solve_toeplitz, toeplitz
from pesq import pesq as pesq_inner  # pip install https://github.com/ludlows/python-pesq/archive/master.zip
from pesq import PesqError
# from pypesq import pesq
import librosa
from pystoi.stoi import stoi  # https://github.com/mpariente/pystoi
import numpy as np
import torch
import sys


#################################################
#
#  SPEECH ENHANCEMENT PERFORMANCE METRICS
#
#  DENGFENG.KE @ 2020-03-14 BEIJING.CHINA
#
#  YOU NEED TO INSTALL THESE PACKAGES FIRST
#
#  pip3 install Cython
#  pip3 install https://github.com/ludlows/python-pesq/archive/refs/heads/dev.zip
#  pip3 install pystoi
#

def extractOverlappedWindows(x, nperseg, noverlap, window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result


def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps = np.finfo(np.float64).eps

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extractOverlappedWindows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extractOverlappedWindows(processed_speech, winlength, winlength - skiprate, hannWin)

    signal_energy = np.power(clean_speech_framed, 2).sum(-1)
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
    segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
    segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
    segmental_snr = segmental_snr[:-1]  # remove last frame -> not valid
    return np.mean(segmental_snr)


def fwSNRseg(cleanSig, enhancedSig, fs, frameLen=0.03, overlap=0.75):
    if cleanSig.shape != enhancedSig.shape:
        raise ValueError('The two signals do not match!')
    eps = np.finfo(np.float64).eps
    cleanSig = cleanSig.astype(np.float64) + eps
    enhancedSig = enhancedSig.astype(np.float64) + eps
    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    max_freq = fs / 2  # maximum bandwidth
    num_crit = 25  # number of critical bands
    n_fft = 2 ** np.ceil(np.log2(2 * winlength))
    n_fftby2 = int(n_fft / 2)
    gamma = 0.2

    cent_freq = np.zeros((num_crit,))
    bandwidth = np.zeros((num_crit,))

    cent_freq[0] = 50.0000
    bandwidth[0] = 70.0000
    cent_freq[1] = 120.000
    bandwidth[1] = 70.0000
    cent_freq[2] = 190.000
    bandwidth[2] = 70.0000
    cent_freq[3] = 260.000
    bandwidth[3] = 70.0000
    cent_freq[4] = 330.000
    bandwidth[4] = 70.0000
    cent_freq[5] = 400.000
    bandwidth[5] = 70.0000
    cent_freq[6] = 470.000
    bandwidth[6] = 70.0000
    cent_freq[7] = 540.000
    bandwidth[7] = 77.3724
    cent_freq[8] = 617.372
    bandwidth[8] = 86.0056
    cent_freq[9] = 703.378
    bandwidth[9] = 95.3398
    cent_freq[10] = 798.717
    bandwidth[10] = 105.411
    cent_freq[11] = 904.128
    bandwidth[11] = 116.256
    cent_freq[12] = 1020.38
    bandwidth[12] = 127.914
    cent_freq[13] = 1148.30
    bandwidth[13] = 140.423
    cent_freq[14] = 1288.72
    bandwidth[14] = 153.823
    cent_freq[15] = 1442.54
    bandwidth[15] = 168.154
    cent_freq[16] = 1610.70
    bandwidth[16] = 183.457
    cent_freq[17] = 1794.16
    bandwidth[17] = 199.776
    cent_freq[18] = 1993.93
    bandwidth[18] = 217.153
    cent_freq[19] = 2211.08
    bandwidth[19] = 235.631
    cent_freq[20] = 2446.71
    bandwidth[20] = 255.255
    cent_freq[21] = 2701.97
    bandwidth[21] = 276.072
    cent_freq[22] = 2978.04
    bandwidth[22] = 298.126
    cent_freq[23] = 3276.17
    bandwidth[23] = 321.465
    cent_freq[24] = 3597.63
    bandwidth[24] = 346.136

    W = np.array([0.003, 0.003, 0.003, 0.007, 0.010, 0.016, 0.016, 0.017, 0.017, 0.022,
                  0.027, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.036, 0.036, 0.033,
                  0.030, 0.029, 0.027, 0.026, 0.026])

    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # % -30 dB point of filter

    all_f0 = np.zeros((num_crit,))
    crit_filter = np.zeros((num_crit, int(n_fftby2)))
    j = np.arange(0, n_fftby2)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    num_frames = len(cleanSig) / skiprate - (winlength / skiprate)  # number of frames
    start = 1  # starting sample
    # window     = 0.5*(1 - cos(2*pi*(1:winlength).T/(winlength+1)));

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    f, t, Zxx = stft(cleanSig[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    clean_spec = np.abs(Zxx)
    clean_spec = clean_spec[:-1, :]
    clean_spec = (clean_spec / clean_spec.sum(0))
    f, t, Zxx = stft(enhancedSig[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    enh_spec = np.abs(Zxx)
    enh_spec = enh_spec[:-1, :]
    enh_spec = (enh_spec / enh_spec.sum(0))

    clean_energy = (crit_filter.dot(clean_spec))
    processed_energy = (crit_filter.dot(enh_spec))
    error_energy = np.power(clean_energy - processed_energy, 2)
    error_energy[error_energy < eps] = eps
    W_freq = np.power(clean_energy, gamma)
    SNRlog = 10 * np.log10((clean_energy ** 2) / error_energy)
    fwSNR = np.sum(W_freq * SNRlog, 0) / np.sum(W_freq, 0)
    distortion = fwSNR.copy()
    distortion[distortion < -10] = -10
    distortion[distortion > 35] = 35

    return np.mean(distortion)


# def lpcoeff(speech_frame, model_order):
#    # ----------------------------------------------------------
#    # (1) Compute Autocorrelation Lags
#    # ----------------------------------------------------------

#     R=correlate(speech_frame,speech_frame)
#     R=R[len(speech_frame)-1:len(speech_frame)+model_order]
#    # ----------------------------------------------------------
#    # (2) Levinson-Durbin
#    # ----------------------------------------------------------
#     lpparams=np.ones((model_order+1))
#     lpparams[1:]=solve_toeplitz(R[0:-1],-R[1:])
#     return(lpparams,R)


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocor lags
    eps = np.finfo(np.float64).eps
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        # rcoeff[i] = (R[i+1] - sum_term)/E[i] # fixed by LiHongfeng
        rcoeff[i] = (R[i + 1] - sum_term) / max(E[i], eps)
        # if E[i] == 0:
        #   print(233333, i, eps==0)
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)

    # return acorr, refcoeff, lpparams
    return lpparams, acorr


def llr(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps = np.finfo(np.float64).eps
    alpha = 0.95
    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    if fs < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extractOverlappedWindows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extractOverlappedWindows(processed_speech, winlength, winlength - skiprate, hannWin)
    numFrames = clean_speech_framed.shape[0]
    numerators = np.zeros((numFrames - 1,))
    denominators = np.zeros((numFrames - 1,))

    for ii in range(numFrames - 1):
        A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
        A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)

        numerators[ii] = A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
        denominators[ii] = A_clean.dot(toeplitz(R_clean).dot(A_clean.T))

    frac = numerators / denominators
    frac[frac <= 0] = 1000
    distortion = np.log(frac)
    # distortion[distortion>2]=2 # this line is not in composite measure matlab implementation of loizou
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def findLocPeaks(slope, energy):
    num_crit = len(energy)

    loc_peaks = np.zeros_like(slope)

    for ii in range(len(slope)):
        n = ii
        if slope[ii] > 0:
            while ((n < num_crit - 1) and (slope[n] > 0)):
                n = n + 1
            loc_peaks[ii] = energy[n - 1]
        else:
            while ((n >= 0) and (slope[n] <= 0)):
                n = n - 1
            loc_peaks[ii] = energy[n + 1]

    return loc_peaks


def wss(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    Kmax = 20  # value suggested by Klatt, pg 1280
    Klocmax = 1  # value suggested by Klatt, pg 1280
    alpha = 0.95
    if clean_speech.shape != processed_speech.shape:
        raise ValueError('The two signals do not match!')
    eps = np.finfo(np.float64).eps
    clean_speech = clean_speech.astype(np.float64) + eps
    processed_speech = processed_speech.astype(np.float64) + eps
    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    max_freq = fs / 2  # maximum bandwidth
    num_crit = 25  # number of critical bands
    n_fft = 2 ** np.ceil(np.log2(2 * winlength))
    n_fftby2 = int(n_fft / 2)

    cent_freq = np.zeros((num_crit,))
    bandwidth = np.zeros((num_crit,))

    cent_freq[0] = 50.0000
    bandwidth[0] = 70.0000
    cent_freq[1] = 120.000
    bandwidth[1] = 70.0000
    cent_freq[2] = 190.000
    bandwidth[2] = 70.0000
    cent_freq[3] = 260.000
    bandwidth[3] = 70.0000
    cent_freq[4] = 330.000
    bandwidth[4] = 70.0000
    cent_freq[5] = 400.000
    bandwidth[5] = 70.0000
    cent_freq[6] = 470.000
    bandwidth[6] = 70.0000
    cent_freq[7] = 540.000
    bandwidth[7] = 77.3724
    cent_freq[8] = 617.372
    bandwidth[8] = 86.0056
    cent_freq[9] = 703.378
    bandwidth[9] = 95.3398
    cent_freq[10] = 798.717
    bandwidth[10] = 105.411
    cent_freq[11] = 904.128
    bandwidth[11] = 116.256
    cent_freq[12] = 1020.38
    bandwidth[12] = 127.914
    cent_freq[13] = 1148.30
    bandwidth[13] = 140.423
    cent_freq[14] = 1288.72
    bandwidth[14] = 153.823
    cent_freq[15] = 1442.54
    bandwidth[15] = 168.154
    cent_freq[16] = 1610.70
    bandwidth[16] = 183.457
    cent_freq[17] = 1794.16
    bandwidth[17] = 199.776
    cent_freq[18] = 1993.93
    bandwidth[18] = 217.153
    cent_freq[19] = 2211.08
    bandwidth[19] = 235.631
    cent_freq[20] = 2446.71
    bandwidth[20] = 255.255
    cent_freq[21] = 2701.97
    bandwidth[21] = 276.072
    cent_freq[22] = 2978.04
    bandwidth[22] = 298.126
    cent_freq[23] = 3276.17
    bandwidth[23] = 321.465
    cent_freq[24] = 3597.63
    bandwidth[24] = 346.136

    W = np.array([0.003, 0.003, 0.003, 0.007, 0.010, 0.016, 0.016, 0.017, 0.017, 0.022, 0.027,
                  0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.036, 0.036, 0.033, 0.030, 0.029,
                  0.027, 0.026, 0.026])

    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # % -30 dB point of filter

    all_f0 = np.zeros((num_crit,))
    crit_filter = np.zeros((num_crit, int(n_fftby2)))
    j = np.arange(0, n_fftby2)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0)
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    num_frames = len(clean_speech) / skiprate - (winlength / skiprate)  # number of frames
    start = 1  # starting sample

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    scale = np.sqrt(1.0 / hannWin.sum() ** 2)

    f, t, Zxx = stft(clean_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    clean_spec = np.power(np.abs(Zxx) / scale, 2)
    clean_spec = clean_spec[:-1, :]

    f, t, Zxx = stft(processed_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    proc_spec = np.power(np.abs(Zxx) / scale, 2)
    proc_spec = proc_spec[:-1, :]

    clean_energy = (crit_filter.dot(clean_spec))
    log_clean_energy = 10 * np.log10(clean_energy)
    log_clean_energy[log_clean_energy < -100] = -100
    proc_energy = (crit_filter.dot(proc_spec))
    log_proc_energy = 10 * np.log10(proc_energy)
    log_proc_energy[log_proc_energy < -100] = -100

    log_clean_energy_slope = np.diff(log_clean_energy, axis=0)
    log_proc_energy_slope = np.diff(log_proc_energy, axis=0)

    dBMax_clean = np.max(log_clean_energy, axis=0)
    dBMax_processed = np.max(log_proc_energy, axis=0)

    numFrames = log_clean_energy_slope.shape[-1]

    clean_loc_peaks = np.zeros_like(log_clean_energy_slope)
    proc_loc_peaks = np.zeros_like(log_proc_energy_slope)
    for ii in range(numFrames):
        clean_loc_peaks[:, ii] = findLocPeaks(log_clean_energy_slope[:, ii], log_clean_energy[:, ii])
        proc_loc_peaks[:, ii] = findLocPeaks(log_proc_energy_slope[:, ii], log_proc_energy[:, ii])

    Wmax_clean = Kmax / (Kmax + dBMax_clean - log_clean_energy[:-1, :])
    Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peaks - log_clean_energy[:-1, :])
    W_clean = Wmax_clean * Wlocmax_clean

    Wmax_proc = Kmax / (Kmax + dBMax_processed - log_proc_energy[:-1])
    Wlocmax_proc = Klocmax / (Klocmax + proc_loc_peaks - log_proc_energy[:-1, :])
    W_proc = Wmax_proc * Wlocmax_proc

    W = (W_clean + W_proc) / 2.0

    distortion = np.sum(W * (log_clean_energy_slope - log_proc_energy_slope) ** 2, axis=0)
    distortion = distortion / np.sum(W, axis=0)
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def pesq(clean_speech, processed_speech, fs):
    try:
        if fs == 8000:
            pesq_mos = pesq_inner(fs, clean_speech, processed_speech, 'nb')
            pesq_mos = 46607 / 14945 - (
                    2000 * np.log(1 / (pesq_mos / 4 - 999 / 4000) - 1)) / 2989  # remap to raw pesq score

        elif fs == 16000:
            pesq_mos = pesq_inner(fs, clean_speech, processed_speech, 'wb')
        elif fs >= 16000:
            numSamples = round(len(clean_speech) / fs * 16000)
            pesq_mos = pesq_inner(fs, resample(clean_speech, numSamples),
                                  resample(processed_speech, numSamples), 'wb')
        else:
            numSamples = round(len(clean_speech) / fs * 8000)
            pesq_mos = pesq_inner(fs, resample(clean_speech, numSamples),
                                  resample(processed_speech, numSamples), 'nb')
            pesq_mos = 46607 / 14945 - (
                    2000 * np.log(1 / (pesq_mos / 4 - 999 / 4000) - 1)) / 2989  # remap to raw pesq score
    except PesqError:
        return 0.0
    return pesq_mos


def composite(clean_speech, processed_speech, fs):
    wss_dist = wss(clean_speech, processed_speech, fs)
    llr_mean = llr(clean_speech, processed_speech, fs)
    segSNR = SNRseg(clean_speech, processed_speech, fs)
    pesq_mos = pesq(clean_speech, processed_speech, fs)
    Stoi = stoi(clean_speech, processed_speech, fs)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    Csig = np.max((1, Csig))
    Csig = np.min((5, Csig))  # limit values to [1, 5]
    Cbak = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5, Cbak))  # limit values to [1, 5]
    Covl = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl))  # limit values to [1, 5]

    '''Simplified version for fast debug'''
    # segSNR, Csig, Cbak, Covl = 0, 0, 0, 0
    return segSNR, pesq_mos, Csig, Cbak, Covl, Stoi


def compareone(args):
    import soundfile as sf
    import numpy as np

    # clean, processed = args
    # c,fc = sf.read(clean)
    # p,fp = sf.read(processed)

    c, p = args

    assert len(c) == len(p), 'c.shape=%r, p.shape=%r' % (c.shape, p.shape)
    # assert fc == fp, 'fc=%d fp=%d'%(fc,fp)

    try:
        ssnr, pesq, csig, cbak, covl, stoi = composite(c, p, 16000)
    except np.linalg.LinAlgError:
        print("np.linalg.LinAlgError", flush=True)
        ssnr, pesq, csig, cbak, covl, stoi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # name = clean.split('/')[-1]
    # print('[%s] ssnr:%5.2f pesq:%5.2f csig:%5.2f cbak:%5.2f covl:%5.2f'%(name,
    #             ssnr, pesq, csig, cbak, covl), flush=True)

    return csig, cbak, covl, pesq, ssnr, stoi


def compareone_load_wav(args):
    import soundfile as sf
    import numpy as np

    clean, processed = args

    c, fc = librosa.load(clean, sr=16000)
    p, fp = librosa.load(processed, sr=16000)

    # c, p = args

    assert len(c) == len(p), 'c.shape=%r, p.shape=%r' % (c.shape, p.shape)
    # assert fc == fp, 'fc=%d fp=%d'%(fc,fp)

    try:
        ssnr, pesq, csig, cbak, covl, stoi = composite(c, p, 16000)
    except np.linalg.LinAlgError:
        print("np.linalg.LinAlgError", flush=True)
        ssnr, pesq, csig, cbak, covl, stoi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # name = clean.split('/')[-1]
    # print('[%s] ssnr:%5.2f pesq:%5.2f csig:%5.2f cbak:%5.2f covl:%5.2f'%(name,
    #             ssnr, pesq, csig, cbak, covl), flush=True)

    return csig, cbak, covl, pesq, ssnr, stoi


def compare_complex(esti_list, label_list, frame_list, feat_type='sqrt'):
    all_csig_list, all_cbak_list, all_covl_list, all_pesq_list, all_ssnr_list, all_stoi_list = [], [], [], [], [], []
    with torch.no_grad():
        esti_mag, esti_phase = torch.norm(esti_list, dim=1), torch.atan2(esti_list[:, -1, :, :], esti_list[:, 0, :, :])
        label_mag, label_phase = torch.norm(label_list, dim=1), torch.atan2(label_list[:, -1, :, :],
                                                                            label_list[:, 0, :, :])
        if feat_type == 'sqrt':
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** 2
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'cubic':
            esti_mag = esti_mag ** (10 / 3)
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** (10 / 3)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'log_1x':
            esti_mag = torch.exp(esti_mag) - 1
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = torch.exp(label_mag) - 1
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        else:
            esti_com = esti_list
            label_com = label_list
        clean_utts, esti_utts = [], []
        utt_num = label_list.size()[0]
        for i in range(utt_num):
            # print("utt_num: ", i)
            tf_esti = esti_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            tf_esti_com = torch.complex(tf_esti[:, :, :, 0], tf_esti[:, :, :, 1])
            t_esti = torch.istft(tf_esti_com, n_fft=320, hop_length=160, win_length=320,
                                 window=torch.hann_window(320)).transpose(1, 0).squeeze(dim=-1).numpy()
            tf_label = label_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            tf_label_com = torch.complex(tf_label[:, :, :, 0], tf_label[:, :, :, 1])
            t_label = torch.istft(tf_label_com, n_fft=320, hop_length=160, win_length=320,
                                  window=torch.hann_window(320)).transpose(1, 0).squeeze(
                dim=-1).numpy()
            t_len = (frame_list[i] - 1) * 160
            t_esti, t_label = t_esti[:t_len], t_label[:t_len]
            esti_utts.append(t_esti)
            clean_utts.append(t_label)

        for c, p in zip(clean_utts, esti_utts):
            # print("clean_utts: ", c)
            batch_result = compareone((c, p))
            all_csig_list.append(batch_result[0])
            all_cbak_list.append(batch_result[1])
            all_covl_list.append(batch_result[2])
            all_pesq_list.append(batch_result[3])
            all_ssnr_list.append(batch_result[4])
            all_stoi_list.append(batch_result[5])
    return np.mean(all_csig_list), np.mean(all_cbak_list), np.mean(all_covl_list), np.mean(all_pesq_list), np.mean(
        all_ssnr_list), np.mean(all_stoi_list)


def compare(refdir, degdir, use_tqdm=True):
    from multiprocessing import cpu_count
    from multiprocessing import Pool
    import numpy as np
    import glob, os

    if os.path.isfile(refdir) and os.path.isfile(degdir):
        return [compareone_load_wav([refdir, degdir])]

    n = np.min([np.max([cpu_count() - 2, 1]), 20])
    pool = Pool(processes=n)
    reffiles = sorted(glob.glob('%s/*.wav' % refdir))
    degfiles = sorted(glob.glob('%s/*.wav' % degdir))

    assert len(reffiles) == len(degfiles)

    import tqdm
    args = list(zip(reffiles, degfiles))
    if use_tqdm:
        res = list(tqdm.tqdm(pool.imap(compareone_load_wav, args), "Calculating", ncols=60))
    else:
        res = list(pool.imap(compareone_load_wav, args))
    pool.close()
    pool.join()
    return res


if __name__ == "__main__":
    import numpy as np
    import sys, time

    t1 = time.time()
    res = compare(sys.argv[1], sys.argv[2])
    t2 = time.time()

    pm = np.array([x[0:] for x in res])
    pm = np.mean(pm, axis=0)
    print('time: %.3f' % (t2 - t1))
    print('ref=', sys.argv[1])
    print('deg=', sys.argv[2])
    print('csig:%6.4f cbak:%6.4f covl:%6.4f pesq:%6.4f ssnr:%6.4f stoi:%6.4f' % tuple(pm))