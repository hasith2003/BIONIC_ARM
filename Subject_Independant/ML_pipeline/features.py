import numpy as np
from scipy import signal

def extract_paper_features(x, fs=1000):
    vmax = np.max(np.abs(x))
    iemg = np.sum(np.abs(x))
    mav = np.mean(np.abs(x))
    ssi = np.sum(x**2)
    var = np.var(x, ddof=1) if len(x) > 1 else 0
    rms = np.sqrt(np.mean(x**2))
    
    f, Pxx = signal.welch(x, fs=fs, nperseg=len(x))
    mnf = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) != 0 else 0
    
    return [vmax, iemg, mav, ssi, var, rms, mnf]
