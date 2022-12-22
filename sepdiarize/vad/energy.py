import numpy as np
import sys, os, errno
from glob import glob
import argparse
import scipy.linalg as spl
import numexpr as ne
import h5py
import soundfile as sf
from scipy.io import wavfile
from scipy.special import logsumexp
from tqdm import tqdm


def gmm_eval(data, GMM, return_accums=0):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.
    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistics.
    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistics.
    For full covariance model second order statistics, only the vectorized upper
    triangular parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data
    # computation of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']
    llh = logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = np.exp(gamma.T - llh)
    N = gamma.sum(axis=1)
    F = gamma.dot(data)
    if return_accums == 1:
        return llh, N, F

    S = gamma.dot(data_sqr)
    return llh, N, F, S


def gmm_eval_prep(weights, means, covs):
    n_mix, dim = means.shape
    GMM = dict()
    is_full_cov = covs.shape[1] != dim
    GMM['utr'], GMM['utc'] = uppertri_indices(dim, not is_full_cov)

    if is_full_cov:
        GMM['gconsts'] = np.zeros(n_mix)
        GMM['invCovs'] = np.zeros_like(covs)
        GMM['invCovMeans']=np.zeros_like(means)
        for ii in range(n_mix):
            uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc'])
            invC, logdetC = inv_posdef_and_logdet(uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc']))

            #log of Gauss. dist. normalizer + log weight + mu' invCovs mu
            invCovMean = invC.dot(means[ii])
            GMM['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['invCovMeans'][ii] = invCovMean

            #Iverse covariance matrices are stored in columns of 2D matrix as vectorized upper triangular parts ...
            GMM['invCovs'][ii] = uppertri1d_from_sym(invC, GMM['utr'], GMM['utc']);
        # ... with elements above the diagonal multiplied by 2
        GMM['invCovs'][:,dim:] *= 2.0
    else: #for diagonal
        GMM['invCovs']  = 1 / covs;
        GMM['gconsts']  = np.log(weights) - 0.5 * (np.sum(np.log(covs) + means**2 * GMM['invCovs'], axis=1) + dim * np.log(2.0*np.pi))
        GMM['invCovMeans'] = GMM['invCovs'] * means

    # for weight = 0, prepare GMM for uninitialized model with single Gaussian
    if len(weights) == 1 and weights[0] == 0:
        GMM['invCovs']     = np.zeros_like(GMM['invCovs'])
        GMM['invCovMeans'] = np.zeros_like(GMM['invCovMeans'])
        GMM['gconsts']     = np.ones(1)
    return GMM


def gmm_llhs(data, GMM):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.
    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistics.
    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistics.
    For full covariance model second order statistics, only the vectorized upper
    triangular parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data
    # computation of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']

    return gamma


def gmm_update(N,F,S):
    """ weights means covs = gmm_update(N,F,S) return GMM parameters, which are
    updated from accumulators
    """
    dim = F.shape[1]
    is_diag_cov = S.shape[1] == dim
    utr, utc = uppertri_indices(dim, is_diag_cov)
    sumN    = N.sum()
    weights = N / sumN
    means   = F / N[:,np.newaxis]
    covs    = S / N[:,np.newaxis] - means[:,utr] * means[:,utc]
    return weights, means, covs


def inv_posdef_and_logdet(A):
    L = np.linalg.cholesky(A)
    logdet = 2*np.sum(np.log(np.diagonal(L)))
    invA = spl.solve(A, np.identity(len(A), A.dtype), sym_pos=True)
    return invA, logdet


def logsumexp(x, axis=0):
    xmax = x.max(axis)
    x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax[not_finite]
    return x


def uppertri_indices(dim, isdiag=False):
    """ [utr utc]=uppertri_indices(D, isdiag) returns row and column indices
    into upper triangular part of DxD matrices. Indices go in zigzag fashion
    starting by diagonal. For convenient encoding of diagonal matrices, 1:D
    ranges are returned for both outputs utr and utc when ISDIAG is true.
    """
    if isdiag:
        utr = np.arange(dim)
        utc = np.arange(dim)
    else:
        utr = np.hstack([np.arange(ii)     for ii in range(dim,0,-1)])
        utc = np.hstack([np.arange(ii,dim) for ii in range(dim)])
    return utr, utc


def add_dither(x, level=8):
    return x + level * (np.random.rand(*x.shape)*2-1)


def compute_vad(s, win_length=160, win_overlap=80, n_realignment=5, threshold=0.3):
    # power signal for energy computation
    s = s**2
    # frame signal with overlap
    F = framing(s, win_length, win_length - win_overlap) 
    # sum frames to get energy
    E = F.sum(axis=1).astype(np.float64)
    
    # normalize the energy
    E -= E.mean()
    try:
        E /= E.std()
        # initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array(( 0.33, 0.33, 0.33))
    
        GMM = gmm_eval_prep(ww, mm, ee)
        
        E = E[:,np.newaxis]
        
        for i in range(n_realignment):
            # collect GMM statistics
            llh, N, F, S = gmm_eval(E, GMM, return_accums=2)
            # update model
            ww, mm, ee   = gmm_update(N, F, S)
            # wrap model
            GMM = gmm_eval_prep(ww, mm, ee)
    
        # evaluate the gmm llhs
        llhs = gmm_llhs(E, GMM)
        llh  = logsumexp(llhs, axis=1)[:,np.newaxis]
        llhs = np.exp(llhs - llh)
        
        out  = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:,0] < threshold] = True
    except RuntimeWarning:
        logging.info("File contains only silence")
        out=np.zeros(E.shape[0],dtype=np.bool)
    return out


def frame_labels2start_ends(speech_frames, frame_rate=100.0):
    decesions = np.r_[False, speech_frames, False]
    return np.nonzero(decesions[1:] != decesions[:-1])[0].reshape(-1,2) / frame_rate

    
def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def median_filter(vad, win_length=50, threshold=25):
    stats = np.r_[vad, np.zeros(win_length)].cumsum()-np.r_[np.zeros(win_length), vad].cumsum()
    return stats[win_length//2:win_length//2-win_length] > threshold
    

def energy_vad(signal, threshold, median_window_length, samplerate=8000):
    if signal.dtype != np.int16:
        signal = (signal * 2**15).astype(np.int16)
    signal = np.r_[np.zeros(samplerate // 2), signal, np.zeros(samplerate // 2)]  # add half second of "silence" at the beginning and the end
    np.random.seed(3)  # for reproducibility
    signal = add_dither(signal, 8.0)
    vad = compute_vad(signal, win_length=int(round(0.025 * samplerate)), win_overlap=int(round(0.015 * samplerate)),
                      n_realignment=5, threshold=threshold)
    vad = median_filter(vad, win_length=median_window_length, threshold=5)
    labels = frame_labels2start_ends(vad[50:-50], frame_rate=1.0) / 100.0
    return labels