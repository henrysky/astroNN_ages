import os
import glob
import h5py
import itertools
import numpy as np
from tqdm import tqdm
import concurrent.futures
import astropy.units as u
import astropy.convolution as conv
from astropy.timeseries import LombScargle

import sys

sys.path.insert(0, os.path.dirname(__file__))

from __init__ import (
    _LK_CAHCE,
    apokasc_klc_h5_path,
    apokasc_kps_h5_path
)

process_pool = concurrent.futures.ProcessPoolExecutor

# based on https://ui.adsabs.harvard.edu/abs/2018ApJ...866...15N
fs = (0.009 * u.uHz).value
min_freq = (2 * u.uHz).value  # 5.78 days or 138.88 hours
max_freq = (270 * u.uHz).value  # 0.04 days or 1.02 hours
freq_range = np.arange(min_freq, max_freq + fs, fs) * u.uHz
filter_width = (0.01 * u.uHz).value
fac = np.max([1, 0.1 / fs])
kernel = conv.Gaussian1DKernel(stddev=fac)

klc_h5f_apokasc = h5py.File(apokasc_klc_h5_path, "r")


def f_process_internal(data):
    flux, time = data[0], data[1]
    time_w_u = time * u.day

    LS = LombScargle(time_w_u, flux, normalization="psd")
    # Lomb-Scargle function to units of flux_variance / [frequency unit]
    # power = LS.power(freq_range) * 2. / (len(time) * fs)  # no need to scale power spectra as we will flatten it
    power = LS.power(freq_range)

    count = np.zeros(len(freq_range.value), dtype=int)
    bkg = np.zeros_like(freq_range.value)
    x0 = np.log10(freq_range[0].value)
    corr_factor = (8.0 / 9.0) ** 3  # predfined??
    while x0 < np.log10(freq_range[-1].value):
        m = np.abs(np.log10(freq_range.value) - x0) < filter_width
        if len(bkg[m] > 0):
            bkg[m] += np.nanmedian(power[m].value) / corr_factor
            count[m] += 1
        x0 += 0.5 * filter_width
    bkg /= count
    flatten_power = power / bkg

    smoothed_flatten_power = conv.convolve(flatten_power, kernel)

    return np.stack([power, flatten_power, smoothed_flatten_power])


def f_process(x):
    """
    Process lightcurve

    x is KIC integer
    """
    kic = int(x)

    # generate power spectra
    data = np.array(klc_h5f_apokasc[f"{kic}"])
    return f_process_internal(data)


def run(func, my_iter):
    # do not increase max_workers even you have more CPU, will have issue with I/O
    with process_pool(max_workers=4) as executor:
        return list(tqdm(executor.map(func, my_iter), total=len(my_iter)))


if __name__ == "__main__":
    # compile all apokasc lightcurves to power spectrum into one file
    results = np.swapaxes(
        np.stack(run(f_process, klc_h5f_apokasc.keys())), 0, 1
    )
    kps_h5f_apokasc = h5py.File(apokasc_kps_h5_path, "w")
    kps_h5f_apokasc.create_dataset("KIC", data=np.array(list(klc_h5f_apokasc.keys()), dtype=int))
    kps_h5f_apokasc.create_dataset("freq_range", data=freq_range)
    kps_h5f_apokasc.create_dataset("powerspec", data=results[0])
    kps_h5f_apokasc.create_dataset("flattened_powerspec", data=results[1])
    kps_h5f_apokasc.create_dataset("smoothed_flattened_powerspec", data=results[2])
    kps_h5f_apokasc.close()
    klc_h5f_apokasc.close()
