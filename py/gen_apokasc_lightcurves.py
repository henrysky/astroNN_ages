import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import lightkurve as lk
import concurrent.futures
from astropy.io import fits, ascii

import sys

sys.path.insert(0, os.path.dirname(__file__))

from __init__ import (
    _LK_CAHCE,
    apokasc2_path,
    yu_etal_2018_path,
    apokasc_klc_h5_path
)

process_pool = concurrent.futures.ProcessPoolExecutor


def f_process(x):
    """
    Process lightcurve

    x is KIC integer
    """
    kic = int(x)
    _kepler_cache = os.path.join(_LK_CAHCE, "mastDownload", "Kepler")
    folder_name = glob.glob(f"{_kepler_cache}/kplr{kic:09d}_lc*")
    if len(folder_name) < 1:
        # try to download again
        lk.search_lightcurve(
            f"KIC {x}", cadence="long", author=["Kepler"]
        ).download_all()
        folder_name = glob.glob(f"{_kepler_cache}/kplr{kic:09d}_lc*")
        if len(folder_name) < 1:  # if still failing then raise error
            raise FileNotFoundError(f"Data for KIC{kic} is not avaliable locally")
    target_path = os.path.join(_kepler_cache, folder_name[0])
    file_list = glob.glob(f"{target_path}/kplr*llc.fits")
    if len(file_list) < 1:
        raise FileNotFoundError(f"Data for KIC{kic} is not avaliable locally")
    lc = [lk.search.read(f_path) for f_path in file_list]
    lkc = lk.LightCurveCollection(lc)
    kic_lc_clean = lkc.stitch(
        corrector_func=lambda x: x.remove_nans().remove_outliers().normalize(unit="ppm")
    )

    return np.stack([kic_lc_clean.flux.value, kic_lc_clean.time.value])


def run(func, my_iter):
    # do not increase max_workers even you have more CPU, will have issue with I/O
    with process_pool(max_workers=8) as executor:
        return list(tqdm(executor.map(func, my_iter), total=len(my_iter)))


if __name__ == "__main__":
    # compile all apokasc lightcurves into one file
    apokasc_f = ascii.read(apokasc2_path)
    yu_etal_f = ascii.read(yu_etal_2018_path)
    all_kic = np.union1d(apokasc_f["KIC"], yu_etal_f["KIC"])
    klc_h5f_apokasc = h5py.File(apokasc_klc_h5_path, "w")
    results = run(f_process, np.array(all_kic))
    for kic, lc in zip(all_kic, results):
        klc_h5f_apokasc.create_dataset(f"{kic}", data=lc)
    klc_h5f_apokasc.close()
