import concurrent.futures

import os
import time
import numpy as np
from tqdm import tqdm
import lightkurve as lk
from astropy.io import fits, ascii

import sys

sys.path.insert(0, os.path.dirname(__file__))

from __init__ import yu_etal_2018_path, apokasc2_path

apokasc_f = ascii.read(apokasc2_path)
yu_etal_f = ascii.read(yu_etal_2018_path)

thread_or_process = concurrent.futures.ThreadPoolExecutor


def f(x, pbar):
    lk.search_lightcurve(
        f"KIC {x}", cadence="long", author=["Kepler", "QLP"]
    ).download_all()
    pbar.update(1)


def run(f, my_iter, pbar):
    # do not increase max_workers even you have more CPU, will have issue with I/O
    with thread_or_process(max_workers=8) as executor:
        # results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
        [executor.submit(f, x, pbar) for x in my_iter]


if __name__ == "__main__":
    total_num = len(np.unique(np.concatenate([apokasc_f["KIC"], yu_etal_f["KIC"]])))
    # prevent getting blocked from SIMBAD for making too many query
    with tqdm(total=total_num, unit="stars") as pbar:
        # download APOKASC2 first
        run(f, np.array(apokasc_f["KIC"]), pbar)

        # and others
        extra_kic = np.setdiff1d(yu_etal_f["KIC"], apokasc_f["KIC"])
        run(f, np.array(extra_kic), pbar)
