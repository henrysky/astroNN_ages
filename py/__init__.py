import os
import pathlib

from astroNN.apogee import allstar

# lightkurve cache directory
_LK_CAHCE = os.path.expanduser("~/.lightkurve-cache")

# absolute path to the parent directory
abs_parent_path = os.path.abspath(
    os.path.join(pathlib.Path(__file__).parent.resolve(), "..")
)

# your lcoation of apogeee continuum spectra, apogee allstar and apogee gaia xmatched file, all need to be row-matched
apogee_contspec_path = os.path.abspath(
    os.path.join("D://astro_data_cache//contspec_dr17_synspec.fits")
)
apogee_allstar_path = allstar(dr=17)
apogee_gaia_path = "D://apogeework//apogeedr17_syncspec_gaiadr3_xmatch.fits"

# ======================= APOKASC related =======================
# absolute path to the data directory
data_parent_path = os.path.join(abs_parent_path, "data_files")

kic_2mass_xmatch_path = os.path.join(data_parent_path, "KIC_2MASS_xmatch.csv")
kic_tic_xmatch_path = os.path.join(data_parent_path, "KIC_TIC_xmatch.csv")

apokasc2_path = os.path.join(data_parent_path, "APOKASC_2_PaperTable.txt")
apokasc2_h5_path = os.path.join(data_parent_path, "APOKASC2.h5")
# APOKASC2 results from individual pieline
apokasc2_pipeline_path = os.path.join(
    data_parent_path, "APOKASC_2_Pipeline_Table_6.txt"
)
apokasc3p_path = os.path.join(data_parent_path, "APOKASC_cat_v6.7.4.fits")
montalban_path = os.path.join(
    data_parent_path, "kepler_low_metallicity_with_samples.fits"
)
miglio2021_path = os.path.join(data_parent_path, "DR17_ALL_TED_APOGEE_DR17.txt")

apokasc_datafile_path = os.path.join(data_parent_path, "APOKASC.fits")
apokasc_klc_h5_path = os.path.join(data_parent_path, "APOKASC_kepler_lightcurves.h5")
apokasc_kps_h5_path = os.path.join(data_parent_path, "APOKASC_kepler_powerspec.h5")
apokasc_acf_h5_path = os.path.join(data_parent_path, "APOKASC_kepler_acf.h5")

tess_tlc_h5_path = os.path.join(data_parent_path, "TESS_lightcurves.h5")

yu_etal_2018_path = os.path.join(data_parent_path, "yu_etal_2018_apogeedr17.dat")
# ======================= APOKASC related =======================

# ======================= TESS CVZ related =======================
tess_cvz_2019_path = os.path.join(data_parent_path, "tess_cvz_master_all_121219.fits")
