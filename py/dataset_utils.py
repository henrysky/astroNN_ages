import os
import sys
import h5py
import numpy as np
from astroNN.apogee import apogee_astronn
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, os.path.dirname(__file__))


from __init__ import (
    apokasc2_h5_path,
    apokasc_datafile_path,
    apokasc_klc_h5_path,
    apokasc_kps_h5_path,
    apogee_contspec_path,
    apogee_allstar_path,
    apogee_gaia_path,
    miglio2021_path,
    yu_etal_2018_path,
)
from .utils import numax_to_deltanu


def stack_label(*arg):
    label_atleast2d = lambda y: np.reshape(y, (-1, 1)) if y.ndim == 1 else y
    return np.hstack([label_atleast2d(i) for i in arg])


class APOKASC:
    """
    Class for handling different APOKASC dataset
    """

    def __init__(
        self,
        dataset_path=apokasc_datafile_path,
        yu_etal_2018_path = yu_etal_2018_path,
        lc_path=apokasc_klc_h5_path,
        ps_path=apokasc_kps_h5_path,
        contspec_path=apogee_contspec_path,
        good_idx=True,
    ):
        """
        :param good_idx: if true then do basic filtering of good stars, can be a list of indices, or False doing nothing
        :type good_idx: bool, list
        """
        self.dataset_path = dataset_path
        self.yu_etal_2018_path = yu_etal_2018_path
        self.lc_path = lc_path
        self.ps_path = ps_path
        self.contspec_path = contspec_path
        
        self.apokasc2_f = h5py.File(apokasc2_h5_path, mode="r")
        self.yu_2018_f = Table.read(self.yu_etal_2018_path, format="ascii")
        # only use the ones also appeared in allstar dr17
        self.yu_2018_f = self.yu_2018_f[~np.isnan(self.yu_2018_f["allstar_dr17_idx"])]

        # combine both
        all_kic = np.union1d(self.yu_2018_f["KIC"], self.apokasc2_f["KIC"])
        all_kic = np.sort(all_kic.astype(str)).astype(int)  # sort in str, older way of soring
        numax = np.ones_like(all_kic) * -9999.
        numax_err = np.ones_like(all_kic) * -9999.
        dnu = np.ones_like(all_kic) * -9999.
        dnu_err = np.ones_like(all_kic) * -9999.
        evostate = np.ones_like(all_kic) * -9999.
        from_yu = np.zeros_like(all_kic, dtype=bool)
        allstar_dr17_idx = np.ones_like(all_kic, dtype=int) * -9999
        _, idx1, idx2 = np.intersect1d(all_kic, self.yu_2018_f["KIC"], return_indices=True)
        numax[idx1] = self.yu_2018_f["numax"][idx2]
        numax_err[idx1] = self.yu_2018_f["e_numax"][idx2]
        dnu[idx1] = self.yu_2018_f["Delnu"][idx2]
        dnu_err[idx1] = self.yu_2018_f["e_Delnu"][idx2]
        evostate[idx1] = self.yu_2018_f["Phase"][idx2]
        allstar_dr17_idx[idx1] = self.yu_2018_f["allstar_dr17_idx"][idx2]
        _, idx1, idx2 = np.intersect1d(all_kic, self.apokasc2_f["KIC"], return_indices=True)
        numax[idx1] = self.apokasc2_f["Numax"][idx2]
        numax_err[idx1] = self.apokasc2_f["Numax_err"][idx2]
        dnu[idx1] = self.apokasc2_f["Deltanu"][idx2]
        dnu_err[idx1] = self.apokasc2_f["Deltanu_err"][idx2]
        evostate[idx1] = self.apokasc2_f["ES"][idx2]
        allstar_dr17_idx[idx1] = self.apokasc2_f["allstar_dr17_idx"][idx2]
        from_yu[idx1] = True
        self.dataset = Table([all_kic, numax, numax_err, dnu, dnu_err, evostate, allstar_dr17_idx, from_yu], 
                             names=["KIC", "Numax", "Numax_error", "Dnu", "Dnu_error", "Evostate", "allstar_dr17_idx", "from_yu"], 
                             dtype=[int, float, float, float, float, int, int, bool])
        
        self.allstar = Table.read(apogee_allstar_path, format="fits", hdu=1)[
            self.dataset["allstar_dr17_idx"]
        ]
        with fits.open(contspec_path) as f:
            apogee_spec_availability = f[1].data[self.dataset["allstar_dr17_idx"]]
        if good_idx is True:
            self.good_idx = (
                (self.dataset["Numax"] > 4.0)
                & (self.dataset["Numax"] < 250.0)
                & (self.dataset["Numax_error"] / self.dataset["Numax"] < 0.1)
                # as some seems very far off (e.g. KIC 10001284), evo state seems to be a good indicator
                # we use -1 for bad evostate, 1=RGB, 2=RC
                & (self.dataset["Evostate"] > 0)
                & (~np.isnan(self.allstar["TEFF"]) & ~np.isnan(self.allstar["FE_H"]))
                & (apogee_spec_availability == 1)
            )
        elif good_idx is False:
            self.good_idx = np.arange(len(self.dataset))
        else:
            self.good_idx = good_idx
            
        # self.dataset is the main one
        self.dataset = self.dataset[self.good_idx]
        self.allstar = self.allstar[self.good_idx]

        self.astronn_allstar = Table.read(apogee_astronn(dr=17), format="fits")[
            self.dataset["allstar_dr17_idx"]
        ]
        try:  # dont really need gaia data per say
            self.gaia = Table.read(apogee_gaia_path, format="fits")[
                self.dataset["allstar_dr17_idx"]
            ]
            # get uncertainty in bp_rp, useful if using pbjam
            # https://dc.zah.uni-heidelberg.de/tableinfo/gaia.dr2epochflux
            self.gaia["bp_rp_error"] = np.sqrt(
                (
                    2.5
                    * (1 / np.log(10))
                    * (
                        self.gaia["phot_bp_mean_flux_error"][1:10]
                        / self.gaia["phot_bp_mean_flux"][1:10]
                    )
                )
                ** 2
                + (
                    2.5
                    * (1 / np.log(10))
                    * (
                        self.gaia["phot_rp_mean_flux_error"][1:10]
                        / self.gaia["phot_rp_mean_flux"][1:10]
                    )
                    ** 2
                )
            )
        except:
            pass
        self.lc_f = h5py.File(self.lc_path, "r")
        self.ps_f = h5py.File(self.ps_path, "r")
        self.contspec = fits.getdata(self.contspec_path)[
            self.dataset["allstar_dr17_idx"]
        ]
        
        # add excess delta_nu
        deltanu_pred = numax_to_deltanu(self.dataset["Numax"][()], pipeline="syd")
        self.dataset["EXCESS_DELTANU"] = (self.dataset["Dnu"] / deltanu_pred)
        
        # ======================= add Miglio et al ======================= #
        # I have checked all stars in miglio also in yu 2018
        miglio_age_f = Table.read(miglio2021_path, format="ascii")
        _, idx1, idx2 = np.intersect1d(miglio_age_f["KIC"], self.dataset["KIC"], return_indices=True)
        
        self.dataset["Miglio_Age"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_LogAge"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_Age_Error"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_LogAge_Error"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_Mass"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_Mass_Error"] = np.ones_like(self.dataset["Numax"]) * -9999.
        self.dataset["Miglio_evstate"] = np.ones_like(self.dataset["Numax"], dtype=int)

        self.dataset["Miglio_Age"][idx2] = miglio_age_f["age"][idx1].data
        self.dataset["Miglio_LogAge"][idx2] = np.log10(miglio_age_f["age"][idx1]).data
        self.dataset["Miglio_Age_Error"][idx2] = miglio_age_f["age_68U"][idx1] - miglio_age_f["age"][idx1].data
        self.dataset["Miglio_LogAge_Error"][idx2] = (np.log10(miglio_age_f["age_68U"][idx1]) - np.log10(miglio_age_f["age_68L"][idx1])).data / 2
        self.dataset["Miglio_Mass"][idx2] = miglio_age_f["mass"][idx1].data
        self.dataset["Miglio_Mass_Error"][idx2] = miglio_age_f["mass_68U"][idx1] - miglio_age_f["mass"][idx1].data
        self.dataset["Miglio_evstate"][idx2] = miglio_age_f["evstate"][idx1].data
        # ======================= end Miglio et al ======================= #

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, idx):
    #     return APOKASC(
    #         dataset_path=self.dataset_path,
    #         lc_path=self.lc_path,
    #         ps_path=self.ps_path,
    #         contspec_path=self.contspec_path,
    #         good_idx=self.good_idx[idx],
    #     )

    def get_lightcurve(self):
        """
        Function to get lightcurve
        """
        return np.stack([self.lc_f[f"{i}"] for i in self.dataset["KIC"]])

    def get_contspec(self):
        """
        Function to get APOGEE spectra
        """
        return self.contspec

    def get_powerspec_freqrange(self):
        """
        Function to get frequency range for the PSD
        """
        return self.ps_f["freq_range"]
    
    def get_smoothed_flattened_powerspec(self):
        """
        Function to get flattened PSD
        """
        return np.stack(
            [
                self.ps_f["smoothed_flattened_powerspec"][
                    np.where(self.ps_f["KIC"][()] == i)[0][0]
                ]
                for i in np.array(self.dataset["KIC"], dtype=int)
            ]
        )

    # def lookup(self, identifier, table="apokasc"):
    #     if table.lower() == "apokasc":
    #         table_f = self.apokasc_f
    #     elif table.lower() == "astronn":
    #         table_f = self.astronn_allstar
    #     elif table.lower() == "allstar":
    #         table_f = self.allstar
    #     else:
    #         raise ValueError(f"Unkown table: {table}")

    #     if "KIC" in identifier:
    #         return table_f[
    #             self.apokasc_f["KEPLER_ID"]
    #             == str("".join(filter(str.isdigit, identifier)))
    #         ]
    #     elif "TIC" in identifier:
    #         return table_f[
    #             self.apokasc_f["TIC"] == str("".join(filter(str.isdigit, identifier)))
    #         ]
    #     elif "2M" in identifier:
    #         return table_f[
    #             self.apokasc_f["2MASS_ID"]
    #             == str("".join(filter(str.isdigit, identifier)))
    #         ]
    #     else:
    #         raise ValueError("Don't understand what is going on")
