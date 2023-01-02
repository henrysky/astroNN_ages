Abstract
===========

To be written

Getting Started
================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper, althought some data used are third party proprietary data.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_ages/tree/master/

This project use `lightkurve`_ to manage `Kepler` and `TESS` data

.. _lightkurve: https://github.com/lightkurve/lightkurve

Some notebooks make use of `milkyway_plot`_ to plot on milkyway and `gaia_tools`_ to do query.

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot
.. _gaia_tools: https://github.com/jobovy/gaia_tools

To continuum normalize all APOGEE DR17 spectra, refers to:
https://github.com/henrysky/astroNN_APOGEE_VAC/blob/e6084eb9c8599cab9c1153b127124decbed744f1/1_continuum_norm.py

To continuum normalize arbitrary APOGEE spectrum, see:
http://astronn.readthedocs.io/en/latest/tools_apogee.html#pseudo-continuum-normalization-of-apogee-spectra

Data Files
------------

Please refer to `here`_ for the list of public data files we used in the paper

.. _here: data_files/README.rst

Python Scripts
------------------

Please refer to `here`_ for the list of scripts. Execute any of these python script at the root directory of this repository.

.. _here: py/README.rst

Jupyter Notebook
------------------

-   | `Datasets_Data_Reduction.ipynb`_
    | The notebook is used to reduce data
-   | `Training_AE_PSD.ipynb`_
    | The notebook is used to train our Encoder-Decoder model
-   | `Testing_AE_PSD.ipynb`_
    | The notebook is used to train our Encoder-Decoder model
-   | `Testing_Age_Spatial_Dist.ipynb`_
    | The notebook is used to plot age spatial distribution

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Training_AE_PSD.ipynb: Training_AE_PSD.ipynb
.. _Testing_AE_PSD.ipynb: Testing_AE_PSD.ipynb
.. _Testing_Age_Spatial_Dist.ipynb: Testing_Age_Spatial_Dist.ipynb

Data Product
--------------

-   | `nn_latent_age_dr17.csv`_
    | Data file containing latent space age from this paper, row matched to APOGEE DR17 ``allStar-dr17-synspec_rev1.fits``

.. _nn_latent_age_dr17.csv: nn_latent_age_dr17.csv

If you need ``allStar-dr17-synspec_rev1.fits``, the direct link is https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits

.. code-block:: python

    import pandas as pd

    latent_space_age_file = pd.read_csv("./nn_latent_age_dr17.csv")

    # 733901 rows
    print(len(latent_space_age_file))


Using the Neural Net model on APOGEE spectra
-----------------------------------------------

You can refer to the following code to use our trained models on arbitrary APOGEE spectra to get PSD reconstruction, latent space vector and age. 
You need to use use the following code at the root directory of this repository.

.. code-block:: python

    import numpy as np

    from astropy.io import fits
    from astroNN.apogee import visit_spectra, apogee_continuum
    from astroNN.models import load_folder
    from py.ensemble import ProbabilisticRandomForestRegressor

    # load the trained encoder-decoder model with astroNN
    neuralnet = load_folder("./models/astroNN_VEncoderDecoder")

    # frequency solution for the PSD reconstruction of the model
    freq_solution = np.genfromtxt("./models/astroNN_VEncoderDecoder/freq_solution.csv")

    # load latent space age model
    latent_age_model = ProbabilisticRandomForestRegressor.load_model("./models/astroNN_VEncoderDecoder/latent_age_model")

    # arbitrary spectrum
    f = fits.open(visit_spectra(dr=17, apogee="2M19060637+4717296"))
    spectrum = f[1].data[0]
    spectrum_err = f[2].data[0]
    spectrum_bitmask = f[3].data[0]

    # using default continuum and bitmask values to continuum normalize
    norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                bitmask=spectrum_bitmask, dr=17)

    # take care of extreme value
    norm_spec[norm_spec>2.] = 1.

    # PSD reconstruction for the spectra
    psd_reconstruction = np.exp(neuralnet.predict(norm_spec)[0])

    # sampled latent space representation of the APOGEE spectrum
    z = neuralnet.predict_encoder(norm_spec)[0]

    # PSD prediction from latent space
    psd_from_z = np.exp(neuralnet.predict_decoder(z)[0])

    # stack latent space representation, ASPCAP DR17 TEFF, ASPCAP DR17 [FE/H] to get latent space age
    # I got the TEFF, [FE/H] from allstar file
    stacked_z = np.hstack([z, [[4698.3677]], [[0.050341]]])

    # predict with the trained random forest model, getting posterior
    age_posterior = 10**latent_age_model.predict(stacked_z)

    # getting final prediction and uncertainty in Gyr
    age, age_error = np.mean(age_posterior), np.std(age_posterior)

Reconstruction of random samples in latent space
----------------------------------------------------

Since we are using a variational encoder-decoder, you can easily draw random samples from latent space and get their reconstruction.
Here is an example:

.. code-block:: python

    import numpy as np
    import pylab as plt

    from astropy.io import fits
    from astroNN.apogee import visit_spectra, apogee_continuum
    from astroNN.models import load_folder
    from py.ensemble import ProbabilisticRandomForestRegressor

    # load the trained encoder-decoder model with astroNN
    neuralnet = load_folder("./models/astroNN_VEncoderDecoder")

    # frequency solution for the PSD reconstruction of the model
    freq_solution = np.genfromtxt("./models/astroNN_VEncoderDecoder/freq_solution.csv")

    latent_dim = neuralnet.latent_dim
    num_samples = 2  # set the number of sample you want to get

    # random sample
    random_z_sample = np.random.normal(0, 1, (num_samples, latent_dim))

    # this is the reconstruction
    psd_from_z = np.exp(neuralnet.predict_decoder(random_z_sample).T[0])

    plt.figure()
    plt.plot(freq_solution, psd_from_z)
    plt.xlabel("Freq (uHz)")
    plt.ylabel("PSD")
    plt.xscale("log")

Contact
===========

-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

.. _henrysky: https://github.com/henrysky

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
