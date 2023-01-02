from matplotlib.cbook import file_requires_unicode
import numpy as np
import scipy


zsolar = 0.0152
ysolar = 0.2485


def feh_to_z(feh):
    """
    Convert Fe_H to Z
    """
    zx = 10.0 ** (feh + np.log10(zsolar / (1.0 - ysolar - 2.78 * zsolar)))
    return (zx - ysolar * zx) / (2.78 * zx + 1.0)


def z_to_feh(z):
    """
    Convert Z to Fe_H
    """
    return np.log10(z / (1.0 - ysolar - 2.78 * z)) - np.log10(
        zsolar / (1.0 - ysolar - 2.78 * zsolar)
    )


def rc_loggteffcut(teff, feh):
    """
    For a T_eff and metallicity, where is the logg cut for RGB-RC
    """
    # feh = z_to_feh(z)
    this_teff = (4760.0 - 4607.0) / (-0.4) * feh + 4607.0
    return 0.0018 * (teff - this_teff) + 2.5


def redgiant_sample(allstar):
    """
    Selects red giants spectroscopically from APOGEE sample

    Returns allstar indices
    """
    jk = (allstar["J"] - (allstar["AK_TARG"] * 2.5)) - (
        allstar["K"] - allstar["AK_TARG"]
    )
    z = feh_to_z(allstar["M_H"])
    z[z > 0.024] = 0.024
    logg = allstar["LOGG"]
    rgindx = (
        (jk >= 0.8) | (logg > rc_loggteffcut(allstar["TEFF"], allstar["FE_H"]))
    ) & (allstar["M_H"] > -0.8)
    return rgindx


def salaris_etal_2015(c_n, m_h):
    """
    Salaris et al. 2015 [C/N]-age relation
    """
    c = 34.88
    lin = 156.3 * c_n + 20.52 * m_h
    quad = 298.6 * c_n**2 + 78.16 * c_n * m_h + 7.82 * m_h**2
    cubic = (
        305.9 * c_n**3
        + 85.02 * c_n**2 * m_h
        + 22.93 * c_n * m_h**2
        + 0.8987 * m_h**3
    )
    quartic = (
        141.1 * c_n**4
        + 21.96 * c_n**3 * m_h
        + 16.14 * c_n**2 * m_h**2
        + 1.447 * c_n * m_h**3
    )
    return c + lin + quad + cubic + quartic


def numax_to_deltanu(numax=np.arange(0.1, 500, 100), numax_err=None, pipeline="syd"):
    """
    Table 1 from Huber et al. 2010

    Convert nu_max to delta_nu approximately

    :param numax: numax to be converted
    :type numax: float, numpy.ndarray
    :param numax_err: [optional] numax uncertainty to be converted
    :type numax_err: NoneType, float, numpy.ndarray
    :param pipeline: which pipeline to use
    :type pipeline: str
    """
    uncertainty = False
    numax = np.array(numax)
    if numax_err is not None:
        numax_err = np.array(numax_err)
        uncertainty = True

    if pipeline.lower() == "syd":
        alpha = 0.268
        beta = 0.758
    elif pipeline.lower() == "can":
        alpha = 0.286
        beta = 0.745
    elif pipeline.lower() == "stello2009":
        alpha = 0.263
        beta = 0.772
    else:
        raise ValueError(f"Unknown pipeline={pipeline}")
    deltanu = alpha * (numax**beta)
    if uncertainty:
        deltanu_err = deltanu * np.sqrt((beta * numax_err / numax) ** 2)
        return deltanu, deltanu_err
    else:
        return deltanu


def mass_scaling_relation(
    numax, deltanu, teff, numax_err=None, deltanu_err=None, teff_err=None, pipeline=None
):
    """
    Mass scaling relation
    """
    numax = np.array(numax)
    deltanu = np.array(deltanu)
    teff = np.array(teff)

    if numax_err is not None and deltanu_err is not None and teff_err is not None:
        numax_err = np.array(numax_err)
        deltanu_err = np.array(deltanu_err)
        teff_err = np.array(teff_err)
        uncertainty = True
    elif numax_err is None and deltanu_err is None and teff_err is None:
        uncertainty = False
    else:
        raise ValueError(
            "If you provide parameter uncertainties, you need to provide uncertainties for all three parameters."
        )

    # all units are in uHz
    if pipeline is None:
        numax_sol = 3076.0
        deltanu_sol = 135.146
    elif pipeline.lower() == "a2z":
        numax_sol = 3097.33
        deltanu_sol = 135.2
    elif pipeline.lower() == "can":
        numax_sol = 3140.0
        deltanu_sol = 134.88
    elif pipeline.lower() == "cor":
        numax_sol = 3050.0
        deltanu_sol = 135.5
    elif pipeline.lower() == "oct":
        numax_sol = 3139.0
        deltanu_sol = 135.05
    elif pipeline.lower() == "syd":
        numax_sol = 3090.0
        deltanu_sol = 135.1
    elif pipeline.lower() == "all":
        return np.array(
            [
                mass_scaling_relation(
                    numax,
                    deltanu,
                    teff,
                    numax_err=numax_err,
                    deltanu_err=deltanu_err,
                    teff_err=teff_err,
                    pipeline=pname,
                )
                for pname in ["a2z", "can", "cor", "oct", "syd"]
            ]
        )
    else:
        raise ValueError(f"Unknown pipeline={pipeline}")

    mass = (
        (numax / numax_sol) ** 3
        * (teff / 5780.0) ** 1.5
        * (deltanu / deltanu_sol) ** -4
    )

    if uncertainty:
        mass_err = mass * np.sqrt(
            (3 * numax_err / numax) ** 2
            + (1.5 * teff_err / teff) ** 2
            + (-4 * deltanu_err / deltanu) ** 2
        )
        return mass, mass_err
    else:
        return mass


def rgb_age(M, feh, dr=17):
    """
    RGB age from Mass and Fe_H
    """
    if dr == 14:
        return 10.25 * M**-3.1 - 6.08 * M * feh + 10.85 * feh
    elif dr == 17:
        return 10.67 * M**-3.06 - 6.16 * M * feh + 11.22 * feh
    else:
        raise ValueError("Only APOGEE DR14 and DR17 are supported")


def rc_age(M, feh, dr=17):
    """
    RC age from Mass and Fe_H
    """
    if dr == 14:
        return 6.11 * M**-2.29 - 1.76 * M * feh + 4.04 * feh
    elif dr == 17:
        return 6.29 * M**-2.25 - 1.62 * M * feh + 3.94 * feh
    else:
        raise ValueError("Only APOGEE DR14 and DR17 are supported")


def numax_teff_L_to_mass(numax, L, teff):
    """
    Kjeldsen and Bedding 1995
    """
    return (numax / 3076.0) * L / ((teff / 5780.0)**3.5)


def powerspec_log_rebin(x, y, f=0.02, dx=None):
    """
    Rebinning powerspectrum into logarithmically spaced frequency points
    
    Lower f means higher resolution, a f=0 means no binning
    """
    if dx is None:
        dx = x[1] - x[0]

    x = np.asarray(x)
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise TypeError("Complex y not supported")

    minx = x[0] - dx * 0.5  # frequency to start from
    maxx = x[-1]  # maximum frequency to end
    binx = [minx, minx + dx]  # first

    # until we reach the maximum frequency, increase the width of each frequency bin by f
    while binx[-1] <= maxx:
        binx.append(binx[-1] + dx * (1.0 + f))
        dx = binx[-1] - binx[-2]

    binx = np.asarray(binx)

    # compute the mean of the ys that fall into each new frequency bin.
    # we cast to np.double due to scipy's bad handling of longdoubles
    biny, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), y.astype(np.double),
        statistic="mean", bins=binx)

    return (binx[:-1] + binx[1:]) / 2, biny


def powerspec_log_rebin_v2(x, y):
    """
    Rebinning powerspectrum into logarithmically spaced frequency points
    """
    x = np.asarray(x)
    x_numax = numax_to_deltanu(x)
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise TypeError("Complex y not supported")
    
    # for the first bin, whats the pixel resolution of deltanu
    dx = x[1] - x[0]
    f = x_numax[0] / dx

    minx = x[0] - dx  # frequency to start from
    maxx = x[-1]  # maximum frequency to end
    binx = [minx, minx + dx]  # first

    # until we reach the maximum frequency, increase the width of each frequency bin by f
    while binx[-1] <= maxx:
        binx.append(binx[-1] + numax_to_deltanu(binx[-1]) / f)
        dx = binx[-1] - binx[-2]

    binx = np.asarray(binx)

    # compute the mean of the ys that fall into each new frequency bin.
    # we cast to np.double due to scipy's bad handling of longdoubles
    biny, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), y.astype(np.double),
        statistic="mean", bins=binx)

    return (binx[:-1] + binx[1:]) / 2, biny


def gaussian_envelope_weights(nu_max, freq):
    """
    Gaussian shaped envelope pixel level weighting
    """
    width = 2. * numax_to_deltanu(nu_max)
    gaussian = np.exp(-(((freq - nu_max) / width) ** 2) / 2) / (width * np.sqrt(2*np.pi))
    # we want heigh as 2. (so 1 before adding 1.)
    factor = 1.0 / np.max(gaussian)
    return 1. + factor * gaussian


def ES_num_to_word(es):
    """
    Function to turn Evo State flag to words
    """
    if es == 1:
        return "RGB"
    elif es == 2:
        return "RC"
    else:
        return "Unidentified"