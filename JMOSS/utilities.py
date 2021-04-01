"""
JMOSS Air Data Utilities

Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.

        Erb, Russell E. "Pitot-Statics Textbook." US Air Force Test Pilot School, Edwards AFB, CA (2020).
"""
from numpy import zeros, sqrt, exp, cumsum, hstack, diff, array, log
from scipy import optimize


def mach_from_qc_pa(qc_over_pa: array):
    # Mach number from qc/Pa pressure ratio
    mach = zeros(qc_over_pa.shape[0])
    high = qc_over_pa > 0.89293
    mach[~high] = sqrt(5 * ((abs(qc_over_pa[~high]) + 1) ** (2 / 7) - 1))
    for index, value in enumerate(high):
        if value:
            sol = optimize.root_scalar(lambda m: m - 0.881284 * sqrt((qc_over_pa[index] + 1)
                                                                     * (1 - 1 / (7 * m ** 2)) ** 2.5),
                                       method='brentq', bracket=[1.0, 3.0])
            mach[index] = sol.root
    return mach


def qc_pa_from_mach(mach: array):
    # qc/Pa pressure ratio from Mach number
    qc_pa = zeros(mach.shape[0])
    high = mach > 1
    qc_pa[~high] = ((1 + 0.2 * (mach[~high] ** 2)) ** (7 / 2)) - 1
    qc_pa[high] = (166.921 * mach[high] ** 7) / ((7 * mach[high] ** 2 - 1) ** 2.5) - 1
    return qc_pa


def delta_from_press_alt(press_alt: array):
    # Pressure ratio from pressure altitude [ft]
    delta = zeros(press_alt.shape[0])
    high = press_alt > 36089.24
    delta[high] = 0.223360 * exp(-4.80637e-5 * (press_alt[high] - 36089.24))
    delta[~high] = (1 - 6.87559e-6 * press_alt[~high]) ** 5.2559
    return delta


def press_alt_from_delta(delta: array):
    # Pressure altitude [ft] from pressure ratio
    press_alt = zeros(delta.shape[0])
    high = delta < 0.223359324957103
    press_alt[~high] = 1.454420638810633e5 * (1 - (delta[~high]) ** (1 / 5.2559))
    press_alt[high] = -2.080572240589052e4 * log(delta[high] / 0.223359324957103) + 36089.24
    return press_alt


def theta_from_press_alt(press_alt: array):
    # Temperature ratio from press_alt [ft]
    theta = zeros(press_alt.shape[0])
    high = press_alt > 36089.24
    theta[high] = 0.751865
    theta[~high] = 1 - 6.87559e-6 * press_alt[~high]
    return theta


def airspeed_from_qc_pa_delta(qc_over_pa: array, delta: array):
    #  Airspeed [kts] from qc/Pa and pressure ratio
    qc_psl = qc_over_pa * delta
    airspeed = airspeed_from_qc_psl(qc_psl)
    return airspeed


def airspeed_from_qc_psl(qc_over_psl: array):
    # Airspeed [kts] from qc/Psl
    c1 = 5.829507067779927e2
    c2 = 661.478827231622
    high = qc_over_psl > 0.89293
    airspeed = zeros(qc_over_psl.shape[0])
    airspeed[~high] = c2 * sqrt(5 * ((qc_over_psl[~high] + 1) ** (2 / 7) - 1))
    for index, value in enumerate(high):
        if value:
            sol = optimize.newton(lambda v: v - c1 * sqrt((qc_over_psl[index] + 1)
                                                               * (1 - (1 / (7 * (v / c2) ** 2))) ** 2.5), x0=c2)
            airspeed[index] = sol
    return airspeed


def iterate_pa_oat(height: array, tot_pres: array, tat: array, pa_bias: array, eta: array):
    pres_alt = height + pa_bias
    amb_pres = 14.6960 * delta_from_press_alt(pres_alt)
    mach_pc = mach_from_qc_pa((tot_pres - amb_pres) / amb_pres)
    oat_from_tat = (tat / (1 + 0.2 * eta * mach_pc ** 2))
    oat_from_atm = 288.15 * theta_from_press_alt(pres_alt)
    bias = oat_from_tat.mean() - oat_from_atm.mean()
    oat = oat_from_atm + bias

    delta_pres_alt = 1000
    delta_oat = 1000
    while (delta_pres_alt > 1e-3) or (delta_oat > 1e-3):
        temp_std = 288.15 * theta_from_press_alt(pres_alt)
        delta = temp_std / oat
        new_press_alt = cumsum(hstack([height[0], delta[1:] * diff(height)])) + pa_bias
        delta_pres_alt = sum((pres_alt - new_press_alt) ** 2)
        pres_alt = new_press_alt

        amb_pres = 14.6960 * delta_from_press_alt(pres_alt)
        mach_pc = mach_from_qc_pa((tot_pres - amb_pres) / amb_pres)
        oat_from_tat = (tat / (1 + 0.2 * eta * mach_pc ** 2))
        oat_from_atm = 288.15 * theta_from_press_alt(pres_alt)
        bias = oat_from_tat.mean() - oat_from_atm.mean()
        new_oat = oat_from_atm + bias
        delta_oat = sum((oat - new_oat) ** 2)
        oat = new_oat
    return amb_pres, oat, mach_pc


def translate_spe_to_errors(spe_ratio: array, mach_ic: array, target_alt_ic: float = None):
    # Translate an array of spe ratio and instrument corrected mach number to  dHpc, dVpc, dMpc at a specified target
    # instrument corrected altitude
    if target_alt_ic is None:
        target_alt_ic = array([0])
    else:
        target_alt_ic = array([target_alt_ic])
    sea_level_pres = 14.6960
    delta_ic = delta_from_press_alt(target_alt_ic)
    stat_pres = sea_level_pres * delta_ic
    amb_pres = stat_pres * (1 - spe_ratio)
    delta = amb_pres / sea_level_pres
    pres_alt = press_alt_from_delta(delta)
    qcic_ps = qc_pa_from_mach(mach_ic)
    ind_spd = airspeed_from_qc_pa_delta(qcic_ps, delta_ic)
    qc_pa = (qcic_ps + 1) / (1 - spe_ratio) - 1
    cal_spd = airspeed_from_qc_pa_delta(qc_pa, delta)
    mach_pc = mach_from_qc_pa(qc_pa)
    alt_corr = pres_alt - target_alt_ic
    asp_corr = cal_spd - ind_spd
    mach_corr = mach_pc - mach_ic
    return alt_corr, asp_corr, mach_corr
