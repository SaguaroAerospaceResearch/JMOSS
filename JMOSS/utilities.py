"""
JMOSS Air Data Utilities

Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.

        Erb, Russell E. "Pitot-Statics Textbook." US Air Force Test Pilot School, Edwards AFB, CA (2020).
"""
from numpy import zeros, sqrt, exp, cumsum, hstack, diff
from scipy import optimize


def mach_from_qc_pa(qc_over_pa):
    mach = zeros(qc_over_pa.shape[0])
    for index, value in enumerate(qc_over_pa):
        if abs(value) > 0.89293:
            sol = optimize.root_scalar(lambda m: m - 0.881284 * sqrt((abs(value) + 1) *
                                                                     (1 - 1 / (7 * m ** 2)) ** 2.5),
                                       bracket=[0.8, 3], method='brentq')
            this_mach = sol.root
        else:
            this_mach = sqrt(5 * ((abs(value) + 1) ** (2 / 7) - 1))
        mach[index] = this_mach
    return mach


def delta_from_press_alt(press_alt):
    # Pressure ratio from press_alt [ft]
    delta = zeros(press_alt.shape[0])
    high = press_alt > 36089.24
    delta[high] = 0.223360 * exp(-4.80637e-5 * (press_alt[high] - 36089.24))
    delta[~high] = (1 - 6.87559e-6 * press_alt[~high]) ** 5.2559
    return delta


def theta_from_press_alt(press_alt):
    # Temperature ratio from press_alt [ft]
    theta = zeros(press_alt.shape[0])
    high = press_alt > 36089.24
    theta[high] = 0.751865
    theta[~high] = 1 - 6.87559e-6 * press_alt[~high]
    return theta


def iterate_pa_oat(height, tot_pres, tat, pa_bias, eta):
    pres_alt = height + pa_bias
    amb_pres = 14.6960 * delta_from_press_alt(pres_alt)
    mach_pc = mach_from_qc_pa((tot_pres - amb_pres) / amb_pres)
    mean_oat = (tat / (1 + 0.2 * eta * mach_pc ** 2)).mean()
    temp_std = 288.15 * theta_from_press_alt(pres_alt)
    oat = temp_std + (mean_oat - temp_std.mean())

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
        mean_oat = (tat / (1 + 0.2 * eta * mach_pc ** 2)).mean()
        new_oat = temp_std + (mean_oat - temp_std.mean())
        delta_oat = sum((oat - new_oat) ** 2)
        oat = new_oat
    return amb_pres, oat, mach_pc
