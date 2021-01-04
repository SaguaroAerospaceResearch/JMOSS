"""
JMOSS Air Data Utilities

Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.

        Erb, Russell E. "Pitot-Statics Textbook." US Air Force Test Pilot School, Edwards AFB, CA (2020).
"""
from numpy import zeros, sqrt
from scipy import optimize


def mach_from_qc_pa(qc_over_pa):
    mach = zeros(qc_over_pa.shape)
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
