"""
JMOSS Python demo for public release

Jurado-McGehee Online Self Survey (JMOSS)
Version 6.0
January 2021
Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.
"""
from JMOSS.estimation import JmossEstimator
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from os import listdir
from os.path import join

# TODO: Visualization class
# TODO: Ingest weather balloon
# TODO: Ingest TFB
# TODO: Model fitting / selection

if __name__ == '__main__':
    # Provide the mapping between JMOSS expected parameters and your specific DAS parameters
    parameter_names = {'time': 'time_s',
                       'north velocity': 'north_vel_fps',
                       'east velocity': 'east_vel_fps',
                       'down velocity': 'down_vel_fps',
                       'geometric height': 'height_ft',
                       'total pressure': 'total_pres_psi',
                       'static pressure': 'static_pres_psi',
                       'total temperature': 'total_temp_K',
                       'angle of attack': 'aoa_rad',
                       'angle of slideslip': 'aos_rad',
                       'roll angle': 'roll_angle_rad',
                       'pitch angle': 'pitch_angle_rad',
                       'true heading': 'true_heading_rad'}

    # Initialize JMOSS estimator
    estimator = JmossEstimator(parameter_names)

    # Load test points into estimator
    data_dir = 'sample_data'
    data_files = [join(data_dir, name) for name in listdir(data_dir) if 'CLASS' in name]
    for filename in data_files:
        estimator.add_test_point(filename)

    # To process all test points, use 'process_test_points()'
    # To process a list of test points use 'process_test_points(list)'
    estimator.process_test_points()

    # To get the results of all points, use 'get_spe_results()'
    # To get the results of a list of test points, use 'get_spe_results(list)'
    results = estimator.get_spe_results()

    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)) # noqa
    colors = cm.get_cmap('tab20').colors
    labels = estimator.test_point_names_list
    for index, point in enumerate(results):
        ax1.plot(point.mach_ic, point.spe_ratio, color=colors[index], linestyle='-', linewidth=2, label=labels[index])
        ax1.plot(point.mach_ic, point.inferences['spe ratio'], color=colors[index], linestyle='--', linewidth=2)
        ax2.plot(point.mach_ic, point.oat, color=colors[index], linestyle='-', linewidth=2, label=labels[index])
        ax2.plot(point.mach_ic, point.inferences['oat'], color=colors[index], linestyle='--', linewidth=2)
    ax1.legend()
    ax1.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
    ax1.set_ylabel("SPE ratio, $\Delta P_p / P_s$")
    ax1.set_title("JMOSS: Individual Test Point Results", weight='bold')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.grid(which='minor', alpha=0.2, linestyle=":")
    ax1.grid(which='major', alpha=0.2, linestyle=":")
    ax1.minorticks_on()

    ax2.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
    ax2.set_ylabel("Ambient temperature, $T_a$ [K]")
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.grid(which='minor', alpha=0.2, linestyle=":")
    ax2.grid(which='major', alpha=0.2, linestyle=":")
    ax2.minorticks_on()

    # Print wind and eta results
    wind_labels = ['north wind', 'east wind', 'down wind']
    for index, point in enumerate(results):
        print('\n' + labels[index] + ':')
        est = point.eta
        ci = point.inferences['eta']
        print('eta: %0.2f \u00B1 %0.2f' % (est, ci))
        for dim, label in enumerate(wind_labels):
            est = point.wind[dim]
            ci = point.inferences[label]
            print('%s: %0.2f \u00B1 %0.2f' % (label, est, ci))

    fig.savefig("jmoss.pdf", dpi=fig.dpi, edgecolor='w', format='pdf', transparent=True, pad_inches=0.1, bbox_inches='tight')