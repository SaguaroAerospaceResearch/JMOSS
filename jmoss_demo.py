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
from os import listdir
from os.path import join

# TODO: Visualize
# TODO: Ingest weather balloon
# TODO: Ingest TFB
# TODO: Model selection

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
    estimator.process_test_points(['CLASS65_12SEP_SUPER', 'CLASS95_12SEP_SUPER'])

    # To get the results of all points, use 'get_spe_results()'
    # To get the results of a list of test points, use 'get_spe_results(list)'
    results = estimator.get_spe_results(['CLASS65_12SEP_SUPER', 'CLASS95_12SEP_SUPER'])

    # Visualize the results
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b']
    for index, point in enumerate(results):
        ax.plot(point.mach_ic, point.spe_ratio, color=colors[index], linestyle='-', linewidth=2, label='mean')
        ax.plot(point.mach_ic, point.inferences['spe ratio'], color=colors[index], linestyle='--', linewidth=2, label='95% C.I')
    plt.legend()
    ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
    ax.set_ylabel("SPE ratio, $\Delta P_p / P_s$")
    ax.set_title("Example JMOSS Python output with confidence intervals", weight='bold')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(which='minor', alpha=0.2, linestyle=":")
    ax.grid(which='major', alpha=0.2, linestyle=":")
    ax.minorticks_on()
