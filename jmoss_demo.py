"""
JMOSS Python demo for public release

Jurado-McGehee Online Self Survey (JMOSS)
Version 6.0
April 2021
Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.
"""
from os import listdir
from os.path import join

from JMOSS.estimation import JmossEstimator
from JMOSS.visualization import JmossVisualizer

# TODO: use_aoa option in plotting
# TODO: Model fitting / selection
# TODO: Ingest weather balloon
# TODO: Ingest TFB

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
    # Default alpha for inferences is 0.05, use alpha=x to set a different alpha
    estimator = JmossEstimator(parameter_names, alpha=0.01)

    # Load test points into estimator
    data_dir = 'sample_data'
    data_files = [join(data_dir, name) for name in listdir(data_dir) if 'CLASS' in name]
    for filename in data_files:
        estimator.add_test_point(filename)

    # To process all test points, use 'process_test_points()'
    # To process a list of test points use 'process_test_points(list)'
    # Where list is a list of point labels
    estimator.process_test_points()

    # To get the results of all points, use 'get_results()'
    # To get the results of a list of test points, use 'get_results(list)'
    # Where list is a list of point labels
    results = estimator.get_results()

    # Visualize the results
    # To plot the results of all points, use 'plot_xxx_results()'
    # To get the results of a list of test points, use 'plot_xxx_results(list)'
    # Where xxx is spe, oat, or adc
    visualizer = JmossVisualizer(estimator)
    visualizer.plot_spe_results()
    visualizer.plot_oat_results()
    visualizer.plot_adc_errors()

    # Fit a model SPE = f(Mic) to a list of results
    # To fit a model using all results, use 'fit_model()'
    # To fit a model using a list of results, use 'fit_model(list)'
    # Where list is a list of point labels
    # To also use AOA as a predictor variable, set 'use_aoa=True'
    estimator.fit_model()

    # Save figures
    # To save all, use save_figures()
    # To save some of them, use save_figures(label) where label is spe, oat, or adc
    visualizer.save_figures()

    # Print wind and eta results
    visualizer.print_aux_variable_results()
