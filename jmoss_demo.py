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
    # Where list is a list of point labels
    estimator.process_test_points()

    # To get the results of all points, use 'get_results()'
    # To get the results of a list of test points, use 'get_results(list)'
    # Where list is a list of point labels
    results = estimator.get_results()

    # Fit the final model SPE = f(Mic) to a list of results
    # To fit a model using all results and default settings, use 'fit_model()'
    # To fit a model using a list of results, use 'fit_model(labels=list)'
    # Where list is a list of point labels
    # If desired, use knots=list to provide a list of mach_ic inflection points to help characterize the transonic curve
    # Alternatively, use num_knots=float to provide a fixed number of inflection points to use in the transonic region
    estimator.fit_model()

    # Extract the final model for viewing/exporting
    # To get the model output for default range of mach_ic values, use predict()
    # Use alpha to specify the significance level of the confidence interval
    # You can also feed in a custom array of mach_ic values using predict(mach_ic=array)
    model = estimator.spe_model
    model_mach_ic = model.mach_ic
    model_spe_ratio, model_ci = model.predict(alpha=0.01)

    # Set up the visualizer for plotting
    visualizer = JmossVisualizer(estimator)

    # Print auxiliary variable (wind and eta) results
    visualizer.print_aux_variable_results()

    # To plot the results of all points, use 'plot_xxx_results()'
    # To get the results of a list of test points, use 'plot_xxx_results(list)'
    # Where xxx is spe, oat, or adc
    visualizer.plot_spe_results()
    visualizer.plot_oat_results()
    visualizer.plot_adc_errors()

    # Visualize the final model
    # Use alpha to specify the significance level of the confidence interval
    # Use standalone=False to plot on top of the results plots
    # Otherwise, use standalone=True to plot on its own plot
    visualizer.plot_spe_model(alpha=0.01, standalone=False)

    # Visualize the ADC error models
    # Use alpha to specify the significance level of the confidence interval
    # Use standalone=False to plot on top of the results plots
    # Otherwise, use standalone=True to plot on its own plot
    visualizer.plot_adc_model(alpha=0.01, standalone=False)

    # Save figures
    # To save all, use save_figures()
    # To save some of them, use save_figures(labels) where "labels" is a list containing any combination of:
    # 'spe results', 'oat results', 'adc results', 'spe model', or 'adc model'
    # visualizer.save_figures()
