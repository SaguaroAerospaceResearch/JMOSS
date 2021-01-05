"""
JMOSS Python implementation for public release

Jurado-McGehee Online Self Survey (JMOSS)
Version 6.0
January 2021
Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.
"""

from pandas import read_csv
from os.path import splitext, basename
from JMOSS.utilities import mach_from_qc_pa
from numpy import rad2deg


class JmossEstimator:
    def __init__(self, parameter_names: dict):
        self.test_points = {}
        self.spe_results = {}
        self.parameter_names = parameter_names
        self.messages = self.generate_console_messages(parameter_names)
        self.print_console_message('initialize')
        self.print_console_message('settings')

    def add_test_point(self, filename: str):
        data = read_csv(filename)
        label = splitext(basename(filename))[0]
        point = self.test_points.get(label, None)
        if point is not None:
            raise IndexError('Test point labeled %s has already been added.' % label)
        self.test_points[label] = data
        info = self.get_test_point_summary(label)
        self.print_console_message('new point', label)
        self.print_console_message('point info', info)

    def get_test_point(self, label: str):
        point = self.test_points.get(label, None)
        if point is None:
            raise IndexError('Test point labeled %s is not in the estimator.' % label)
        return point

    def get_test_point_summary(self, label: str):
        total_pres = self.get_test_point_parameter(label, 'total pressure')
        static_pres = self.get_test_point_parameter(label, 'static pressure')
        mach = mach_from_qc_pa((total_pres - static_pres) / static_pres)
        alt = self.get_test_point_parameter(label, 'geometric height')
        alt_tol = (alt.max() - alt.min()) / 2
        roll = self.get_test_point_parameter(label, 'roll angle')
        turning = abs(rad2deg(roll)) > 15
        turn_mach = mach[turning].mean()
        info = {'Min. speed': '%0.2f M' % mach.min(), 'Max. speed': '%0.2f M' % mach.max(),
                'Level turn': '%0.2f M' % turn_mach.mean(), 'Min. alt': '%0.2f Kft' % (alt.min() / 1000),
                'Max. alt': '%0.2f Kft' % (alt.max() / 1000), 'Alt. tolerance': u'\u00B1%0d ft' % alt_tol}
        return info

    def get_test_point_parameter(self, label: str, parameter_name: str):
        data = self.get_test_point(label)
        das_name = self.parameter_names[parameter_name]
        parameter = data[das_name].to_numpy()
        return parameter

    @property
    def test_point_names_list(self):
        return list(self.test_points.keys())

    @property
    def num_test_points(self):
        return len(self.test_points.keys())

    @staticmethod
    def generate_console_messages(settings: dict):
        # Create empty dictionary
        messages = {}

        # Write initialization message
        init_strs = [
            '*********************************************************************************************',
            '**************** Jurado-McGehee Online Self Survey ADS Calibration Algorithm ****************',
            '**************************      Version 6.0, January 2021     *******************************',
            '*********************************************************************************************']
        init_msg = '\n'.join(init_strs)
        messages['initialize'] = init_msg

        # Write settings message
        settings_strs = ['%s : %s' % (key, value) for key, value in settings.items()]
        settings_msg1 = '\nA JMOSS estimator has been initialized with the following DAS parameter names:\n'
        settings_msg2 = '\n'.join(settings_strs)
        messages['settings'] = settings_msg1 + settings_msg2 + '\n\n'

        # New test point
        messages['new point'] = 'Test point %s has been added:\n'
        messages['point info'] = '%s\n\n'

        # Processing test point
        messages['processing'] = 'Processing test point %s...'
        messages['done'] = 'Done.\n'

        return messages

    def print_console_message(self, message_id: str, message_variables=None):
        message = self.messages[message_id]
        if message_variables is not None:
            if isinstance(message_variables, dict):
                dict_variables = ['%s : %s' % (key, value) for key, value in message_variables.items()]
                dict_strs = '\n'.join(dict_variables)
                print(message % dict_strs, end='')
            else:
                print(message % message_variables, end='')
        else:
            print(message, end='')

    def get_spe_results(self, label: str):
        results = self.spe_results.get(label, None)
        if results is None:
            raise IndexError('Test point %s has not been processed.' % label)
        return results

    def process_test_points(self, labels: list = None):
        if labels is None:
            labels = self.test_point_names_list
        for label in labels:
            self.__process_test_point(label)

    def __process_test_point(self, label):
        # Print processing message
        self.print_console_message('processing', label)

        # Unload test point data frame
        data = self.get_test_point(label)

        # Unpack parameters
        # Air data
        tot_pres = self.get_test_point_parameter(label, 'total pressure')
        stat_pres = self.get_test_point_parameter(label, 'static pressure')
        tot_temp = self.get_test_point_parameter(label, 'total temperature')
        aoa = self.get_test_point_parameter(label, 'angle of attack')
        aos = self.get_test_point_parameter(label, 'angle of slideslip')

        # GPS
        n_vel = self.get_test_point_parameter(label, 'north velocity')
        e_vel = self.get_test_point_parameter(label, 'east velocity')
        d_vel = self.get_test_point_parameter(label, 'down velocity')
        height = self.get_test_point_parameter(label, 'geometric height')

        # EGI
        roll = self.get_test_point_parameter(label, 'roll angle')
        pitch = self.get_test_point_parameter(label, 'pitch angle')
        yaw = self.get_test_point_parameter(label, 'true heading')

        # Compute SPE

        # Print done message
        self.print_console_message('done')

