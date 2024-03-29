"""
JMOSS Python implementation for public release

Jurado-McGehee Online Self Survey (JMOSS)
Version 6.0
April 2021
Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.
"""

from pandas import read_csv
from os.path import splitext, basename
from JMOSS.utilities import mach_from_qc_pa, iterate_pa_oat
from numpy import rad2deg, sqrt, zeros, column_stack, cos, tan, arcsin, arctan, diag, ix_, argmax, argmin, array, \
    hstack, linspace, ones
from numpy.linalg import inv, eig
from scipy.optimize import least_squares
from scipy.spatial.transform import rotation as r
from scipy.stats import chi2


class JmossEstimator:
    def __init__(self, parameter_names: dict):
        self.flight_data = {}
        self.spe_results = {}
        self.spe_model = None
        self.parameter_names = parameter_names
        self.messages = self.generate_console_messages(parameter_names)
        self.print_console_message('initialize')
        self.print_console_message('settings')
        if 'ambient temperature' in parameter_names.keys():
            self.print_console_message('using oat')
            self.use_oat = True
        elif 'total temperature' in parameter_names.keys():
            self.print_console_message('using tat')
            self.use_oat = False
        else:
            raise (KeyError('Neither ambient temperature nor total temperature were provided.'))

    @property
    def test_point_names_list(self):
        return list(self.flight_data.keys())

    @property
    def num_test_points(self):
        return len(self.flight_data.keys())

    @property
    def results_names_list(self):
        return list(self.spe_results.keys())

    @property
    def num_results(self):
        return len(self.spe_results.keys())

    def add_test_point(self, filename: str):
        label = splitext(basename(filename))[0]
        point = self.flight_data.get(label, None)
        if point is not None:
            raise IndexError('Test point labeled %s has already been added.' % label)
        dataframe = read_csv(filename)
        self.flight_data[label] = dataframe
        info = self.get_test_point_summary(label)
        self.print_console_message('new point', label)
        self.print_console_message('point info', info)

    def get_test_point(self, label: str):
        point = self.flight_data.get(label, None)
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

    @staticmethod
    def generate_console_messages(settings: dict):
        # Create empty dictionary
        messages = {}

        # Write initialization message
        init_strs = [
            '*********************************************************************************************',
            '**************** Jurado-McGehee Online Self Survey ADS Calibration Algorithm ****************',
            '****************************      Version 6.0, April 2021     *******************************',
            '*********************************************************************************************']
        init_msg = '\n'.join(init_strs)
        messages['initialize'] = init_msg

        # Temperature messages
        messages['using oat'] = 'Ambient temperature data provided. JMOSS will not derive OAT.\n\n'
        messages['using tat'] = 'Total temperature data provided. JMOSS will self-derive OAT.\n\n'

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

        # Fitting model
        messages['fitting'] = '\nFitting model to %d tests points...'

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

    def get_results(self, labels: list = None):
        if labels is None:
            labels = self.test_point_names_list
        all_results = []
        for label in labels:
            results = self.spe_results.get(label, None)
            if results is None:
                raise IndexError('Test point %s has not been processed.' % label)
            all_results.append(results)
        return all_results

    def process_test_points(self, labels: list = None):
        if labels is None:
            labels = self.test_point_names_list
        for label in labels:
            self.__process_test_point(label)

    def __process_test_point(self, label):
        # Check for flight data
        if label not in self.test_point_names_list:
            raise IndexError('Test point %s not found.' % label)

        # Print processing message
        self.print_console_message('processing', label)

        # Use nonlinear least squares to solve for unknown variables
        params = zeros(5)
        wind_to_nav = self.get_frame_transform(label)
        flight_data = self.extract_flight_data(label)
        lsq = least_squares(self.jmoss_obj_tat, params, args=[wind_to_nav, flight_data], method='lm')

        # Use the lsq results to produce and record SPE results along wth auxiliary data for model fitting
        roll = self.get_test_point_parameter(label, 'roll angle')
        turn_idx = abs(rad2deg(roll)) > 10
        results = self.SpeResults(flight_data, lsq, turn_idx)
        self.spe_results[label] = results

        # Print done message
        self.print_console_message('done')

    @staticmethod
    def jmoss_obj_tat(params, wind_to_nav, flight_data):
        # Unload data
        tot_pres = flight_data[:, 0]
        tat = flight_data[:, 1]
        gs_n_meas = flight_data[:, 2:5]
        height = flight_data[:, 5]

        # Unload model parameters
        pa_bias = params[0]
        wind = params[[1, 2, 3]]
        eta_model = params[4]

        # Iterate to find ambient pressure, oat, and mach based on current parameter estimates
        amb_pres, oat, mach_pc = iterate_pa_oat(height, tot_pres, tat, pa_bias, eta_model)

        # Compute true airspeed then rotate from wind frame to nav frame to compare against GPS
        tas_w = mach_pc * sqrt(oat / 288.1500) * 661.478827231622
        tas_n = wind_to_nav.apply(column_stack([tas_w, 0 * tas_w, 0 * tas_w]))
        gs_n_est = tas_n + wind
        error = gs_n_est.flatten(order='F') - gs_n_meas.flatten(order='F')
        return error

    def extract_flight_data(self, label):
        tot_pres = self.get_test_point_parameter(label, 'total pressure')
        tat = self.get_test_point_parameter(label, 'total temperature')
        n_vel = (1 / 1.6878) * self.get_test_point_parameter(label, 'north velocity')
        e_vel = (1 / 1.6878) * self.get_test_point_parameter(label, 'east velocity')
        d_vel = (1 / 1.6878) * self.get_test_point_parameter(label, 'down velocity')
        height = self.get_test_point_parameter(label, 'geometric height')
        stat_pres = self.get_test_point_parameter(label, 'static pressure')
        aoa = self.get_test_point_parameter(label, 'angle of attack')
        data = column_stack([tot_pres, tat, n_vel, e_vel, d_vel, height, stat_pres, aoa])
        return data

    def get_frame_transform(self, label):
        # Unload angular data
        roll = self.get_test_point_parameter(label, 'roll angle')
        pitch = self.get_test_point_parameter(label, 'pitch angle')
        yaw = self.get_test_point_parameter(label, 'true heading')
        aos_ind = self.get_test_point_parameter(label, 'angle of slideslip')

        # Unload GPS data
        n_vel = self.get_test_point_parameter(label, 'north velocity')
        e_vel = self.get_test_point_parameter(label, 'east velocity')
        d_vel = self.get_test_point_parameter(label, 'down velocity')

        # Compute and correct vane angles
        gamma = arcsin(-d_vel / (sqrt(n_vel ** 2 + e_vel ** 2)))
        alpha_corr = pitch - gamma
        beta_corr = arctan(cos(alpha_corr) * tan(aos_ind))

        # Compute wind-to-nav rotation matrices
        body_to_nav = r.Rotation.from_euler('ZYX', column_stack([yaw, pitch, roll]))
        wind_to_body = r.Rotation.from_euler('ZYX', column_stack([-beta_corr, alpha_corr, 0 * alpha_corr])).inv()
        wind_to_nav = body_to_nav * wind_to_body
        return wind_to_nav

    # Subclass for computing and storing SPE estimates with uncertainty
    class SpeResults:
        def __init__(self, flight_data, lsq_results, turn_idx):
            # Unload flight data
            tot_pres = flight_data[:, 0]
            tat = flight_data[:, 1]
            height = flight_data[:, 5]
            stat_pres = flight_data[:, 6]
            aoa = flight_data[:, 7]

            # Unload least squares results
            beta = lsq_results.x
            jac = lsq_results.jac
            sse = lsq_results.cost
            mse = sse / (jac.shape[0] - beta.shape[0])
            beta_cov = mse * inv(jac.T @ jac)

            # Compute results
            pa_bias = beta[0]
            eta_model = beta[4]
            amb_pres, oat, mach_pc = iterate_pa_oat(height, tot_pres, tat, pa_bias, eta_model)
            mach_ic = mach_from_qc_pa((tot_pres - stat_pres) / stat_pres)
            spe_ratio = (stat_pres - amb_pres) / stat_pres

            # Store results
            self.amb_pres = amb_pres
            self.stat_pres = stat_pres
            self.spe_ratio = spe_ratio
            self.mach_pc = mach_pc
            self.mach_ic = mach_ic
            self.aoa = aoa
            self.oat = oat
            self.turn_idx = turn_idx
            self.pa_bias = beta[0]
            self.eta = beta[4]
            self.wind = beta[1:4]
            self.raw_stats = dict(parameters=beta, covariance=beta_cov)
            self.flight_data = flight_data
            self.sigmas = {}
            self.generate_inferences()

        def generate_inferences(self):
            # Unload flight data
            flight_data = self.flight_data
            tot_pres = flight_data[:, 0]
            tat = flight_data[:, 1]
            height = flight_data[:, 5]
            stat_pres = flight_data[:, 6]

            # For SPE and OAT, we need to consider the covariance matrix of Pa bias and Eta
            # Generate unit circle in 2D using 4 corners
            circle = array([[1, 0], [0, 1], [-1, 0], [0, -1]]).T

            # Scale unit sphere using eigenvalues and eigenvectors
            param_id = [0, 4]
            dim = len(param_id)
            parameters = self.raw_stats['parameters']
            covariance = self.raw_stats['covariance']
            sub_p = parameters[param_id].reshape((dim, 1))
            sub_cov = covariance[ix_(param_id, param_id)]
            w, v = eig(sub_cov)
            scale = sqrt(diag(w))
            ellipse = v @ scale @ circle + sub_p

            # Now evaluate all four points to find the min/max of pa and oat
            amb_press = []
            oats = []
            for point in ellipse.T:
                amb_pres, oat, _ = iterate_pa_oat(height, tot_pres, tat, point[0], point[1])
                amb_press.append(amb_pres.mean())
                oats.append(oat.mean())
            id_min_pa = argmin(amb_press)
            id_max_pa = argmax(amb_press)
            id_min_oat = argmin(oats)
            id_max_oat = argmax(oats)

            low_pa, _, _ = iterate_pa_oat(height, tot_pres, tat, ellipse[0, id_min_pa], ellipse[1, id_min_pa])
            hi_pa, _, _ = iterate_pa_oat(height, tot_pres, tat, ellipse[0, id_max_pa], ellipse[1, id_max_pa])
            _, low_oat, _ = iterate_pa_oat(height, tot_pres, tat, ellipse[0, id_min_oat], ellipse[1, id_min_oat])
            _, hi_oat, _ = iterate_pa_oat(height, tot_pres, tat, ellipse[0, id_max_oat], ellipse[1, id_max_oat])

            hi_spe = (stat_pres - low_pa) / stat_pres
            low_spe = (stat_pres - hi_pa) / stat_pres

            # Record values
            self.sigmas['spe ratio'] = (hi_spe - low_spe) / 2
            self.sigmas['oat'] = (hi_oat - low_oat) / 2

            # For winds, we will just record the confidence interval for each dimension separately
            labels = ['north wind', 'east wind', 'down wind']
            wind_cov = self.raw_stats['covariance'][1:4, 1:4]
            for index, label in enumerate(labels):
                # For each dimension of wind, compute its 1-alpha CI
                sigma2 = wind_cov[index, index]
                sigma = sqrt(sigma2)
                self.sigmas[label] = sigma

            # Finally eta
            sigma2 = self.raw_stats['covariance'][4, 4]
            sigma = sqrt(sigma2)
            self.sigmas['eta'] = sigma

    class SpeModel:
        def __init__(self, mach_ic: array, stats: dict):
            self.mach_ic = mach_ic
            self.stats = stats

        def predict(self, mach_ic: array = None, alpha=None):
            if mach_ic is None:
                mach_ic = self.mach_ic
            # Build regression matrix
            x_pred = column_stack([ones(mach_ic.shape), mach_ic, mach_ic ** 2])
            # If there are knots, append the regression matrix with the knot columns
            knots = self.stats['knots']
            if knots is not None:
                mach_knots = mach_ic.reshape(-1, 1) - knots.reshape(-1, 1).T
                mach_knots[mach_knots < 0] = 0
                x_pred = hstack([x_pred, mach_knots ** 2])
            # Predict spe ratio
            spe_ratio = x_pred @ self.stats['betas']
            # Compute the standard error at the prediction points
            mse = self.stats['mse']
            kernel = self.stats['kernel']
            spe_std = sqrt(diag(mse * x_pred @ kernel @ x_pred.T))
            # If a significance level was provided, multiply the standard error to cover the (1-alpha)% interval
            if alpha is None:
                chi2val = 1
            else:
                chi2val = chi2.ppf(1 - alpha, 1)
            ci = sqrt(chi2val) * spe_std
            return spe_ratio, ci

    def fit_model(self, labels=None, knots=None, num_knots=None):
        # Collect results
        results = self.get_results(labels)
        self.print_console_message('fitting', len(results))
        # Collect mach and spe ratio from results, excluding samples where aircraft was turning
        machs = []
        sigmas = []
        spes = []
        for point in results:
            turning = point.turn_idx
            mach_ic = point.mach_ic[~turning]
            spe = point.spe_ratio[~turning]
            sigma = point.sigmas['spe ratio'][~turning]
            machs.append(mach_ic)
            spes.append(spe)
            sigmas.append(sigma)
        machs = hstack(machs)
        spes = hstack(spes)
        sigmas = hstack(spes) ** 2
        # Compute the weight matrix
        w = 1 / sigmas
        w_mat = diag(w)
        # Build the regression matrix
        x_mat = column_stack([ones(machs.shape), machs, machs ** 2])
        # If supersonic, use spline knots to characterize transonic shapes and append to the regression matrix
        if machs.max(initial=None) > 0.9:
            if knots is None:
                if num_knots is None:
                    num_knots = 20
                knots = linspace(0.9, machs.max(initial=None), num_knots)[0:-1]
            else:
                knots = array(knots)

            mach_knots = machs.reshape(-1, 1) - knots.reshape(-1, 1).T
            mach_knots[mach_knots < 0] = 0
            x_mat = hstack([x_mat, mach_knots ** 2])
        # Build the stats model
        kernel = inv(x_mat.T @ w_mat @ x_mat)
        betas = kernel @ x_mat.T @ w_mat @ spes
        res = spes - x_mat @ betas
        sse = (res ** 2).sum()
        mse = sse / (res.shape[0] - x_mat.shape[1])
        stats = dict(betas=betas, kernel=kernel, mse=mse, knots=knots)
        smooth_mach = linspace(machs.min(initial=None), machs.max(initial=None), 1000)
        self.spe_model = self.SpeModel(smooth_mach, stats)
        self.print_console_message('done')
