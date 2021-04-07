"""
JMOSS Air Data Visualization

Written by Juan Jurado, Clark McGehee

    Based on:
        Jurado, Juan D., and Clark C. McGehee. "Complete Online Algorithm for
        Air Data System Calibration." Journal of Aircraft 56.2 (2019): 517-528.

        Erb, Russell E. "Pitot-Statics Textbook." US Air Force Test Pilot School, Edwards AFB, CA (2020).
"""
from matplotlib import cm, rc
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from JMOSS.utilities import translate_spe_to_errors


class JmossVisualizer:
    def __init__(self, estimator):
        self.estimator = estimator
        self.figures = {}
        self.axes = {}
        self.colors = cm.get_cmap('tab20').colors
        self.font_rc = {'family': 'serif', 'weight': 'normal', 'size': 12}

    def plot_spe_results(self, labels=None, title=None):
        if labels is None:
            labels = self.estimator.results_names_list
            results = self.estimator.get_results()
        else:
            results = self.estimator.get_results(labels)
        if title is None:
            title = 'Static Position Error Ratio Results'
        rc('font', **self.font_rc)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # noqa
        colors = self.colors
        for index, point in enumerate(results):
            mach_ic = point.mach_ic
            spe_ratio = point.spe_ratio
            spe_std = point.sigmas['spe ratio']
            ax.plot(mach_ic, spe_ratio, color=colors[index], linestyle='-', linewidth=2,
                    label=labels[index] + '$\pm 1\sigma$')
            ax.plot(mach_ic, spe_ratio + spe_std, color=colors[index], linestyle='--', linewidth=2)
            ax.plot(mach_ic, spe_ratio - spe_std, color=colors[index], linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        ax.set_ylabel("SPE ratio, $\Delta P_p / P_s$")
        ax.set_title(title, weight='bold')
        self.grid_on(ax, 3)
        self.figures['spe results'] = [fig, ax]

    def plot_oat_results(self, labels=None, title=None):
        if labels is None:
            labels = self.estimator.results_names_list
            results = self.estimator.get_results()
        else:
            results = self.estimator.get_results(labels)
        if title is None:
            title = 'Outside Air Temperature Results'
        rc('font', **self.font_rc)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # noqa
        colors = self.colors
        for index, point in enumerate(results):
            mach_ic = point.mach_ic
            sigma = point.sigmas['oat']
            ax.plot(mach_ic, point.oat, color=colors[index], linestyle='-', linewidth=2,
                    label=labels[index] + '$\pm 1\sigma$')
            ax.plot(mach_ic, point.oat + sigma, color=colors[index], linestyle='--', linewidth=2)
            ax.plot(mach_ic, point.oat - sigma, color=colors[index], linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        ax.set_ylabel("Ambient temperature, $T_a$ [K]")
        ax.set_title(title, weight='bold')
        self.grid_on(ax)
        self.figures['oat results'] = [fig, ax]

    def plot_adc_errors(self, target_alt_ic: float = 0, labels=None, title=None):
        # Visualize altitude, airspeed, and Mach number corrections at sea level indicated altitude
        if labels is None:
            labels = self.estimator.results_names_list
            results = self.estimator.get_results()
        else:
            results = self.estimator.get_results(labels)
        if title is None:
            title = 'Air Data Computer Errors'
        rc('font', **self.font_rc)
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=False)  # noqa
        colors = self.colors
        curve_labels = ['$\Delta H_{pc}$ [ft]', '$\Delta V_{pc}$ [kts]', '$\Delta M_{pc}$']
        for index, point in enumerate(results):
            mach_ic = point.mach_ic
            spe_ratio = point.spe_ratio
            spe_std = point.sigmas['spe ratio']
            spe_curves = translate_spe_to_errors(spe_ratio, mach_ic, target_alt_ic)
            ci_curves1 = translate_spe_to_errors(spe_ratio + spe_std, mach_ic, target_alt_ic)
            ci_curves2 = translate_spe_to_errors(spe_ratio - spe_std, mach_ic, target_alt_ic)
            for ax_num in range(3):
                axs[ax_num].plot(mach_ic, spe_curves[ax_num], color=colors[index], linestyle='-',
                                 linewidth=2, label=labels[index] + '$\pm 1\sigma$')
                axs[ax_num].plot(mach_ic, ci_curves1[ax_num], color=colors[index], linestyle='--',
                                 linewidth=2)
                axs[ax_num].plot(mach_ic, ci_curves2[ax_num], color=colors[index], linestyle='--',
                                 linewidth=2)
                axs[ax_num].set_ylabel(curve_labels[ax_num])
        axs[0].set_title('%s, Ind. Alt:  %0.0f ft PA' % (title, target_alt_ic), weight='bold')
        axs[0].legend()
        axs[2].set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        self.grid_on(axs[0])
        self.grid_on(axs[1])
        self.grid_on(axs[2], 2)
        self.figures['adc results'] = [fig, axs]

    def plot_spe_model(self, title=None, alpha=0.05, standalone=False):
        if title is None:
            title = 'Static Position Error Ratio Model'
        rc('font', **self.font_rc)
        if standalone:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # noqa
        else:
            fig = self.figures['spe results'][0]
            ax = self.figures['spe results'][1]

        model = self.estimator.spe_model
        mach_ic = model.mach_ic
        spe_ratio, ci = model.predict(alpha=alpha)
        ax.plot(mach_ic, spe_ratio, color='r', linestyle='-', linewidth=2, label='model')
        ax.plot(mach_ic, spe_ratio + ci, color='r', linestyle='--', linewidth=2, label='%d%% CI' % (100 * (1 - alpha)))
        ax.plot(mach_ic, spe_ratio - ci, color='r', linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        ax.set_ylabel("SPE ratio, $\Delta P_p / P_s$")
        ax.set_title(title, weight='bold')
        if standalone:
            self.grid_on(ax, 3)
            self.figures['spe model'] = [fig, ax]

    def plot_adc_model(self, target_alt_ic: float = 0, title=None, standalone=False, alpha=0.05):
        # Visualize altitude, airspeed, and Mach number corrections at sea level indicated altitude
        if title is None:
            title = 'Air Data Computer Error Models'
        rc('font', **self.font_rc)
        if standalone:
            fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=False) # noqa
        else:
            fig = self.figures['adc results'][0]
            axs = self.figures['adc results'][1]
        curve_labels = ['$\Delta H_{pc}$ [ft]', '$\Delta V_{pc}$ [kts]', '$\Delta M_{pc}$']
        mach_ic = self.estimator.spe_model.mach_ic
        spe_ratio, spe_ci = self.estimator.spe_model.predict(alpha=alpha)
        spe_curves = translate_spe_to_errors(spe_ratio, mach_ic, target_alt_ic)
        ci_curves1 = translate_spe_to_errors(spe_ratio + spe_ci, mach_ic, target_alt_ic)
        ci_curves2 = translate_spe_to_errors(spe_ratio - spe_ci, mach_ic, target_alt_ic)
        for ax_num in range(3):
            axs[ax_num].plot(mach_ic, spe_curves[ax_num], color='r', linestyle='-', linewidth=2, label='model')
            axs[ax_num].plot(mach_ic, ci_curves1[ax_num], color='r', linestyle='--', linewidth=2,
                             label='%d%% CI' % (100 * (1 - alpha)))
            axs[ax_num].plot(mach_ic, ci_curves2[ax_num], color='r', linestyle='--', linewidth=2)
            axs[ax_num].set_ylabel(curve_labels[ax_num])
        axs[0].set_title('%s, Ind. Alt:  %0.0f ft PA' % (title, target_alt_ic), weight='bold')
        axs[0].legend()
        axs[2].set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        if standalone:
            self.grid_on(axs[0])
            self.grid_on(axs[1])
            self.grid_on(axs[2], 2)
            self.figures['adc model'] = [fig, axs]

    @staticmethod
    def grid_on(ax, sig_figs: float = None):
        if sig_figs is None:
            sig_figs = 1
        fmt = '%%.%df' % sig_figs
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.grid(which='minor', alpha=0.3, linestyle=":")
        ax.grid(which='major', alpha=0.3, linestyle=":")
        ax.minorticks_on()

    def save_figures(self, labels=None):
        if labels is None:
            labels = list(self.figures.keys())
        for label in labels:
            fig = self.figures[label][0]
            fig.savefig(label + '.pdf', dpi=fig.dpi, edgecolor='w', format='pdf', transparent=True,
                        pad_inches=0.1, bbox_inches='tight')

    def print_aux_variable_results(self, labels=None):
        if labels is None:
            labels = self.estimator.results_names_list
        results = self.estimator.get_results()
        wind_labels = ['north wind', 'east wind', 'down wind']
        print('\nAuxiliary variable results with 1-sigma values:')
        for index, point in enumerate(results):
            print(labels[index] + ':')
            est = point.eta
            ci = point.sigmas['eta']
            print('eta: %0.3f \u00B1 %0.3f' % (est, ci))
            for dim, label in enumerate(wind_labels):
                est = point.wind[dim]
                ci = point.sigmas[label]
                print('%s: %0.3f \u00B1 %0.3f' % (label, est, ci))
            print()
