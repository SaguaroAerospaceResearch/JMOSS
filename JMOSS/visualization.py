from matplotlib.pyplot import plot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm, rc
from JMOSS.utilities import translate_spe_to_errors
from numpy import array, zeros


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
            ax.plot(mach_ic, point.spe_ratio, color=colors[index], linestyle='-', linewidth=2, label=labels[index])
            ax.plot(mach_ic, point.inferences['spe ratio'], color=colors[index], linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        ax.set_ylabel("SPE ratio, $\Delta P_p / P_s$")
        ax.set_title(title, weight='bold')
        self.grid_on([ax])
        self.figures['spe'] = fig

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
            ax.plot(mach_ic, point.oat, color=colors[index], linestyle='-', linewidth=2, label=labels[index])
            ax.plot(mach_ic, point.inferences['oat'], color=colors[index], linestyle='--', linewidth=2)
        ax.set_xlabel("Instrument corrected Mach number, $M_{ic}$")
        ax.set_ylabel("Ambient temperature, $T_a$ [K]")
        ax.set_title(title, weight='bold')
        self.grid_on([ax])
        self.figures['oat'] = fig

    def plot_adc_errors(self, target_alt: array = zeros(1), labels=None, title=None):
        # Visualize altitude, airspeed, and Mach number corrections at sea level indicated altitude
        if labels is None:
            labels = self.estimator.results_names_list
            results = self.estimator.get_results()
        else:
            results = self.estimator.get_results(labels)
        if title is None:
            title = 'Air Data Computer Errors'
        rc('font', **self.font_rc)
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))  # noqa
        colors = self.colors
        curve_labels = ['Altitude corr. $\Delta H_{pc}$ [ft]', 'Airspeed corr. $\Delta V_{pc}$ [kts]',
                        'Mach corr. $\Delta M_{pc}$']
        for index, point in enumerate(results):
            curves = translate_spe_to_errors(point.spe_ratio, point.mach_ic, target_alt)
            for ax_num in range(3):
                axs[ax_num].plot(point.mach_ic, curves[ax_num], color=colors[index], linestyle='-',
                                 linewidth=2, label=labels[index])
                axs[ax_num].legend()
                axs[ax_num].set_xlabel("Instrument corrected Mach number, $M_{ic}$")
                axs[ax_num].set_ylabel(curve_labels[ax_num])
        axs[0].set_title('%s, Target Alt:  %0.1f ft PA' % (title, target_alt[0]), weight='bold')
        self.grid_on(axs)
        self.figures['adc'] = fig

    @staticmethod
    def grid_on(ax_list):
        for ax in ax_list:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(which='minor', alpha=0.3, linestyle=":")
            ax.grid(which='major', alpha=0.3, linestyle=":")
            ax.minorticks_on()

    def save_figures(self, labels=None):
        if labels is None:
            labels = list(self.figures.keys())
        for label in labels:
            fig = self.figures[label]
            fig.savefig(label + '.pdf', dpi=fig.dpi, edgecolor='w', format='pdf', transparent=True,
                        pad_inches=0.1, bbox_inches='tight')

    def print_aux_variable_results(self, labels=None):
        if labels is None:
            labels = self.estimator.results_names_list
        results = self.estimator.get_results()
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
