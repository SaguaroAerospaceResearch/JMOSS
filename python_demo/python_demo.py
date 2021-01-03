#! python3
"""
This is a basic demo script for getting familiar with Python-based data processing and plotting.
Written by Silv Jurado, January 20201
"""
from pandas import read_csv
from matplotlib.pyplot import subplots, FormatStrFormatter, rc, show
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

if __name__ == '__main__':
    # Use Pandas to read CSV and import data as a Pandas DataFrame
    filename = 'C12_data.csv'
    data = read_csv(filename)

    # Slice data to events 5 through 7 and use a dictionary to associate C-12 parameter names to common names
    # range creates an iterable list that is understood by all for loops and list comprehension
    my_events = range(7, 10)
    c12_params = {'altitude': 'ADC_ALT_29',  # this is a dictionary with keys and values, values can be any object
                  'airspeed': 'ADC_IAS',
                  'time': 'Delta_Irig',
                  'events': 'ICU_EVNT_CNT'}

    # Here we can use list comprehension to grab the index of every row in data where the event in that row is found in
    # the list of events created by range. The enumerate method outputs two lists, the index and the value at that index
    row_idx = [index for index, value in enumerate(data[c12_params['events']]) if value in my_events]

    # Cut the data to the desired rows
    sliced_data = data.iloc[row_idx].reset_index()

    # Verify the "cut" worked and print some info
    time = sliced_data[c12_params['time']].to_numpy()  # .to_numpy turns it from a dataframe column to a numeric array
    airspeed = sliced_data[c12_params['airspeed']].to_numpy()
    altitude = sliced_data[c12_params['altitude']].to_numpy() / 1000
    events = sliced_data[c12_params['events']].to_numpy()
    info_list = ['Min event: %d', 'Max event: %d',
                 'Min airspeed: %0.2f KIAS', 'Max airspeed: %0.2f KIAS',
                 'Min altitude: %0.3f Kft', 'Max altitude: %0.3f Kft']
    info_str = '\n'.join(info_list)  # even the string '\n' is an object, and it has a 'join' method that works on lists
    info_data = (events.min(), events.max(), airspeed.min(), airspeed.max(), altitude.min(), altitude.max())
    print(info_str % info_data)

    # Now do a basic plot of time versus altitude and color code by speed
    mpl.use('Qt5Agg')
    rc('font', **{'family': 'serif', 'weight': 'normal', 'size': 12})
    fig, ax = subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Plot the trace of altitude vs. time
    scatter_handle = ax.scatter(time, altitude, s=3, c=airspeed, cmap='jet')
    fig.colorbar(scatter_handle, cax=cax, orientation='vertical')

    # Plot every event in the list of events with a different color and label
    colors = cm.tab20.colors
    handles = []
    for num, event in enumerate(my_events):
        idx = (events == event).argmax()
        handle, = ax.plot(time[idx], altitude[idx], 'o', color=colors[num], markerfacecolor=colors[num],
                          markersize=10, label='Event %d' % event)
        handles.append(handle)

    # Add legend, labels, and grids
    ax.legend(handles=handles, loc='upper right', fancybox=True, framealpha=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(which='minor', alpha=0.2, linestyle=":")
    ax.grid(which='major', alpha=0.2, linestyle=":")
    ax.minorticks_on()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [Kft MSL]")
    ax.set_title("C-12 Phugoid Example", fontweight='bold')
    cax.set_title('A/S')

    # Show figure and save figure. export_fig style export is native to matplotlib
    show()
    fig.savefig('demo.pdf', dpi=fig.dpi, edgecolor='w', format='pdf', transparent=True, pad_inches=0.1,
                bbox_inches='tight')
