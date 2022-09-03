"""
Development script for Virgin Galactic air data solution.
Written by Juan Jurado and Clark McGehee
August 28, 2022
"""

# Build AtmoModel class to ingest VG balloon data
from JMOSS.atmosphere import AtmoModel
from os import listdir
from os.path import join
from datetime import datetime
from pytz import timezone

if __name__ == '__main__':
    # Create model
    atmo_model = AtmoModel()

    # Add available .gps balloon files to model
    data_dir = '../vg_data/vh01/balloon'
    data_files = [join(data_dir, name) for name in listdir(data_dir) if '.gps' in name]
    for filename in data_files:
        atmo_model.add_balloon_data(filename)

    # Fit model to all available balloon data
    atmo_model.fit_model()

    # Sample model at some point
    # Expected output is meas = [Pa, Ta, Wspd, Wdir] and meas_cov is cov(meas)
    date_time = datetime(2021, 5, 23, 14, 31, 22, tzinfo=timezone('UTC'))
    meas, meas_cov = atmo_model.get_atmosphere(32.6057, -106.266603, 10310.4, date_time)

    print(f'Measurement:\n{meas}')
    print(f'Meas. covariance:\n{meas_cov}')

