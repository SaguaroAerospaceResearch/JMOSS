"""
Atmosphere Model class for modeling atmosphere from balloon and tower fly by data.
Written by Juan Jurado and Clark McGehee
"""

from typing import Union
from numpy import array, sin, cos, pi, diag
from pandas import DataFrame, to_numeric, concat
from datetime import datetime, timedelta
from pytz import timezone
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import re


# TODO: GPR


class AtmoModel:
    def __init__(self):
        self.dataframes = {}
        self.models = None

    def add_balloon_data(self, filename: str):
        # Open file
        file = open(filename, 'r', encoding='iso-8859-1')
        text_data = file.read()
        file.close()

        # Read launch date time info
        datetime_pattern = r'Launch Time: (\d{4})Z\n\n\s+Launch Date: (\d+ \w+ \d+)'
        datetime_match = re.search(datetime_pattern, text_data)
        time_str = datetime_match.group(1)
        date_str = datetime_match.group(2)
        launch_datetime = datetime.strptime(date_str + ' ' + time_str, '%d %b %y %H%M').replace(tzinfo=timezone('UTC'))

        # Read atmospheric data
        data_pattern = r'(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+)\s+(-?\d+.\d+)' \
                       r'\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\n'
        data_match = re.findall(data_pattern, text_data)
        labels = ['time_s', 'height_m', 'pressure_hpa', 'temp_degc', 'dewp_degc', 'relhum_pct', 'wspeed_ms',
                  'wdir_deg', 'lat_deg', 'lon_deg', 'gpsalt_m']
        df = DataFrame(data_match, columns=labels)
        df = df.apply(to_numeric)

        # Clean up TM dropouts
        df = df.drop(df[abs(df['lat_deg']) > 90].index)
        df = df.drop(df[abs(df['lon_deg']) > 180].index)
        df = df.drop(df[abs(df['gpsalt_m']) < 0].index)
        df = df.drop(df[abs(df['wspeed_ms']) > 150].index)

        # Add derived columns
        df['pressure_psi'] = df.apply(lambda row: row['pressure_hpa'] * 0.0145038, axis=1)
        df['datetime'] = df.apply((lambda row: launch_datetime + timedelta(seconds=row['time_s'])), axis=1)
        df['ssm'] = df.apply(lambda row: self.compute_ssm(row['datetime']), axis=1)
        df['sin_ssm'] = df.apply(lambda row: self.compute_sin_ssm(row['ssm']), axis=1)
        df['cos_ssm'] = df.apply(lambda row: self.compute_cos_ssm(row['ssm']), axis=1)

        # Set datetime as index
        df.set_index('datetime')

        # Record dataframe
        self.dataframes[filename] = df

    def fit_model(self, balloon_labels: list[str] = None) -> array:
        if balloon_labels is None:
            balloon_labels = self.dataframes.keys()

        dfs = [self.dataframes[label] for label in balloon_labels]
        master_df = concat(dfs)
        design_mtx = master_df[['lat_deg', 'lon_deg', 'gpsalt_m', 'sin_ssm', 'cos_ssm']]
        output_labels = ['pressure_psi', 'temp_degc', 'wspeed_ms', 'wdir_deg']

        for label in output_labels:
            mdl_info = self.train_model(master_df[label], design_mtx)
            if self.models is None:
                self.models = {label: mdl_info}
            else:
                self.models[label] = mdl_info

    @staticmethod
    def compute_ssm(date_time: datetime, time_zone: str = None) -> float:
        if time_zone is None:
            time_zone = 'MST'

        year = date_time.year
        month = date_time.month
        day = date_time.day
        midnight_dt = datetime(year, month, day, 0, tzinfo=timezone(time_zone))
        time_delta = date_time - midnight_dt
        ssm = time_delta.total_seconds()
        return ssm

    @ staticmethod
    def compute_sin_ssm(ssm: float) -> float:
        seconds_in_day = 24 * 60 * 60
        sin_ssm = sin(2 * pi * ssm / seconds_in_day)

        return sin_ssm

    @ staticmethod
    def compute_cos_ssm(ssm: float) -> float:
        seconds_in_day = 24 * 60 * 60
        cos_ssm = cos(2 * pi * ssm / seconds_in_day)

        return cos_ssm

    @staticmethod
    def train_model(output: Union[DataFrame, array], features: Union[DataFrame, array]) -> dict:
        # Scale features
        scaler = MinMaxScaler()
        scaler.fit(features)
        scaled_features = scaler.transform(features)

        # Fit model
        model = sm.OLS(output, scaled_features)
        res = model.fit()

        # Return model and scales
        model_info = {'model': res, 'scaler': scaler}

        return model_info

    def get_atmosphere(self, lat: float, lon: float, gps_alt: float, time: datetime) -> tuple[array, array]:
        if self.models is None:
            print('No model found, fitting model to all available data first.')
            self.fit_model()

        # Build predictor variable vector
        ssm = self.compute_ssm(time)
        sin_ssm = self.compute_sin_ssm(ssm)
        cos_ssm = self.compute_cos_ssm(ssm)
        labels = ['lat_deg', 'lon_deg', 'gpsalt_m', 'sin_ssm', 'cos_ssm']
        features = DataFrame(array([lat, lon, gps_alt, sin_ssm, cos_ssm]).reshape(1, -1), columns=labels)

        # Predict outputs
        output_labels = ['pressure_psi', 'temp_degc', 'wspeed_ms', 'wdir_deg']
        meas = []
        meas_var = []
        for label in output_labels:
            mdl_info = self.models[label]
            mdl = mdl_info['model']
            scaler = mdl_info['scaler']
            scaled_features = scaler.transform(features)
            pred = mdl.get_prediction(scaled_features)
            meas.append(pred.predicted_mean)
            meas_var.append(pred.var_pred_mean[0])

        meas = array(meas)
        meas_cov = diag(meas_var)

        return meas, meas_cov


