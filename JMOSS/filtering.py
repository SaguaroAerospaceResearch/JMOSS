"""
Object Oriented Extended Kalman Filter class.
Written by Juan Jurado and Clark McGehee
Copyright Saguaro Aerospace Research LLC.
"""

from numpy import zeros, eye, array, s_
from numpy.linalg import inv
from datetime import datetime
from typing import Callable


class StateBlock:
    def __init__(self, label: str, num_states: int):
        self.label = label
        self.num_states = num_states

    def get_label(self) -> str:
        return self.label

    def get_num_states(self) -> int:
        return self.num_states

    def generate_dynamics(self, time_from: datetime, time_to: datetime) -> dict:
        pass


class PressureStateBlock(StateBlock):
    def __init__(self, label: str):
        StateBlock.__init__(self, label, 1)

    def generate_dynamics(self, time_from: datetime, time_to: datetime) -> dict:
        pass


class TemperatureStateBlock(StateBlock):
    def __init__(self, label: str):
        StateBlock.__init__(self, label, 1)

    def generate_dynamics(self, time_from: datetime, time_to: datetime) -> dict:
        pass


class WindStateBlock(StateBlock):
    def __init__(self, label: str):
        StateBlock.__init__(self, label, 3)

    def generate_dynamics(self, time_from: datetime, time_to: datetime) -> dict:
        pass


class DeltaPStateBlock(StateBlock):
    def __init__(self, label: str):
        StateBlock.__init__(self, label, 1)

    def generate_dynamics(self, time_from: datetime, time_to: datetime) -> dict:
        pass


class MeasurementProcessor:
    def __init__(self, label: str, state_block_labels: list):
        self.label = label
        self.state_block_labels = state_block_labels

    def get_label(self):
        return self.label

    def get_state_block_labels(self):
        return self.state_block_labels

    def generate_model(self, measurement: dict, generate_x_and_p_fun: Callable) -> dict:
        pass


class BalloonMeasurementProcessor(MeasurementProcessor):
    def __init__(self, label: str, state_block_labels: list):
        MeasurementProcessor.__init__(self, label, state_block_labels)
        self.meas_matrix = eye(4)

    def meas_func(self, state: array) -> array:
        meas = self.meas_matrix @ state
        return meas

    def generate_model(self, measurement: dict, generate_x_and_p_fun: Callable) -> dict:
        meas_est = measurement['estimate']
        meas_cov = measurement['covariance']
        model = {'z': meas_est, 'h': self.meas_func, 'H': self.meas_matrix, 'R': meas_cov}
        return model


class ExtendedKalmanFilter:
    def __init__(self, init_time: datetime = None):
        if init_time is None:
            init_time = datetime.now()

        self.time = init_time
        self.state = array([])
        self.covariance = array([[]])
        self.state_blocks = {}
        self.meas_processors = {}
        self.state_block_idx = {}
        self.history = {'time': []}

    def get_time(self):
        return self.time

    def add_state_block(self, state_block: StateBlock):
        # Store SB information
        sb_label = state_block.get_label()
        num_states = state_block.get_num_states()
        self.state_blocks[sb_label] = state_block

        # Create history container
        self.history[sb_label] = {'estimate': [], 'covariance': []}

        # Determine new size of filter state and covariance arrays, and the index for this SB
        cur_states = self.state.shape[0]
        total_states = cur_states + num_states
        sb_idx = s_[cur_states:total_states]
        self.state_block_idx[sb_label] = sb_idx

        # Create new state and covariance arrays
        new_state = zeros(total_states)
        new_state[0:cur_states] = self.state

        new_covariance = eye(total_states)
        new_covariance[0:cur_states, 0:cur_states] = self.covariance

        self.state = new_state
        self.covariance = new_covariance

    def get_stateblock_names_list(self) -> list:
        return list(self.state_block_idx.keys())

    def get_measurement_processor_names_list(self) -> list:
        return list(self.meas_processors.keys())

    def set_state_block_estimate(self, label: str, estimate: array):
        index = self.state_block_idx[label]
        self.state[index] = estimate

    def set_state_block_covariance(self, label: str, covariance: array):
        index = self.state_block_idx[label]
        self.covariance[index, index] = covariance

    def get_stateblock_estimate(self, label: str) -> array:
        index = self.state_block_idx[label]
        state = self.state[index]
        return state

    def get_stateblock_covariance(self, label: str) -> array:
        index = self.state_block_idx[label]
        covariance = self.covariance[index, index]
        return covariance

    def get_cross_covariance(self, sb_label1: str, sb_label2: str) -> array:
        index1 = self.state_block_idx[sb_label1]
        index2 = self.state_block_idx[sb_label2]
        cross_cov = self.covariance[index1, index2]
        return cross_cov

    def generate_x_and_p(self, state_block_labels: list) -> tuple[array, array]:
        # Get total number of states
        num_states = 0
        for label in state_block_labels:
            state_block = self.state_blocks[label]
            num_states += state_block.get_num_states()

        # Allocate containers
        state = zeros(num_states)
        covariance = zeros(num_states, num_states)

        # Fill in block diagonal values
        for label in state_block_labels:
            index = self.state_block_idx[label]
            this_state = self.get_stateblock_estimate(label)
            this_cov = self.get_stateblock_covariance(label)
            state[index] = this_state
            covariance[index, index] = this_cov

        # Fill in cross terms
        for label1 in state_block_labels:
            for label2 in state_block_labels:
                if label1 != label2:
                    index1 = self.state_block_idx[label1]
                    index2 = self.state_block_idx[label2]
                    cross_cov = self.get_cross_covariance(label1, label2)
                    covariance[index1, index2] = cross_cov

        return state, covariance

    def record_history(self, state_block_labels: list = None):
        if state_block_labels is None:
            state_block_labels = self.get_stateblock_names_list()

        self.history['time'].append(self.get_time())

        for label in state_block_labels:
            estimate = self.get_stateblock_estimate(label)
            covariance = self.get_stateblock_covariance(label)
            self.history[label]['estimate'].append(estimate)
            self.history[label]['covariance'].append(covariance)

    def propagate(self, to_time: datetime):
        from_time = self.get_time()
        state_block_labels = self.get_stateblock_names_list()
        global_phi = zeros(self.covariance.shape)
        global_qd = zeros(self.covariance.shape)
        for label in state_block_labels:
            # Unload state block dynamics
            state_block = self.state_blocks[label]
            dynamics = state_block.generate_dynamics(from_time, to_time)
            transition_fun = dynamics['g']
            transition_mtx = dynamics['Phi']
            process_noise_mtx = dynamics['Qd']

            # Propagate state estimates
            estimate = self.get_stateblock_estimate(label)
            new_estimate = transition_fun(estimate)
            self.set_state_block_estimate(label, new_estimate)

            # Record Qd and Phi for global propagate
            index = self.state_block_idx[label]
            global_phi[index, index] = transition_mtx
            global_qd[index, index] = process_noise_mtx

        # Update global covariance
        new_covariance = global_phi @ self.covariance @ global_phi + global_qd
        self.covariance = new_covariance

    def update(self, measurement: dict):
        # Unload measurement model
        mp_label = measurement['mp_label']
        meas_proc = self.meas_processors[mp_label]
        meas_model = meas_proc.generate_model(measurement, self.generate_x_and_p)
        meas = meas_model['z']
        meas_func = meas_model['h']
        jacobian = meas_model['H']
        meas_cov = meas_model['R']

        # Load Jacobian into global matrix
        num_states = self.state.shape[0]
        num_meas = meas.shape[0]
        global_jacob = zeros(num_meas, num_states)
        state_block_labels = meas_proc.get_state_block_labels()
        local_index = 0
        for label in state_block_labels:
            state_block = self.state_blocks[label]
            num_states = state_block.get_num_states()
            global_index = self.state_block_idx[label]
            global_jacob[:, global_index] = jacobian[:, local_index:num_states + 1]
            local_index += num_states

        # Execute update equations
        state, _ = self.generate_x_and_p(state_block_labels)
        covariance = self.covariance
        residual = meas - meas_func(state)
        kalman_gain = covariance @ global_jacob.T @ inv(global_jacob @ covariance @ global_jacob.T + meas_cov)
        new_state = self.state + kalman_gain @ residual
        new_cov = covariance - kalman_gain @ global_jacob @ covariance

        # Record
        self.state = new_state
        self.covariance = new_cov
