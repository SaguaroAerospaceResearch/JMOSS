import numpy as np
import math
from scipy.linalg import block_diag


# State update equations (AKA YDOT if you calculate them all)

def update_step(states, controls, constants, TimeStep):
    # 8 states
    pa_state = states[0]
    ta_state = states[1]
    vwn_state = states[2]
    vwe_state = states[3]
    vwd_state = states[4]
    d_pp_state = states[5]
    alpha_state = states[6]
    beta_state = states[7]

    # controls are balloon survey parameters
    l_dot = controls[0]
    lambda_dot = controls[1]
    h_dot = controls[2]
    time_of_day = controls[3]

    # survey model constants (including omega)
    b_pa1 = constants[0][0]
    b_pa2 = constants[0][1]
    b_pa3 = constants[0][2]
    b_pa4 = constants[0][3]
    b_pa5 = constants[0][4]

    b_ta1 = constants[1][0]
    b_ta2 = constants[1][1]
    b_ta3 = constants[1][2]
    b_ta4 = constants[1][3]
    b_ta5 = constants[1][4]

    b_vwn1 = constants[2][0]
    b_vwn2 = constants[2][1]
    b_vwn3 = constants[2][2]
    b_vwn4 = constants[2][3]
    b_vwn5 = constants[2][4]

    b_vwe1 = constants[3][0]
    b_vwe2 = constants[3][1]
    b_vwe3 = constants[3][2]
    b_vwe4 = constants[3][3]
    b_vwe5 = constants[3][4]

    omega = constants[4][0]

    # update equation
    g_update = np.mat([
        [TimeStep * (l_dot * b_pa1 + lambda_dot * b_pa2 + h_dot * b_pa3 + omega * math.cos(omega * time_of_day)
                     * b_pa4 - omega * math.sin(omega * time_of_day) * b_pa5) + pa_state],
        [TimeStep * (l_dot * b_ta1 + lambda_dot * b_ta2 + h_dot * b_ta3 + omega * math.cos(omega * time_of_day)
                     * b_ta4 - omega * math.sin(omega * time_of_day) * b_ta5) + ta_state],
        [TimeStep * (l_dot * b_vwn1 + lambda_dot * b_vwn2 + h_dot * b_vwn3 + omega * math.cos(omega * time_of_day)
                     * b_vwn4 - omega * math.sin(omega * time_of_day) * b_vwn5) + vwn_state],
        [TimeStep * (l_dot * b_vwe1 + lambda_dot * b_vwe2 + h_dot * b_vwe3 + omega * math.cos(omega * time_of_day)
                     * b_vwe4 - omega * math.sin(omega * time_of_day) * b_vwe5) + vwe_state],
        [vwd_state],
        [d_pp_state],
        [alpha_state],
        [beta_state]
    ])

    return g_update


def phi_matrix(states, controls, constants, TimeStep):
    identity = np.mat([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return identity


def noise_matrix_q(design_matrix, controls, constants):
    # design matrix from balloon model

    # controls are balloon survey parameters
    l_dot = controls[0]
    lambda_dot = controls[1]
    h_dot = controls[2]
    time_of_day = controls[3]

    # survey model constants (including omega)
    b_pa1 = constants[0][0]
    b_pa2 = constants[0][1]
    b_pa3 = constants[0][2]
    b_pa4 = constants[0][3]
    b_pa5 = constants[0][4]

    b_ta1 = constants[1][0]
    b_ta2 = constants[1][1]
    b_ta3 = constants[1][2]
    b_ta4 = constants[1][3]
    b_ta5 = constants[1][4]

    b_vwn1 = constants[2][0]
    b_vwn2 = constants[2][1]
    b_vwn3 = constants[2][2]
    b_vwn4 = constants[2][3]
    b_vwn5 = constants[2][4]

    b_vwe1 = constants[3][0]
    b_vwe2 = constants[3][1]
    b_vwe3 = constants[3][2]
    b_vwe4 = constants[3][3]
    b_vwe5 = constants[3][4]

    omega = constants[4][0]

    ydot = np.mat([
        [l_dot * b_pa1 + lambda_dot * b_pa2 + h_dot * b_pa3 + omega * math.cos(omega * time_of_day)
         * b_pa4 - omega * math.sin(omega * time_of_day) * b_pa5,
         l_dot * b_ta1 + lambda_dot * b_ta2 + h_dot * b_ta3 + omega * math.cos(omega * time_of_day)
         * b_ta4 - omega * math.sin(omega * time_of_day) * b_ta5,
         l_dot * b_vwn1 + lambda_dot * b_vwn2 + h_dot * b_vwn3 + omega * math.cos(omega * time_of_day)
         * b_vwn4 - omega * math.sin(omega * time_of_day) * b_vwn5,
         l_dot * b_vwe1 + lambda_dot * b_vwe2 + h_dot * b_vwe3 + omega * math.cos(omega * time_of_day)
         * b_vwe4 - omega * math.sin(omega * time_of_day) * b_vwe5]
    ])

    q_upper = np.matmul(ydot.transpose(), ydot)

    q_output = block_diag(q_upper, np.identity(4))

    return q_output