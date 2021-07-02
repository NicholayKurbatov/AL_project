import numpy as np
import matplotlib.pyplot as plt
from SolveClass import Solve_AL_overlap
from PostProcessing.DataPPClass import DataResClass
from PostProcessing.VisualPPClass import VisualResClass


def main():
    Vm = 18
    eta = 0.5
    Tn = 3
    Ph = 0.07
    config_data = {'dx': 0.74, 'dy': 0, 'b1': 1.5, 'b2': 1.5, 'H': 60,
                   'mu1': 0.194, 'mu2': 0.194, 'D1': 9.6e-3, 'D2': 9.6e-3,
                   'Vm1': Vm, 'Vm2': 0.97 * Vm, 'eta': eta, 'Tn': Tn,
                   'Ph': Ph, 'rho': 1.28, 't_end': 10}
    solve_ex = Solve_AL_overlap(config_data, terrian_type='suburban',
                                wind_force_type='pue')
    solve_ex.find_sol(report=True)
    data_cl = DataResClass(solve_ex)
    vis_cl = VisualResClass(solve_ex)

    wire_dist = data_cl.get_WireToWireDistance()
    Mw1, Mw2 = data_cl.get_WindMomentum()
    phi1, phi2 = data_cl.get_Phi()
    Dphi1, Dphi2 = data_cl.get_DPhi()
    t = data_cl.get_t()

    # data_cl.ResData(csv=True, filename='PostProcessing/DataResults.csv')
    '''
        graphs of the time dependence of the distance between the wires
    '''
    _, ax = plt.subplots()
    ax.plot(t, wire_dist)
    ax.set_xlabel('t, [s]'), ax.set_ylabel('wire distance, [m]')
    ax.grid()
    '''
        graphs of the dependence of the moments of wind forces on time
    '''
    _, ax = plt.subplots()
    ax.plot(t, Mw1, label='momentum for phi1')
    ax.plot(t, Mw2, label='momentum for phi2')
    ax.set_xlabel('t, [s]')
    ax.grid()
    '''
        waveform graphs
    '''
    _, ax = plt.subplots(nrows=2, facecolor='w', figsize=(14, 6))
    ax[0].plot(t, phi1, label='phi1')
    ax[0].plot(t, phi2, label='phi2')
    ax[0].set_ylabel('Phi')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, Dphi1 / np.max(Dphi1), label='Dphi1')
    ax[1].plot(t, Dphi2 / np.max(Dphi2), label='Dphi2')
    ax[1].set_ylabel('normalized d(Phi)/dt')
    ax[1].grid()
    ax[1].legend()
    '''
        graphs of phase trajectories
    '''
    _ = vis_cl.PhaseTrContour_plot(F_state_plot=True)
    plt.show()