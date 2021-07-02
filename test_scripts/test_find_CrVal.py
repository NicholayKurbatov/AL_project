import numpy as np
from СriticalValuesClass import SagBoomCrValClass


def main():
    Vm = 18
    eta = 0.5
    Tn = 10
    Ph = 0.07
    config_data = {'dx': 0.74, 'dy': 0, 'H': 60,
                   'mu1': 0.194, 'mu2': 0.194, 'D1': 9.6e-3, 'D2': 9.6e-3,
                   'Vm1': Vm, 'Vm2': 0.97 * Vm, 'eta': eta, 'Tn': Tn,
                   'Ph': Ph, 'rho': 1.28, 't_end': 10}
    CrVar_cl = SagBoomCrValClass(config_data, b=np.arange(0.91, 5., 1),
                                 terrian_type='suburban',
                                 wind_force_type='pue', symm=True)

    CrVar_cl.find_CrVal(fp_to_savejson='test_scripts/test_CrVal.json')
    CrVar_cl.CrVal_plot()
