import numpy as np
from scipy.interpolate import interp1d


def fun_alphaW(x):
    '''
        coefficient that takes into account the unevenness of
        the wind pressure over the span of the overhead line :
            x -- float, windload in [Pa]
    '''
    tab_alphaW = np.array([[200, 240, 280, 300, 320, 360, 400, 500, 580],
                           [1, 0.94, 0.88, 0.85, 0.83, 0.8, 0.76, 0.71, 0.7]])
    if x <= 200:
        return 1
    elif x >= 580:
        return 0.7
    elif (x > 200) & (x < 580):
        f = interp1d(tab_alphaW[0,:], tab_alphaW[1,:])
        return f(x)


def fun_kL(x):
    '''
        coefficient that takes into account the effect of
        the span length on the wind load :
            x -- float, H (wire lenght) in [m]
    '''
    tab_kL = np.array([[50, 100, 150, 250],
                     [1.2, 1.1, 1.05, 1]])
    if x <= 50:
        return 1.2
    elif x >= 250:
        return 1
    elif (x > 50) & (x < 250):
        f = interp1d(tab_kL[0,:], tab_kL[1,:])
        return f(x)