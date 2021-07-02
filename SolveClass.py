##==============================================================================
# Imports
##==============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import sys
import os

from Func_using_in_SolveClass.CorrFactors import fun_kL, fun_alphaW


##==============================================================================
# Сlass for solving the problem of overhead line overlap
##==============================================================================
class Solve_AL_overlap():


    def __init__(self, anyData, terrian_type, wind_force_type='pue'):
        '''
            Object constructor:
                anyData -- filepath name/dict/DataFrame,
                           parametors data of OTL (overhead transmission lines), all in (СИ)
                terrian_type -- str: 'open'; 'suburban'
                wind_force_type -- str, if 'custom' F/H = Cx * WindLoad * D (simple form)
                                        else 'pue' F/H = (sime COEF.) * Cx * WindLoad * D (form by PUE)
        '''
        if isinstance(anyData, str) and os.path.exists(anyData):  # filepath
            data = pd.read_csv(anyData).astype(float)
        elif isinstance(anyData, dict):  # dict
            data = pd.DataFrame([anyData], dtype=float)
        elif isinstance(anyData, pd.core.frame.DataFrame):  # DataFrame
            data = anyData.astype(float)
        else:
            sys.exit('Constructor is empty or you are inputing wrong data format')
        # Checking parameters initialisation
        pos_cols = set(data.columns) - {'dy', 'Vm1', 'Vm2'}
        assert len(set(data.columns) - {'b1', 'b2'}) == 14, 'Not enough input parameters in anyData (without b1, b2)'
        assert np.all(data[pos_cols].values >= 0), \
            'Invalid values in your input anyData: values \'{}\' could not be negative'.format(
                ', '.join(data[pos_cols].columns[[data[pos_cols].values < 0][0][0]])
            )
        # Parameters initialisation
        df = data.to_dict('list')
        for i in df.keys():
            df[i] = df[i][0]
        self.sys_p = df
        self.area = terrian_type
        self.wF_type = wind_force_type

    def forward_F_state(self, phi_1):
        """
            returns forward F_state values for phi_1
        """
        phi_1 = np.asarray(phi_1)
        dx = self.sys_p['dx']
        dy = self.sys_p['dy']
        b1 = self.sys_p['b1']
        b2 = self.sys_p['b2']

        # make useful parameters
        r = lambda x: (b1 / b2) * (dx * np.cos(x) + dy * np.sin(x))
        alpha = (dx ** 2 + dy ** 2) ** 0.5

        psi = np.arctan2(-dy, dx)

        phi2_n = -psi - np.arccos(r(phi_1) / alpha)  # <--- negative solution branch
        phi2_p = -psi + np.arccos(r(phi_1) / alpha)  # <--- positive solution branch
        phi2 = np.vstack((phi2_n, phi2_p))  # <--- combine two solution branch together

        return phi2  # <-------- phi_2

    def backward_F_state(self, phi_2):
        """
            returns inv (backward) F_state values for phi_2
        """
        phi_2 = np.asarray(phi_2)
        dx = self.sys_p['dx']
        dy = self.sys_p['dy']
        b1 = self.sys_p['b1']
        b2 = self.sys_p['b2']

        # make useful parameters
        r = lambda x: (b2 / b1) * (dx * np.cos(x) + dy * np.sin(x))
        alpha = (dx ** 2 + dy ** 2) ** 0.5

        psi = np.arctan2(-dy, dx)

        phi1_n = -psi - np.arccos(r(phi_2) / alpha)  # <--- negative solution branch
        phi1_p = -psi + np.arccos(r(phi_2) / alpha)  # <--- positive solution branch
        phi1 = np.vstack((phi1_n, phi1_p))  # <--- combine two solution branch together

        return phi1  # <-------- phi_1

    def phys_check(self, phi_1, phi_2, get_a1_a2=False):
        """
            returns result of checking a pair of angles to satisfy the overlap condition TRUE, False
                phi_1, phi_2 -- float/arrays, angles
                get_a1_a2 -- bool, if TRUE return a1, a2 (local sag arrow), else pass
        """
        phi_1, phi_2 = np.asarray(phi_1), np.asarray(phi_2)
        dx = self.sys_p['dx']
        dy = self.sys_p['dy']
        b1 = self.sys_p['b1']
        b2 = self.sys_p['b2']

        # Check sizes
        assert phi_1.shape == phi_2.shape, 'The dimensions of the arguments phi_1, phi_2 are not the same'

        if dy != 0:
            a2 = dy / (-(b1 / b2) * np.cos(phi_1) + np.cos(phi_2))
        elif dx != 0:
            a2 = dx / ((b1 / b2) * np.sin(phi_1) - np.sin(phi_2))

        a1 = (b1 / b2) * a2
        # check phys correction
        check_true = np.all(np.hstack((a1 <= b1, a2 <= b2,
                                       a1 > 0, a2 > 0)), axis=-1)
        if get_a1_a2:
            return check_true, a1[check_true], a2[check_true]
        else:
            return check_true

    def _windforce_(self, mode, windload):
        '''
            Return value of wind force by windload:
                windload (W) -- float, load in [Pa]
                mode -- int, indicator of angle (1, 2)
        '''
        if mode == 1:
            D = self.sys_p['D1']
        elif mode == 2:
            D = self.sys_p['D2']
        # drag coef. that takes into account wire diameter value
        Cx = lambda x: 1.1 if (x >= 20e-3) else 1.2
        # check if need to return simple form
        if self.wF_type == 'custom':
            return Cx(D) * D * windload
        elif self.wF_type == 'pue':
            # coefficient of change in wind pressure in height,
            # depending on the type of terrain
            if self.area == 'open':
                kW = 1
            elif self.area == 'suburban':
                kW = 0.65
            # retrun wind force by PUE
            return (fun_alphaW(windload) * fun_kL(self.sys_p['H']) * kW) * Cx(D) * D * windload

    def _ode_for_singe_angle_(self, y, t, mode, g=9.81):
        """
            returns the values of the derivatives in the system of one angle:
                t -- float, time
                y -- array of float, vector with angles and their derivative
                mode -- int, indicator of angle (1, 2)
        """
        # initialization for phi and phi'
        D_phi, phi = y

        if mode == 1:
            try:
                b = self.sys_p['b1']
            except Exception:
                sys.exit('You did not specify the parameter b1')
            mu = self.sys_p['mu1']
            # wind velocity
            Vw = self.sys_p['Vm1'] * (1 + self.sys_p['eta'] *
                                      np.cos(2 * np.pi * t / self.sys_p['Tn']))
        elif mode == 2:
            try:
                b = self.sys_p['b2']
            except Exception:
                sys.exit('You did not specify the parameter b2')
            mu = self.sys_p['mu2']
            # wind velocity
            Vw = self.sys_p['Vm2'] * (1 + self.sys_p['eta'] *
                                      np.cos(2 * np.pi * t / self.sys_p['Tn'] + self.sys_p['Ph']))

        # let's introduce wind load
        wload = np.sqrt(Vw ** 2 - (4 / 3) * Vw * b * D_phi * np.cos(phi) + (4 / 9) * b ** 2 * D_phi ** 2)
        wload *= self.sys_p['rho'] * 0.5 * (Vw * np.cos(phi) - (2 / 3) * b * D_phi)
        # differential equations system
        dphi_dtdt = 1.25 * (- (g / b) * np.sin(phi) +
                            1 / (mu * b) * self._windforce_(mode, wload))
        dphi_dt = D_phi
        return [dphi_dtdt, dphi_dt]

    def _sys_odeEq_(self, t, y):
        """
            returns the values of the derivatives in the joint system of equations:
                t -- float, time
                y -- array of float, vector with angles and their derivative (D_phi1, phi1, D_phi2, phi2)
        """
        # initialization variables for phi1, phi2 and their derivatives
        D_phi1, phi1, D_phi2, phi2 = y
        # complete system of differential equations for ---phi1---
        dphi1_dtdt, dphi1_dt = self._ode_for_singe_angle_([D_phi1, phi1], t, mode=1)
        # complete system of differential equations for ---phi2---
        dphi2_dtdt, dphi2_dt = self._ode_for_singe_angle_([D_phi2, phi2], t, mode=2)

        return [dphi1_dtdt, dphi1_dt,
                dphi2_dtdt, dphi2_dt]

    def _stop_conditional_(self, t, y, atol=2e-2):
        """
            returns result of checking a pair of angles to satisfy the overlap condition
                t -- float, time
                y -- array of float, vector with angles and their derivative (D_phi1, phi1, D_phi2, phi2)
                atol -- float, absolute tollerance
        """
        # getting the angles phi1, phi2 from the solution vector:
        _, phi1, _, phi2 = y
        dx = self.sys_p['dx']
        dy = self.sys_p['dy']
        try:
            b1 = self.sys_p['b1']
            b2 = self.sys_p['b2']
        except Exception:
            sys.exit('You did not specify the parameter b1 or/and b2')

        phi_2_forwardF = self.forward_F_state(phi1)
        phi_1_backwardF = self.backward_F_state(phi2)

        check_any_possible_overlap = ~np.all(np.isnan(phi_2_forwardF)) | ~np.all(np.isnan(phi_1_backwardF))

        if check_any_possible_overlap:
            ''' analytic overlap condition based on F_state (forward and backward)'''
            # we check whether there is an overlap and leave the physical solutions
            phi_1_backwardF = phi_1_backwardF[self.phys_check(phi_1=phi_1_backwardF,
                                                              phi_2=phi2 * np.ones(phi_1_backwardF.shape))]
            phi_2_forwardF = phi_2_forwardF[self.phys_check(phi_1=phi1 * np.ones(phi_2_forwardF.shape),
                                                            phi_2=phi_2_forwardF)]
            # finding the residuals (analitic and num vals.) for phi_1
            if phi_1_backwardF.size == 1:
                diff_1 = np.abs(phi1 - phi_1_backwardF) <= atol
            elif phi_1_backwardF.size == 2:
                diff_1 = np.any(np.abs(phi1 - phi_1_backwardF) <= atol)
            else:
                diff_1 = False
            # finding the residuals for phi_2
            if phi_2_forwardF.size == 1:
                diff_2 = np.abs(phi2 - phi_2_forwardF) <= atol
            elif phi_2_forwardF.size == 2:
                diff_2 = np.any(np.abs(phi2 - phi_2_forwardF) <= atol)
            else:
                diff_2 = False
            # we handle all cases of overlap registration using the functions F_state
            if diff_1 or diff_2:
                return 0.
            else:
                return 1.
        # if no overlap
        else:
            return 1.

    def find_sol(self, find_overlap=True, atol=1e-15, rtol=1e-15, report=False):
        """
            solving the overlap problem:
                find_overlap -- bool, if 'True' so solver will find overlap, if 'False' so solver will find ode solution only
                atol -- float, absolute tolerance of num. solution
                rtol -- float, relative tolerance of num. solution
                report -- bool, need or no info about solution
        """
        t_range = [0., self.sys_p['t_end']]  # time range
        t = np.arange(t_range[0], t_range[1],
                      step=self.sys_p['Tn'] / 2e4)  # build time solution grid
        if find_overlap:
            stop_fun = lambda t, y: self._stop_conditional_(t, y)  # rebuild stop function to switch on terminal
            stop_fun.terminal = True  # whether to terminate integration if this event occurs.
            # num integration ode: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
            self.sol = solve_ivp(fun=self._sys_odeEq_,
                                 t_span=t_range,  # time range
                                 y0=[0, 0, 0, 0],  # initial conditions
                                 t_eval=t,  # time solution grid
                                 events=stop_fun,  # use stop condition overlap
                                 method='LSODA',  # OdeSolver, # method BDF
                                 atol=atol,
                                 rtol=rtol)
        else:
            self.sol = solve_ivp(fun=self._sys_odeEq_,
                                 t_span=t_range,  # time range
                                 y0=[0, 0, 0, 0],  # initial conditions
                                 t_eval=t,  # time solution grid
                                 method='LSODA',  # OdeSolver, # method BDF
                                 atol=atol,
                                 rtol=rtol)
        if report:
            if self.sol.t_events[0].size != 0:
                y_ev = np.copy(self.sol.y_events[0][0])
                print('============Report============\
                      \nOverlap: True\
                      \n-->y_event: \
                      \nDphi_1 = {Dphi_1:0.4f}; Dphi_2 = {Dphi_2:0.4f}; phi_1 = {phi_1:0.4f}; phi_2 = {phi_2:0.4f}\
                      \n-->t_event: t = {t:0.4f}'.format(Dphi_1=y_ev[0], phi_1=y_ev[1], Dphi_2=y_ev[2],
                                                         phi_2=y_ev[3], t=self.sol.t_events[0][0]))
            else:
                print('============Report============\
                      \nOverlap: False')

    def get_data(self):
        """returns the data of parameters"""
        return self.sys_p

    def get_solution(self):
        """returns the solution of overlap problem"""
        return self.sol

    def get_overlap_res(self):
        """returns the overlap indicator"""
        return bool(self.sol.t_events[0].size)