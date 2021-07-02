# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from SolveClass import Solve_AL_overlap
import pandas as pd
import os


# ==============================================================================
# Subclass that calculates the desired values
# ==============================================================================
class DataResClass(Solve_AL_overlap):

    def __init__(self, Solve_AL_overlap):
        super(DataResClass, self).__init__(Solve_AL_overlap.sys_p,
                                           Solve_AL_overlap.area,
                                           Solve_AL_overlap.wF_type)
        self.t = Solve_AL_overlap.get_solution().t
        self.Dphi_1, self.phi_1, self.Dphi_2, self.phi_2 = Solve_AL_overlap.get_solution().y
        self.overlap = Solve_AL_overlap.get_overlap_res()

    def get_t(self):
        """returns time"""
        return self.t

    def get_Phi(self):
        """returns the angles phi1, phi2"""
        return (self.phi_1, self.phi_2)

    def get_DPhi(self):
        """returns the first derivatives of the angles phi1, phi2"""
        return (self.Dphi_1, self.Dphi_2)

    def get_DDPhi(self):
        """returns the second derivatives of the angles phi1, phi2"""
        DDphi_1 = np.gradient(self.Dphi_1, self.t)
        DDphi_2 = np.gradient(self.Dphi_2, self.t)
        return (DDphi_1, DDphi_2)

    def _current_SagArrow_at_z_(self, b, num=50):
        """
            returns the current sag arrow b at the z coordinate:
            b -- array, sag arrow for 1 or 2 line
            num -- int, number of point on z coordinate
        """
        z = np.linspace(0, self.sys_p['H'] / 2, num=num)
        return np.abs(b * ((2 * z / self.sys_p['H']) ** 2 - 1))

    def get_WireToWireDistance(self):
        """return minimum distance between wires at pair phi_1 and phi_2"""
        min_Dist = []
        for phi_1, phi_2 in zip(self.phi_1, self.phi_2):
            r1 = self._current_SagArrow_at_z_(self.sys_p['b1'])
            x1 = r1 * np.sin(phi_1)
            y1 = -r1 * np.cos(phi_1)

            r2 = self._current_SagArrow_at_z_(self.sys_p['b2'])
            x2 = self.sys_p['dx'] + r2 * np.sin(phi_2)
            y2 = self.sys_p['dy'] - r2 * np.cos(phi_2)
            # the square of the Euclidean distance between two vectors
            min_Dist.append(np.sqrt(np.min((x1 - x2) ** 2 + (y1 - y2) ** 2)))

        return np.asarray(min_Dist)

    def _momentum_(self, mode):
        """
            returns the wind force momentum that acts on phi1 or phi2
            mode -- int, if 1 -- momentum on phi1, if 2 -- -- momentum on phi2
        """
        H = self.sys_p['H']
        t = self.t

        if mode == 1:
            b = self.sys_p['b1']
            phi = self.phi_1
            D_phi = self.Dphi_1
            # wind velocity
            Vw = self.sys_p['Vm1'] * (1 + self.sys_p['eta'] *
                                      np.cos(2 * np.pi * t / self.sys_p['Tn']))
        elif mode == 2:
            b = self.sys_p['b2']
            phi = self.phi_2
            D_phi = self.Dphi_2
            # wind velocity
            Vw = self.sys_p['Vm2'] * (1 + self.sys_p['eta'] *
                                      np.cos(2 * np.pi * t / self.sys_p['Tn'] + self.sys_p['Ph']))

        # let's introduce wind load
        wload = np.sqrt(Vw ** 2 - (4 / 3) * Vw * b * D_phi * np.cos(phi) + (4 / 9) * b ** 2 * D_phi ** 2)
        wload *= self.sys_p['rho'] * 0.5 * (Vw * np.cos(phi) - (2 / 3) * b * D_phi)
        # wind momentum
        Mw = [(2 / 3) * b * H * super(DataResClass, self)._windforce_(mode, x) for x in wload]

        return Mw

    def get_WindMomentum(self):
        """returns the wind force momentum that acts on phi1 and phi2"""
        return [self._momentum_(mode=1), self._momentum_(mode=2)]

    def ResData(self, csv=False, filename=None):
        """
            returns the data results by numerical solution:
                csv -- bool, if TRUE return data in .csv
                filename -- str or None, if None save data in current dir, else in custom dir
        """
        result_data = {'Phi_1': self.get_Phi()[0], 'Phi_2': self.get_Phi()[1],
                       'DPhi_1': self.get_DPhi()[0], 'DPhi_2': self.get_DPhi()[1],
                       'DDPhi_1': self.get_DDPhi()[0], 'DDPhi_2': self.get_DDPhi()[1],
                       'Momentum_1': self.get_WindMomentum()[0], 'Momentum_2': self.get_WindMomentum()[1],
                       't': self.get_t(), 'wire_dist': self.get_WireToWireDistance(),
                       'overlap': self.overlap}
        if csv:
            num_of_passes = [np.nan] * (len(result_data['Phi_1']) - 1)
            result_data['overlap'] = [result_data['overlap']] + num_of_passes
            df = pd.DataFrame.from_dict(result_data)

            if not (filename):
                dirpath = os.path.dirname(__file__)
                filepath = os.path.join(dirpath, 'DataResults.csv')
                df.to_csv(filepath, index=False)
            else:
                df.to_csv(filename, index=False)

        else:
            return result_data