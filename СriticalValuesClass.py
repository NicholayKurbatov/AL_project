# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from SolveClass import Solve_AL_overlap
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


# =========================================================================================
# Subclass that calculates the critical values of the boom sags of the overhead wires
# =========================================================================================
class SagBoomCrValClass(Solve_AL_overlap):

    def __init__(self, Data_without_b, b, terrian_type, wind_force_type='pue', symm=False):
        '''
           initialization
              Data_without_b -- dict/DataFrame/csv configuration data without slack arrow (b)
              b -- array or tuple of arrays, slack arrow
              terrian_type -- str: 'open'; 'suburban'
              wind_force_type -- str, if 'custom' F/H = Cx * WindLoad * D (simple form)
                                      else 'pue' F/H = (sime COEF.) * Cx * WindLoad * D (form by PUE)
              symm -- bool, if TRUE (so the system has horizontal or vertical symmetry)
                              solve only for (b1, b2) with Vm>0,
                              else solve for (b1, b2) with Vm>0 and (b2, b1) with Vm<0
        '''
        super(SagBoomCrValClass, self).__init__(Data_without_b, terrian_type, wind_force_type)
        self.b = b
        self.symm = symm

    def find_CrVal(self, get_dict=False, fp_to_savejson=None):
        '''
            a method for finding critical values of slack arrows
                get_dict -- bool, if TRUE return dict with finding overlap data, else pass
                fp_to_savejson -- str or None, if str so save in taken file path
        '''
        self.check_dict = {'b1': [], 'b2': [], 'overlap': [], 'dir_Vm': []}
        pos_Vm1 = abs(self.sys_p['Vm1'])
        pos_Vm2 = abs(self.sys_p['Vm2'])
        neg_Vm1 = -abs(self.sys_p['Vm1'])
        neg_Vm2 = -abs(self.sys_p['Vm1'])

        if self.symm:
            B1 = np.asarray(self.b)
            B2 = np.copy(B1)
            with tqdm(desc="num iter in outer loop (outer by b2)", total=B2.shape[0] - 1) as pbar_outer:
                for i, b2 in enumerate(B2):
                    for b1 in B1[i:]:
                        self.sys_p['b1'] = b1
                        self.sys_p['b2'] = b2
                        self.sys_p['Vm1'] = pos_Vm1
                        self.sys_p['Vm2'] = pos_Vm2

                        super().find_sol()

                        self.check_dict['b1'].append(b1), self.check_dict['b2'].append(b2), \
                        self.check_dict['dir_Vm'].append('+')
                        self.check_dict['overlap'].append(super().get_overlap_res())

                    for b1 in B1[:i + 1]:
                        self.sys_p['b1'] = b1
                        self.sys_p['b2'] = b2
                        self.sys_p['Vm1'] = neg_Vm1
                        self.sys_p['Vm2'] = neg_Vm2

                        super().find_sol()

                        self.check_dict['b1'].append(b1), self.check_dict['b2'].append(b2), \
                        self.check_dict['dir_Vm'].append('-')
                        self.check_dict['overlap'].append(super().get_overlap_res())

                    pbar_outer.update(1)
        else:
            B1 = np.asarray(self.b)
            B2 = np.copy(B1)
            with tqdm(desc="num iter in outer loop (outer by b2)", total=B2.shape[0] - 1) as pbar_outer:
                for i, b2 in enumerate(B2):
                    for b1 in B1:
                        self.sys_p['b1'] = b1
                        self.sys_p['b2'] = b2
                        self.sys_p['Vm1'] = pos_Vm1
                        self.sys_p['Vm2'] = pos_Vm2

                        super().find_sol()

                        self.check_dict['b1'].append(b1), self.check_dict['b2'].append(b2), \
                        self.check_dict['dir_Vm'].append('+')
                        self.check_dict['overlap'].append(super().get_overlap_res())

                    for b1 in B1:
                        self.sys_p['b1'] = b1
                        self.sys_p['b2'] = b2
                        self.sys_p['Vm1'] = neg_Vm1
                        self.sys_p['Vm2'] = neg_Vm2

                        super().find_sol()

                        self.check_dict['b1'].append(b1), self.check_dict['b2'].append(b2), \
                        self.check_dict['dir_Vm'].append('-')
                        self.check_dict['overlap'].append(super().get_overlap_res())

                    pbar_outer.update(1)
        if not (not (fp_to_savejson)):
            with open(fp_to_savejson, 'w') as f:
                json.dump(self.check_dict, f)
        if get_dict:
            return self.check_dict

    def CrVal_plot(self, give_dict=None, compare_plot=True):
        '''
            a method for plotting critical values of slack arrows
                give_dict -- str or None, if None so the check_dict takes from solution
                             by find_CrVal method,
                             if str so the check_dict takes from file path in give_dict
                compare_plot -- bool, TRUE plot search result graph with turned axes for direction $V_m < 0$
                                (only if slack arrow it is tuple of arrays), else pass
        '''
        if not (not (give_dict)):
            with open(give_dict) as f:
                check_dict = json.load(f)
        else:
            check_dict = self.check_dict

        b1 = np.asarray(self.b)
        b2 = np.copy(b1)
        negVm_plot_b1 = []
        negVm_plot_b2 = []
        posVm_plot_b1 = []
        posVm_plot_b2 = []
        for i in range(len(check_dict['b1'])):
            if (check_dict['dir_Vm'][i] == '-') & check_dict['overlap'][i]:
                negVm_plot_b1.append(check_dict['b1'][i])
                negVm_plot_b2.append(check_dict['b2'][i])
            elif (check_dict['dir_Vm'][i] == '+') & check_dict['overlap'][i]:
                posVm_plot_b1.append(check_dict['b1'][i])
                posVm_plot_b2.append(check_dict['b2'][i])

        if compare_plot:
            fig, ax = plt.subplots(ncols=2, facecolor='w', figsize=(18, 8))

            ax[0].scatter(posVm_plot_b1, posVm_plot_b2,
                          marker='x', c='r', s=30, lw=3, label='for direction $V_m > 0$')
            ax[0].scatter(negVm_plot_b1, negVm_plot_b2,
                          marker='.', c='#330C73', s=30, lw=2, label='for direction $V_m < 0$')
            ax[0].set_ylim([np.min(b2) - 0.1, np.max(b2) + 0.1])
            ax[0].set_xlim([np.min(b1) - 0.1, np.max(b1) + 0.1])
            ax[0].legend(loc='upper right')
            ax[0].set_title('Search result plot')
            ax[0].set_xlabel('b1'), ax[0].set_ylabel('b2')
            ax[0].grid(True, linestyle='--')

            # together
            ax[1].scatter(posVm_plot_b1, posVm_plot_b2,
                          marker='x', c='r', s=30, lw=3, label='for direction $V_m > 0$')
            ax[1].scatter(negVm_plot_b2, negVm_plot_b1,
                          marker='.', c='#330C73', s=30, lw=2, label='for direction $V_m < 0$')
            ax[1].set_ylim([np.min(b2) - 0.1, np.max(b2) + 0.1])
            ax[1].set_xlim([np.min(b1) - 0.1, np.max(b1) + 0.1])
            ax[1].legend(loc='upper right')
            ax[1].set_title('Search result plot with turned axes for direction $V_m < 0$')
            ax[1].set_xlabel('b1'), ax[1].set_ylabel('b2')
            ax[1].grid(True, linestyle='--')

        else:
            fig, ax = plt.subplots(facecolor='w', figsize=(9, 8))

            ax.scatter(negVm_plot_b1, negVm_plot_b2,
                       marker='.', c='r', s=30, label='for direction $V_m < 0$')
            ax.scatter(posVm_plot_b1, posVm_plot_b2,
                       marker='.', c='b', s=30, label='for direction $V_m > 0$')
            ax.set_ylim([np.min(b2) - 0.1, np.max(b2) + 0.1])
            ax.set_xlim([np.min(b1) - 0.1, np.max(b1) + 0.1])
            ax.legend()
            ax.set_xlabel('b1'), ax.set_ylabel('b2')
            ax.grid(True, linestyle='--')

        return fig
