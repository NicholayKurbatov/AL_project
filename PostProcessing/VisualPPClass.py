# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from SolveClass import Solve_AL_overlap
from PostProcessing.DataPPClass import DataResClass


# ==============================================================================
# Subclass that visualized the desired values
# ==============================================================================
class VisualResClass(DataResClass, Solve_AL_overlap):

    def __init__(self, Solve_AL_overlap):
        super(VisualResClass, self).__init__(Solve_AL_overlap)

    def _matrix_minDist_plot_(self, Phi_1, Phi_2, ax, num=100):
        '''
            let's start a method that outputs the matrix (cols -- phi1, rows -- phi2)
            matrix element -- minimum distance between wires at fixed phi_1 and phi_2:
                Phi_1 -- array, values for phi1
                Phi_2 -- array, values for phi2
                ax -- object of plot class, axis
                num -- int, counts cols and rows
        '''
        phi_1 = np.linspace(np.min(Phi_1) - np.pi / 10, np.max(Phi_1) + np.pi / 10, num=num)
        phi_2 = np.linspace(np.min(Phi_2) - np.pi / 10, np.max(Phi_2) + np.pi / 10, num=num)
        min_Dist = np.zeros((phi_2.shape[0], phi_1.shape[0]))
        for p in range(len(phi_2)):
            for k in range(len(phi_1)):
                r1 = super()._current_SagArrow_at_z_(self.sys_p['b1'])
                x1 = r1 * np.sin(phi_1[k])
                y1 = -r1 * np.cos(phi_1[k])

                r2 = super()._current_SagArrow_at_z_(self.sys_p['b2'])
                x2 = self.sys_p['dx'] + r2 * np.sin(phi_2[p])
                y2 = self.sys_p['dy'] - r2 * np.cos(phi_2[p])

                min_Dist[p][k] = np.sqrt(np.min((x1 - x2) ** 2 + (y1 - y2) ** 2))

        # plotting
        for i in range(len(ax)):
            im = ax[i].contourf(phi_1, phi_2, min_Dist)
            plt.colorbar(im, ax=ax[i])
            ax[i].set_xlim([phi_1[0], phi_1[-1]])
            ax[i].set_ylim([phi_2[0], phi_2[-1]])
            ax[i].set_xlabel('$\phi_1$'), ax[i].set_ylabel('$\phi_2$')

    def _physCorr_Uncorr_val_plot_(self, phi_1, phi_2, ax, labels=None):
        '''
            a method that draws physically correct and incorrect F_state function values:
                phi_1 -- array, values for phi1
                phi_1 -- array, values for phi2
                ax -- object of plot class, axis
                labels -- list of str
        '''
        # let's highlight physically correct solutions
        check_true = [super(VisualResClass, self).phys_check(i, j) for i, j in zip(phi_1, phi_2)]
        check_false = list(map(lambda x: not (x), check_true))

        # plotting analitic solution
        if not (labels):
            for check, color in zip((check_true, check_false), ['w', 'r']):
                ax.scatter(phi_1[check], phi_2[check],
                           c=color, alpha=0.9)
        else:
            for check, color, label in zip((check_true, check_false), ['w', 'r'], labels):
                ax.scatter(phi_1[check], phi_2[check],
                           c=color, alpha=0.9, label=label)

    def _F_state_plots_with_labels_(self, ax):
        '''
            a method that create figure with all need plots for visualized solution and F_state function:
                ax -- object of plot class, axis
        '''
        # init
        phi_1, phi_2 = super(VisualResClass, self).get_Phi()

        # make dist matrix
        self._matrix_minDist_plot_(phi_1, phi_2, ax)

        # analitic solition phi_2 by forward F_state
        phi_2_an_n, phi_2_an_p = super().forward_F_state(phi_1)
        # analitic solition phi_1 by backward F_state
        phi_1_an_n, phi_1_an_p = super().backward_F_state(phi_2)

        # let's highlight physically correct solutions
        # plotting analitic solution for phi_2
        self._physCorr_Uncorr_val_plot_(
            phi_1, phi_2_an_n, ax[0], labels=['phys. correct val. F_state', 'phys. uncorrect val. F_state']
        )
        self._physCorr_Uncorr_val_plot_(phi_1, phi_2_an_p, ax[0])
        # plotting analitic solution for phi_1
        self._physCorr_Uncorr_val_plot_(
            phi_1_an_n, phi_2, ax[1], labels=['phys. correct val. F_state', 'phys. uncorrect val. F_state']
        )
        self._physCorr_Uncorr_val_plot_(phi_1_an_p, phi_2, ax[1])

        # labels and both graphs
        for i, title in enumerate(['by forward F_state', 'by backward F_state']):
            ax[i].set_title(title)
            ax[i].grid(True, linestyle='--')
            ax[i].legend(fontsize=12, loc='upper left', facecolor='tan', edgecolor='k')
            for item in ([ax[i].xaxis.label, ax[i].yaxis.label] +
                         ax[i].get_xticklabels() + ax[i].get_yticklabels()):
                item.set_fontsize(12)

    def PhaseTrContour_plot(self, F_state_plot=False, title=None, ax=None):
        '''
            a method that create figure with all need plots for visualized solution with/without F_state function:
                F_state_plot -- bool, if TRUE draw F_state values, esle pass
                title -- str or None, optional
                ax -- object of plot class, axis
        '''
        # init
        phi_1, phi_2 = super(VisualResClass, self).get_Phi()

        # check axis
        if not (ax):
            if F_state_plot:
                fig, ax = plt.subplots(ncols=2, figsize=(16, 6), facecolor='w')
            else:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')
        # check if need plot F_state function
        if F_state_plot:
            # ploting all need
            self._F_state_plots_with_labels_(ax)
            # overlap point if exist
            for Ax in ax:
                Ax.scatter(phi_1, phi_2, marker='.', c='k', alpha=0.9, label='by solution')
                if self.overlap:
                    Ax.scatter(phi_1[-1], phi_2[-1],
                               s=90, lw=2.5, c='gold', edgecolor='k', alpha=0.9, label='overlap')
        # if not need plot F_state function
        else:
            # make dist matrix
            self._matrix_minDist_plot_(self, phi_1, phi_2, ax)
            # solution graph
            ax.scatter(phi_1, phi_2, marker='.', c='k', alpha=0.9, label='by solution')
            # overlap point if exist
            if self.overlap:
                ax.scatter(phi_1[-1], phi_2[-1],
                           s=90, lw=2.5, c='gold', edgecolor='k', alpha=0.9, label='overlap')
            # grid and etc.
            ax.grid(True, linestyle='--')
            ax.legend(fontsize=12, loc='upper left', facecolor='tan', edgecolor='k')
            # title
            if not (not (title)):
                ax.set_title(title)
            else:
                ax.set_title('phase trajectory of the solution')

            for item in ([ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)

        return fig

    def _wire_curves_(self, phi1, phi2, get_a1_a2=False):
        '''
            a method that return some arrays witch discribe wire curves:
                phi1 -- array, solving values for angle 1
                phi2 -- array, solving values for angle 2
                get_a1_a2 -- bool, if TRUE return local sag boom, else pass
        '''
        z = np.linspace(0, self.sys_p['H'] / 2, num=50)
        z = np.concatenate((-z[::-1][:-1], z))
        a1 = np.abs(self.sys_p['b1'] * ((2 * z / self.sys_p['H']) ** 2 - 1))
        a2 = np.abs(self.sys_p['b2'] * ((2 * z / self.sys_p['H']) ** 2 - 1))

        x1 = a1 * np.sin(phi1)
        y1 = -a1 * np.cos(phi1)

        x2 = self.sys_p['dx'] + a2 * np.sin(phi2)
        y2 = self.sys_p['dy'] - a2 * np.cos(phi2)

        if get_a1_a2:
            return z, x1, x2, y1, y2, a1, a2
        else:
            return z, x1, x2, y1, y2

    def _find_nearest_(self, array, value):
        """a method that index nearest element in array to value"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _animate3D_(self, i, ax):
        """
            a method witch animate the 3D motion by wires:
                i -- int, index of image
                ax -- object of plot class, axis
        """
        # data
        dx, dy = self.sys_p['dx'], self.sys_p['dy']
        b1, b2 = self.sys_p['b1'], self.sys_p['b2']
        H = self.sys_p['H']
        # axis
        ax.clear()
        ax.set_xlim(-b1, dx + b2)
        ax.set_ylim(-H / 2, H / 2)
        ax.set_zlim(min(-b1, -(b2 - dy)), max(b1, (b2 - dy)))
        # plots
        if (i == len(self.phi_1) - 1) and self.overlap:
            z, x1, x2, y1, y2, a1, a2 = self._wire_curves_(self.phi_1[i], self.phi_2[i], get_a1_a2=True)
            _, over_a1, over_a2 = super().phys_check(self.phi_1[i],
                                                     self.phi_2[i],
                                                     get_a1_a2=True)
            idx = self._find_nearest_(a1, over_a1)
            over_x, over_y = x1[idx], y1[idx]
            ax.scatter([over_x] * 2, [z[idx], -z[idx]], [over_y] * 2, s=80, c='gold', edgecolor='k', label='overlap')

        else:
            z, x1, x2, y1, y2 = self._wire_curves_(self.phi_1[i], self.phi_2[i])

        # plots
        ax.plot(x1, z, y1, c='b', lw=2.5, label='#1')
        ax.scatter([0, 0], [-H / 2, H / 2], [0, 0], c='b')
        ax.plot(x2, z, y2, c='g', lw=2.5, label='#2')
        ax.scatter([dx, dx], [-H / 2, H / 2], [dy, dy], c='g')

        # settings
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('z', fontsize=12)
        ax.set_zlabel('y', fontsize=12)
        ax.grid()
        ax.set_title('3d view, time = %.2f[s]' % self.t[i], fontsize=12)
        ax.legend(fontsize=12)
        ax.view_init(elev=20, azim=-50)

    def _animate_(self, i, fig):
        """
            a method witch animate the motion by wires in xy, yz plates:
                i -- int, index of image
                fig -- object of plot class, figure
        """
        # data
        dx, dy = self.sys_p['dx'], self.sys_p['dy']
        b1, b2 = self.sys_p['b1'], self.sys_p['b2']
        H = self.sys_p['H']
        # axis
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax = [ax1, ax2]
        # init
        for j in ax:
            j.clear()
        # limits
        ax[0].set_xlim(-b1, dx + b2)
        ax[0].set_ylim(min(-b1, -(b2 - dy)), max(b1, (b2 - dy)))
        ax[0].set_title('side view, time = %.2f[s]' % self.t[i], fontsize=12)
        ax[1].set_xlim(-H / 2 - 5, H / 2 + 5)
        ax[1].set_ylim(-b1, dx + b2)
        ax[1].set_title('view from above, time = %.2f[s]' % self.t[i], fontsize=12)

        # plots
        if (i == len(self.phi_1) - 1) and self.overlap:
            z, x1, x2, y1, y2, a1, a2 = self._wire_curves_(self.phi_1[i], self.phi_2[i], get_a1_a2=True)
            _, over_a1, over_a2 = super().phys_check(self.phi_1[i],
                                                     self.phi_2[i],
                                                     get_a1_a2=True)
            idx = self._find_nearest_(a1, over_a1)
            over_x, over_y = x1[idx], y1[idx]

            ax[0].scatter(over_x, over_y, s=80, c='gold', edgecolor='k', label='overlap')
            ax[1].scatter([z[idx], -z[idx]], [over_x] * 2, s=80, c='gold', edgecolor='k')

        else:
            z, x1, x2, y1, y2 = self._wire_curves_(self.phi_1[i], self.phi_2[i])

        # plots
        ax[0].plot(x1, y1, c='b', lw=2.5, label='#1')
        ax[0].plot(x2, y2, c='g', lw=2.5, label='#2')
        ax[0].scatter(0, 0, c='b')
        ax[0].scatter(dx, dy, c='g')

        ax[1].plot(z, x1, c='b', lw=2.5, label='#1')
        ax[1].plot(z, x2, c='g', lw=2.5, label='#2')
        ax[1].scatter([-H / 2, H / 2], [0, 0], c='b')
        ax[1].scatter([-H / 2, H / 2], [dx, dx], c='g')

        # settings
        ax[0].set_xlabel('x', fontsize=12)
        ax[0].set_ylabel('y', fontsize=12)
        ax[1].set_xlabel('z', fontsize=12)
        ax[1].set_ylabel('x', fontsize=12)
        ax[0].grid(), ax[1].grid()
        ax[0].legend(fontsize=12)

    def Animation_wire_motion(self, fp, im_count_ratio=1, fps=500, plot3D=False):
        """
            a method witch animate the 3D/2D motion by wires and save video in fp:
                fp -- str, file path to save
                plot3D -- bool, if TRUE animate only 3D motion, else only 2D
        """
        im_num = len(self.phi_1) // im_count_ratio
        new_ind = np.linspace(0, len(self.phi_1) - 1, num=im_num, dtype=int)
        # plotting
        if plot3D:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca(projection='3d')
            Animate = lambda i: self._animate3D_(new_ind[i], ax)
        else:
            fig = plt.figure(figsize=(14, 6))
            Animate = lambda i: self._animate_(new_ind[i], fig)

        anim = FuncAnimation(fig, Animate,
                             frames=im_num, interval=0.05)

        writer = PillowWriter(fps=fps)
        anim.save(fp, writer=writer)