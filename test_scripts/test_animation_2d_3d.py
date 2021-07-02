from SolveClass import Solve_AL_overlap
from PostProcessing.VisualPPClass import VisualResClass


def main():
        config_data = {'dx': 1.48, 'dy': 0, 'H': 60, 'L1': float('nan'), 'L2': float('nan'),
                       'b1': 2, 'b2': 1, 'mu1': 0.194, 'mu2': 0.194,
                       'D1': 9.6e-3, 'D2': 9.6e-3, 'Cx1': 1, 'Cx2': 1,
                       'Vm1': 18, 'eta1': 0.5, 'Tn1': 10,
                       'Vm2': 19, 'eta2': 0.6, 'Tn2': 11, 'Ph':0.5,
                       'rho': 1.28, 't_end': 10}
        solve_ex = Solve_AL_overlap(config_data)
        solve_ex.find_sol(report=True)

        vis_cl = VisualResClass(solve_ex)
        vis_cl.Animation_wire_motion('readme_assets/checkAnim_2D.gif',
                                     im_count_ratio=50, fps=1000, plot3D=False)
        vis_cl.Animation_wire_motion('readme_assets/checkAnim_3D.gif',
                                     im_count_ratio=50, fps=1000, plot3D=True)
