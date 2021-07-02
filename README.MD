**Project for numerical simulation of pendulum oscillations of power line wires**

## Description

The project consists of several classes. Using which you can simulate the behavior of two wires for a specific wind impact.
You can also describe the accident rate of a certain area of the network. And there are some tools for PostProcessing work.

<p align="center">
<img src="https://github.com/UnavailableUser/Power_line_oscillations/blob/main/AL_Overlap_project/readme_assets/overlap_tr.png" width="75%"></p>

## Project content

1. The class that simulates the overlap of two wires in a single span of overhead lines -- `SolveClass.py`.

This is the solver class. By feeding the span data to the input, you can simulate the behavior of the wires. Main public methods:<br />
`Solve_AL_overlap.find_sol()` <br />
`Solve_AL_overlap.get_data()` <br />
`Solve_AL_overlap.get_solution()` <br />
`Solve_AL_overlap.get_overlap_res()` <br />

2. The PostProcessing class that counts all the necessary values -- `PostProcessing.DataPPClass.py`.
This subclass calculates: 
-- angles `DataResClass.get_Phi()`,<br />
-- angular velocities and accelerations `DataResClass.get_DPhi(), DataResClass.get_DDPhi()`,<br />
-- minimum distance between wires `DataResClass.get_WireToWireDistance()`,<br />
-- moments of wind forces acting on the wires `DataResClass.get_WindMomentum()`<br />
   
It can also return all data as a csv file `DataResClass.ResData()`
   
3. The PostProcessing class that renders graphics and creates animations -- `PostProcessing.VisualPPClass.py`

This subclass draws the phase paths on a contour plot of the minimum distance between the wires. 
Optionally draws the values of the forward and backward overlap state function: `VisualResClass.PhaseTrContour_plot()`

Method for creating 2D and 3D wire motion animation `VisualResClass.Animation_wire_motion()`.

<p align="center">
<img src="https://github.com/UnavailableUser/Power_line_oscillations/blob/main/AL_Overlap_project/readme_assets/checkAnim_2D.gif" width="75%"></p>

<p align="center">
<img src="https://github.com/UnavailableUser/Power_line_oscillations/blob/main/AL_Overlap_project/readme_assets/checkAnim_3D.gif" width="50%"></p>

4. Subclass that calculates the critical values of the boom sags of the overhead wires -- `CriticalValuesClass.py`

Method for finding critical values of slack arrows: `SagBoomCrValClass.find_CrVal()`

Method for plotting critical values of slack arrows: `SagBoomCrValClass.CrVal_plot()`
