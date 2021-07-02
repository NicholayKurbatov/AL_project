import numpy as np
import matplotlib.pyplot as plt
import json

from СriticalValuesClass import SagBoomCrValClass
from Func_using_in_SolveClass.StatTests_prop_diff import *


def calc_CrVal_areas(config_data, area_dep_data,
                     wind_force_type, fp_class, symm=None):
    '''
        Return CritVals in dict[datadict, SagBoomCrValClass object] for diffenet areas:
            config_data -- dict/DataFrame/csv configuration data without slack arrow (b)
            area_dep_data -- dict['area_name': dict['eta': float, <-- necessary
                                                    'b': list/array, <-- necessary
                                                    'H': float]], area_name: open/suburban,
            wind_force_type -- str, if 'custom' F/H = Cx * WindLoad * D (simple form)
                                    else 'pue' F/H = (sime COEF.) * Cx * WindLoad * D (form by PUE)
            fp_class -- str, file name where would be saved CrVal dict for each area
            symm -- bool, if TRUE (so the system has horizontal or vertical symmetry)
                             solve only for (b1, b2) with Vm>0,
                          else solve for (b1, b2) with Vm>0 and (b2, b1) with Vm<0
    '''
    CrVal_results = {}
    for area_type in area_dep_data.keys():
        config_data['eta'] = area_dep_data[area_type]['eta']
        if 'H' in area_dep_data[area_type].keys():
            config_data['H'] = area_dep_data[area_type]['H']
        CrVar = SagBoomCrValClass(config_data, area_dep_data[area_type]['b'],
                                      area_type, wind_force_type, symm=symm)
        open_fp = f'{fp_class}_{area_type}.json'
        data = CrVar.find_CrVal(get_dict=True, fp_to_savejson=open_fp)
        CrVal_results[area_type] = {'data': data,
                                    'object': CrVar}
        print(f'\nalready calc for {area_type} area type')
    return CrVal_results


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)


def comparingCrVal_plot(config_data, give_sourse, ax=None):
    '''
        Return fig (if ax=None) and dict with data when overvap is True
            config_data -- dict/DataFrame/csv configuration data without slack arrow (b)
            give_sourse -- dict/list[str], if dict: num data after modeling in dict
                                           if list[str]: file names with num data after modeling
                                           (Ex: ['filepth ../eta_open.json', 'filepth ../eta_suburb'])
            ax -- axis, if not None
    '''
    dx = config_data['dx']
    dy = config_data['dy']
    T = config_data['Tn']
    Vm = config_data['Vm1']
    # create/make dict with all info about overlaps
    plot_data = {}
    if isinstance(give_sourse, dict):
        for eta_key in give_sourse.keys():
            plot_pos_b1 = []
            plot_pos_b2 = []
            plot_neg_b1 = []
            plot_neg_b2 = []
            data = give_sourse[eta_key]['data']
            for i in range(len(data['b1'])):
                if (data['dir_Vm'][i] == '+') & data['overlap'][i]:
                  plot_pos_b1.append(data['b1'][i])
                  plot_pos_b2.append(data['b2'][i])
                elif (data['dir_Vm'][i] == '-') & data['overlap'][i]:
                  plot_neg_b1.append(data['b1'][i])
                  plot_neg_b2.append(data['b2'][i])
            plot_data[eta_key] = {'pos_b1': plot_pos_b1, 'pos_b2': plot_pos_b2,
                                  'neg_b1': plot_neg_b1, 'neg_b2': plot_neg_b2}

    elif all(isinstance(elem, str) for elem in give_sourse):
        dict_data = []
        for s, i in enumerate(give_sourse):
            with open(s) as f:
                dict_data += [json.load(f)]

        for data, eta_key in zip(dict_data, ['eta_open', 'eta_suburb']):
            plot_pos_b1 = []
            plot_pos_b2 = []
            plot_neg_b1 = []
            plot_neg_b2 = []
            for i in range(len(data['b1'])):
                if (data['dir_Vm'][i] == '+') & data['overlap'][i]:
                  plot_pos_b1.append(data['b1'][i])
                  plot_pos_b2.append(data['b2'][i])
                elif (data['dir_Vm'][i] == '-') & data['overlap'][i]:
                  plot_neg_b1.append(data['b1'][i])
                  plot_neg_b2.append(data['b2'][i])
            plot_data[eta_key] = {'pos_b1': plot_pos_b1, 'pos_b2': plot_pos_b2,
                                  'neg_b1': plot_neg_b1, 'neg_b2': plot_neg_b2}


    # plotting CrVal for different areas together (on the same figure)
    if not(ax):
        fig, ax = plt.subplots(ncols=2, facecolor='w', figsize=(18, 8))
    markers = ['x', '+']
    colors = ['r', '#330C73']
    titles = [f'for $\Delta$x={dx}[m], $\Delta$y={dy}[m], V_m1~{+Vm}[m/s]',
              f'for $\Delta$x={dx}[m], $\Delta$y={dy}[m], V_m1~{-Vm}[m/s]']
    for eta_key, m, color in zip(['suburban', 'open'], markers, colors):
        ax[0].scatter(plot_data[eta_key]['pos_b1'], plot_data[eta_key]['pos_b2'],
                      marker=m, c=color, s=40, lw=3, label=eta_key)
        ax[1].scatter(plot_data[eta_key]['neg_b1'], plot_data[eta_key]['neg_b2'],
                      marker=m, c=color, s=40, lw=3, label=eta_key)
    for i in range(2):
        ax[i].set_ylim([np.min(give_sourse[eta_key]['data']['b1'])-0.1,
                        np.max(give_sourse[eta_key]['data']['b1'])+0.1])
        ax[i].legend()
        ax[i].set_xlabel('b1'), ax[i].set_ylabel('b2')
        ax[i].grid(True, linestyle='--')
        ax[i].set_title(titles[i])

    return fig, plot_data


def comparingCrVal_barplot(config_data, plot_data, area_dep_data, ax=None):
    '''
        Return figure (if ax=None) with compare barplot:
            config_data -- dict/DataFrame/csv configuration data without slack arrow (b)
            plot_data -- dict, data when overvap is True
            area_dep_data -- dict['area_name': dict['eta': float, <-- necessary
                                                    'b': list/array, <-- necessary
                                                    'H': float]], area_name: open/suburban,
            ax -- axis, if not None
    '''
    dx = config_data['dx']
    dy = config_data['dy']
    T = config_data['Tn']
    Vm = config_data['Vm1']
    # counting all pairs (b1, b2) and all finding overlaps
    N_all_pairs_op = len(area_dep_data['open']['b']) * (len(area_dep_data['open']['b']) + 1)
    N_all_pairs_sub = len(area_dep_data['suburban']['b']) * (len(area_dep_data['suburban']['b']) + 1)
    over_suburb = len(plot_data['suburban']['pos_b1']) + len(plot_data['suburban']['neg_b1'])
    over_open = len(plot_data['open']['pos_b1']) + len(plot_data['open']['neg_b1'])

    # bar plot % overlaps in each groups
    x = np.arange(1)  # the label locations
    width = 0.25  # the width of the bars
    if not(ax):
        fig, ax = plt.subplots(figsize=(6, 5))
    rects1 = ax.bar(x - 2 / 3 * width,
                    np.round(100 * over_suburb / N_all_pairs_sub, 2),
                    width, alpha=0.75, zorder=2.5, color='#EB89B5')
    rects2 = ax.bar(x + 2 / 3 * width,
                    np.round(100 * over_open / N_all_pairs_op, 2),
                    width, alpha=0.75, zorder=2.5, color='#330C73')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('overlaps \npercentage, %', rotation='horizontal', labelpad=50,
                  fontsize=12, color='k')
    ax.set_xlim([x - 2 * width, x + 2 * width])
    ax.set_ylim([0, 1.1 * rects1[0].get_height()])
    ax.grid(linestyle='--', zorder=1.5)
    ax.set_xticks([x - 2 * width / 3, x + 2 * width / 3])
    ax.set_xticklabels(['open area', 'suburban'], fontsize=12)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    ax.set_title(f'for $\Delta$x={dx}[m], $\Delta$y={dy}[m], V_m1~{Vm}[m/s], T={T}[s]')
    fig.tight_layout()

    return fig


def analisys_equalSagArr(config_data, eta_open, eta_suburb, b,
                         wind_force_type, fp_class, symm=True):
    '''
        Return all need info about overlaps count in each of two areas:
            config_data -- dict/DataFrame/csv configuration data without slack arrow (b)
            eta_open -- float, coef eta in open area
            eta_suburb -- float, coef eta in suburban area
            b -- list/array of float, sag boom arrow
            wind_force_type -- str, if 'custom' F/H = Cx * WindLoad * D (simple form)
                                    else 'pue' F/H = (sime COEF.) * Cx * WindLoad * D (form by PUE)
            fp_class -- str, file name where would be saved CrVal dict for each area
            symm -- bool, if you have symmetry in your geometry (if dy == 0)
    '''
    # create area_dep_data dict and start solve
    area_dep_data = {'open': {'eta': eta_open, 'b': b},
                     'suburban': {'eta': eta_suburb, 'b': b}}
    CrVal = calc_CrVal_areas(config_data, area_dep_data,
                             wind_force_type, fp_class, symm)
    # comparing plot for CrVal
    fig1, plot_data = comparingCrVal_plot(config_data, CrVal)
    fig1.show()
    # print p-val for ztest and CI
    print("p-value for ztest by diff % suburb and open: {}".format(prop_diff_ZTest(
        prop_diff_ZStat_ind(CrVal['suburban']['data']['overlap'],
                            CrVal['open']['data']['overlap']))
    ))
    print('CI :', prop_diff_CI_rel(CrVal['suburban']['data']['overlap'],
                                   CrVal['open']['data']['overlap']))
    # counting all pairs (b1, b2) and all finding overlaps
    N_all_pairs = len(b) * (len(b) + 1)
    over_suburb = len(plot_data['suburban']['pos_b1']) + len(plot_data['suburban']['neg_b1'])
    over_open = len(plot_data['open']['pos_b1']) + len(plot_data['open']['neg_b1'])
    print('num all pairs: %i\
          \nnum overlaps in suburb: %i\
          \nnum overlaps in open_area: %i' % (N_all_pairs, over_suburb, over_open))
    # bar plot % overlaps in each groups
    fig2 = comparingCrVal_barplot(plot_data, area_dep_data)
    fig2.show()

    return CrVal


def analisys_diffSagArr(config_data, eta_open, eta_suburb,
                        b_open, b_suburb, H_open, H_suburb,
                        wind_force_type, fp_class, symm=True):
    '''
        Return all need info about overlaps count in each of two areas:
            config_data -- dict/DataFrame/csv configuration data without slack arrow (b)
            eta_open -- float, coef eta in open area
            eta_suburb -- float, coef eta in suburban area
            b_open -- list/array of float, sag boom arrow in open area
            b_suburb -- list/array of float, sag boom arrow in suburban area
            H_open -- float, wire distance in open area
            H_suburb -- float, wire distance in suburban area
            wind_force_type -- str, if 'custom' F/H = Cx * WindLoad * D (simple form)
                                    else 'pue' F/H = (sime COEF.) * Cx * WindLoad * D (form by PUE)
            fp_class -- str, file name where would be saved CrVal dict for each area
            symm -- bool, if you have symmetry in your geometry (if dy == 0)
    '''
    # create area_dep_data dict and start solve
    area_dep_data = {'open': {'eta': eta_open, 'b': b_open, 'H': H_open},
                     'suburban': {'eta': eta_suburb, 'b': b_suburb, 'H': H_suburb}}
    CrVal = calc_CrVal_areas(config_data, area_dep_data,
                             wind_force_type, fp_class, symm)
    # comparing plot for CrVal
    fig1, plot_data = comparingCrVal_plot(config_data, CrVal)
    fig1.show()
    # print p-val for ztest and CI
    print("p-value for ztest by diff % suburb and open: {}".format(prop_diff_ZTest(
        prop_diff_ZStat_rel(CrVal['suburban']['data']['overlap'],
                            CrVal['open']['data']['overlap']))
    ))
    print('CI :', prop_diff_CI_rel(CrVal['suburban']['data']['overlap'],
                                   CrVal['open']['data']['overlap']))
    # counting all pairs (b1, b2) and all finding overlaps
    N_all_pairs_op = len(area_dep_data['open']['b']) * (len(area_dep_data['open']['b']) + 1)
    N_all_pairs_sub = len(area_dep_data['suburban']['b']) * (len(area_dep_data['suburban']['b']) + 1)
    over_suburb = len(plot_data['suburban']['pos_b1']) + len(plot_data['suburban']['neg_b1'])
    over_open = len(plot_data['open']['pos_b1']) + len(plot_data['open']['neg_b1'])
    print('num all pairs in suburb: %i\
          \nnum all pairs in open: %i\
          \nnum overlaps in suburb: %i\
          \nnum overlaps in open_area: %i' % (N_all_pairs_sub, N_all_pairs_op,
                                              over_suburb, over_open))
    # bar plot % overlaps in each groups
    fig2 = comparingCrVal_barplot(plot_data, area_dep_data)
    fig2.show()

    return CrVal