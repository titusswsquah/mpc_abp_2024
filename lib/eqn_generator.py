# import custompath
#
# custompath.add()

import sympy as sp
import re
import numpy as np
from types import SimpleNamespace
from lib import utils
from scipy.linalg import lstsq

real = sp.re
imag = sp.im
j = sp.I


def symbol_string_gen(letter1, letter2, n):
    if 'i' in letter2:
        symbol_string = None
        for k in range(1, n):
            if symbol_string is None:
                symbol_string = '{0}{1}{2:0=3d}'.format(letter1, letter2, k)
            else:
                symbol_string += ' {0}{1}{2:0=3d}'.format(letter1, letter2, k)
    else:
        symbol_string = '{0}{1}000'.format(letter1, letter2)
        for k in range(1, n):
            symbol_string += ' {0}{1}{2:0=3d}'.format(letter1, letter2, k)
    return symbol_string


def e_to_list(e):
    if not isinstance(e, list):
        if not isinstance(e, tuple):
            e = [e]
    return e


def expr_gen(p_n, w_n, d_n):
    ld2 = sp.Symbol('ld2', real=True)
    g1 = 0
    g2 = sp.Symbol('g2', real=True)
    lap = sp.Symbol('lap', real=True)
    if p_n > 1:
        p_re = sp.symbols(symbol_string_gen('p', 'r', p_n), real=True)
        p_im = sp.symbols(symbol_string_gen('p', 'i', p_n), real=True)
    else:
        p_re = [sp.symbols('pr000'.format(p_n), real=True)]
        p_im = ()
    p_re = e_to_list(p_re)
    p_im = e_to_list(p_im)
    p_all = [p_re[-i] + sp.I * p_im[-i] for i in range(1, p_n)]
    p_all.append(p_re[0])
    p_all.extend([p_re[i] - sp.I * p_im[i - 1] for i in range(1, p_n)])
    if w_n > 1:
        w_re = sp.symbols(symbol_string_gen('w', 'r', w_n), real=True)
        w_im = sp.symbols(symbol_string_gen('w', 'i', w_n), real=True)
    else:
        w_re = [sp.symbols('wr000', real=True)]
        w_im = ()
    w_re = e_to_list(w_re)
    w_im = e_to_list(w_im)
    w_all = [w_re[-i] + sp.I * w_im[-i] for i in range(1, w_n)]
    w_all.append(w_re[0])
    w_all.extend([w_re[i] - sp.I * w_im[i - 1] for i in range(1, w_n)])

    if d_n > 1:
        d_re = sp.symbols(symbol_string_gen('d', 'r', d_n), real=True)
        d_im = sp.symbols(symbol_string_gen('d', 'i', d_n), real=True)
    else:
        d_re = [sp.symbols('dr000', real=True)]
        d_im = ()
    d_re = e_to_list(d_re)
    d_im = e_to_list(d_im)
    d_all = [d_re[-i] + sp.I * d_im[-i] for i in range(1, d_n)]
    d_all.append(d_re[0])
    d_all.extend([d_re[i] - sp.I * d_im[i - 1] for i in range(1, d_n)])

    ode_exps = []
    for k_p in range(p_n):
        ind1 = k_p + p_n - 1
        k_pm1 = k_p - 1
        k_pp1 = k_p + 1
        if abs(k_pm1) > p_n - 1:
            p_km1 = 0 + 0 * j
        else:
            p_km1 = p_all[ind1 - 1]
        if abs(k_pp1) > p_n - 1:
            p_kp1 = 0 + 0 * j
        else:
            p_kp1 = p_all[ind1 + 1]
        term1 = sp.simplify(sp.simplify(1 / 2 * ld2 * (g1 * p_km1 +
                                                       g2 * (-j * p_km1) + g1 * p_kp1 + g2 * (j * p_kp1))).expand())

        term3 = -lap * p_all[ind1]
        term4 = 0
        for ind3 in range(2 * w_n - 1):
            k_w = ind3 - w_n + 1
            k_pmw = k_p - k_w
            if abs(k_pmw) > p_n - 1:
                p_pmw = 0 + 0 * j
            else:
                p_pmw = p_all[k_pmw + p_n - 1]
            term4 += j * k_p * p_pmw * w_all[ind3]
        term4 *= (1 / (2 * sp.pi) * ld2)
        term5 = ld2 * k_p ** 2 * p_all[ind1]
        term6 = 0
        for ind4 in range(2 * d_n - 1):
            k_d = ind4 - d_n + 1
            k_pmd = k_p - k_d
            if abs(k_pmd) > p_n - 1:
                p_pmd = 0 + 0 * j
            else:
                p_pmd = p_all[k_pmd + p_n - 1]
            term6 += k_p * (k_p - k_d) * p_pmd * d_all[ind4]
        term6 *= (1 / (2 * sp.pi) * ld2)
        ode_exps.append((term1 + term3 + term4 + term5 + term6) / -ld2)
    alg_exps = []
    for k_p in range(p_n):
        ind1 = k_p + p_n - 1
        k_pm1 = k_p - 1
        k_pp1 = k_p + 1
        p_k = p_all[ind1]
        if abs(k_pm1) > p_n - 1:
            p_km1 = 0 + 0 * j
        else:
            p_km1 = p_all[ind1 - 1]
        if abs(k_pp1) > p_n - 1:
            p_kp1 = 0 + 0 * j
        else:
            p_kp1 = p_all[ind1 + 1]
        alg_exps.append(1 / 2 * ld2 * (-j * p_km1 + j * p_kp1) - g2 * p_k)
    return ode_exps, alg_exps


def eqn_gen(p_n, w_n, d_n):
    ode_exps, alg_exps = expr_gen(p_n, w_n, d_n)
    ode_exec_strings = []
    for ind, e in enumerate(ode_exps):
        ode_exec_strings.append(str(sp.simplify(real(e).expand())))
        if ind != 0:
            ode_exec_strings.append(str(sp.simplify(real(j * e).expand())))
    for ind in range(len(ode_exec_strings)):
        ode_exec_strings[ind] = ode_exec_strings[ind].replace('lap*', "odeB @ ").replace("g2*", "odeA @ ")
        p = re.compile(r"@ (\w{2}\d+)")
        ode_exec_strings[ind] = p.sub(r'@ all_vars["\1"]', ode_exec_strings[ind])
        p = re.compile(r'(p(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'ode_vars["\1"]', ode_exec_strings[ind])
        p = re.compile('"ode_vars\["(\w{2}\d+)"\]"')
        ode_exec_strings[ind] = p.sub(r'"\1"', ode_exec_strings[ind])
        p = re.compile(r'(w(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'w_ctrl["\1"]', ode_exec_strings[ind])
        p = re.compile(r'(d(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'd_ctrl["\1"]', ode_exec_strings[ind])

    alg_exec_strings = []
    for ind, e in enumerate(alg_exps):
        alg_exec_strings.append(str(sp.simplify(real(e).expand())))
        if ind != 0:
            alg_exec_strings.append(str(sp.simplify(real(j * e).expand())))
    for ind in range(len(alg_exec_strings)):
        alg_exec_strings[ind] = alg_exec_strings[ind].replace("g2*", "algA @ ")
        p = re.compile(r"@ (\w{2}\d+)")
        alg_exec_strings[ind] = p.sub(r'@ all_vars["\1"]', alg_exec_strings[ind])
        p = re.compile(r'(p(r|i)\d+)')
        alg_exec_strings[ind] = p.sub(r'alg_vars["\1"]', alg_exec_strings[ind])
        p = re.compile('"alg_vars\["(\w{2}\d+)"\]"')
        alg_exec_strings[ind] = p.sub(r'"\1"', alg_exec_strings[ind])
    return ode_exec_strings, alg_exec_strings


def eqn_gen_v2(p_n, w_n, d_n):
    ode_exps, alg_exps = expr_gen(p_n, w_n, d_n)
    ode_exec_strings = []
    for ind, e in enumerate(ode_exps):
        ode_exec_strings.append(str(sp.simplify(real(e).expand())))
        if ind != 0:
            ode_exec_strings.append(str(sp.simplify(real(j * e).expand())))
    for ind in range(len(ode_exec_strings)):
        ode_exec_strings[ind] = ode_exec_strings[ind].replace('lap*', "odeB @ ").replace("g2*", "odeA @ ")
        p = re.compile(r"@ (\w{2}\d+)")
        ode_exec_strings[ind] = p.sub(r'@ ode_vars["\1"]', ode_exec_strings[ind])
        p = re.compile(r'(p(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'ode_vars["\1"]', ode_exec_strings[ind])
        p = re.compile('"ode_vars\["(\w{2}\d+)"\]"')
        ode_exec_strings[ind] = p.sub(r'"\1"', ode_exec_strings[ind])
        p = re.compile(r'(w(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'w_ctrl["\1"]', ode_exec_strings[ind])
        p = re.compile(r'(d(r|i)\d+)')
        ode_exec_strings[ind] = p.sub(r'd_ctrl["\1"]', ode_exec_strings[ind])

    alg_exec_strings = []
    for ind, e in enumerate(alg_exps):
        alg_exec_strings.append(str(sp.simplify(real(e).expand())))
        if ind != 0:
            alg_exec_strings.append(str(sp.simplify(real(j * e).expand())))
    for ind in range(len(alg_exec_strings)):
        alg_exec_strings[ind] = alg_exec_strings[ind].replace("g2*", "algA @ ")
        p = re.compile(r"@ (\w{2}\d+)")
        alg_exec_strings[ind] = p.sub(r'@ ode_vars["\1"]', alg_exec_strings[ind])
        p = re.compile(r'\*(p(r|i)\d+)')
        alg_exec_strings[ind] = p.sub(r'*st_to_edge @ ode_vars["\1"]', alg_exec_strings[ind])
    return ode_exec_strings, alg_exec_strings


def ss_eqn_gen(p_n, w_n, d_n):
    ode_exps, alg_exps = expr_gen(p_n, w_n, d_n)
    ode_exec_strings = []
    for ind, e in enumerate(ode_exps):
        ode_exec_strings.append(str(sp.simplify(real(e).expand()).expand()))
        if ind != 0:
            ode_exec_strings.append(str(sp.simplify(real(j * e).expand()).expand()))
    for ind in range(len(ode_exec_strings)):
        ode_exec_strings[ind] = ode_exec_strings[ind].replace("- ", "-")
        ode_exec_strings[ind] = ode_exec_strings[ind].replace("+ ", "+")
        ode_exec_strings[ind] = ode_exec_strings[ind].split(" ")
    alg_exec_strings = []
    for ind, e in enumerate(alg_exps):
        alg_exec_strings.append(str(sp.simplify(real(e).expand()).expand()))
        if ind != 0:
            alg_exec_strings.append(str(sp.simplify(real(j * e).expand()).expand()))
    for ind in range(len(alg_exec_strings)):
        alg_exec_strings[ind] = alg_exec_strings[ind].replace("- ", "-")
        alg_exec_strings[ind] = alg_exec_strings[ind].replace("+ ", "+")
        alg_exec_strings[ind] = alg_exec_strings[ind].split(" ")
    return ode_exec_strings, alg_exec_strings


def ode_list_to_row(term_list, ctrl_key_list, plant_pars):
    pars = SimpleNamespace(**plant_pars)
    n_state_pts = pars.n_state_pts
    odeA = pars.state_dz[1:-1]
    odeB = pars.state_dz2[1:-1]
    ld2 = pars.ld2
    model_order = pars.model_order
    pi = np.pi
    ode_terms = {}
    ctrl_constants_dict = {key: {} for key in ctrl_key_list}
    for key in ctrl_key_list:
        for k_p in range(model_order):
            if k_p == 0:
                ctrl_constants_dict[key]['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts - 2))
            else:
                ctrl_constants_dict[key]['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts - 2))
                ctrl_constants_dict[key]['pi{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts - 2))
    for k_p in range(model_order):
        if k_p == 0:
            ode_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts))
        else:
            ode_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts))
            ode_terms['pi{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts - 2, n_state_pts))
    row_matrix = None
    for key, item in ode_terms.items():
        regexp1 = re.compile(r'([w|d])([r|i]\d+)\*{0}'.format(key))
        regexp2 = re.compile(r'{0}\*([w|d])([r|i]\d+)'.format(key))
        for term in term_list:
            mod_term = term.replace('+', '')
            if key in mod_term:
                if 'g2' in mod_term:
                    mod_term = mod_term.replace('g2*{0}'.format(key), 'odeA')
                elif 'lap' in mod_term:
                    mod_term = mod_term.replace('lap*{0}'.format(key), 'odeB')
                elif regexp1.search(mod_term):
                    searched = regexp1.search(mod_term)
                    ctrl_key = f'{searched.group(1)}{searched.group(2)}'
                    if ctrl_key in ctrl_key_list:
                        ctrl_constant = regexp1.sub("1", mod_term)
                        ctrl_constants_dict[ctrl_key][key] += eval(ctrl_constant) * np.eye(n_state_pts - 2)
                    mod_term = '0'
                elif regexp2.search(mod_term):
                    searched = regexp2.search(mod_term)
                    ctrl_key = f'{searched.group(1)}{searched.group(2)}'
                    if ctrl_key in ctrl_key_list:
                        ctrl_constant = regexp2.sub("1", mod_term)
                        ctrl_constants_dict[ctrl_key][key] += eval(ctrl_constant) * np.eye(n_state_pts - 2)
                    mod_term = '0'
                else:
                    mod_term = mod_term.replace('{0}'.format(key),
                                                r'np.hstack((np.zeros((n_state_pts-2,1)),'
                                                r'np.eye(n_state_pts-2),'
                                                r'np.zeros((n_state_pts-2,1))))')
                ode_terms[key] += eval(mod_term)
        if row_matrix is None:
            row_matrix = ode_terms[key]
        else:
            row_matrix = np.hstack((row_matrix, ode_terms[key]))
    ctrl_mats = {}
    for key1 in ctrl_key_list:
        ctrl_mat = None
        for key, val in ctrl_constants_dict[key1].items():
            if ctrl_mat is None:
                ctrl_mat = val
            else:
                ctrl_mat = np.hstack((ctrl_mat, val))
        ctrl_mats[key1] = ctrl_mat
    return row_matrix, ctrl_mats


def alg_list_to_row(term_list, plant_pars):
    pars = SimpleNamespace(**plant_pars)
    algA = pars.state_dz[[0, -1]]
    ld2 = pars.ld2
    n_state_pts = pars.n_state_pts
    model_order = pars.model_order
    pi = np.pi
    alg_terms = {}
    for k_p in range(model_order):
        if k_p == 0:
            alg_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts))
        else:
            alg_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts))
            alg_terms['pi{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts))
    row_matrix = None
    for key, item in alg_terms.items():
        # print(key)
        for term in term_list:
            mod_term = term.replace('+', '')
            if key in mod_term:
                if 'g2' in mod_term:
                    mod_term = mod_term.replace('g2*{0}'.format(key), 'algA')
                else:
                    evec1 = np.array([1, 0]).reshape(-1, 1)
                    evec2 = np.array([0, 1]).reshape(-1, 1)
                    mod_term = mod_term.replace('{0}'.format(key),
                                                r'np.hstack((evec1,np.zeros((2,n_state_pts-2)),evec2))')
                # print(mod_term)
                alg_terms[key] += eval(mod_term)
        if row_matrix is None:
            row_matrix = alg_terms[key]
        else:
            row_matrix = np.hstack((row_matrix, alg_terms[key]))
    return row_matrix


def ode_list_to_row_v2(term_list, ctrl_key_list, plant_pars):
    pars = SimpleNamespace(**plant_pars)
    n_state_pts = pars.n_state_pts
    odeA = pars.state_dz[1:-1]
    odeB = pars.state_dz2[1:-1]
    ld2 = pars.ld2
    model_order = pars.model_order
    pi = np.pi
    ode_terms = {}
    ctrl_constants_dict = {key: {} for key in ctrl_key_list}
    for key in ctrl_key_list:
        for k_p in range(model_order):
            if k_p == 0:
                ctrl_constants_dict[key]['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts))
            else:
                ctrl_constants_dict[key]['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts))
                ctrl_constants_dict[key]['pi{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts))
    for k_p in range(model_order):
        if k_p == 0:
            ode_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts + 2))
        else:
            ode_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts + 2))
            ode_terms['pi{0:0=3d}'.format(k_p)] = np.zeros((n_state_pts, n_state_pts + 2))
    row_matrix = None
    for key, item in ode_terms.items():
        regexp1 = re.compile(r'([w|d])([r|i]\d+)\*{0}'.format(key))
        regexp2 = re.compile(r'{0}\*([w|d])([r|i]\d+)'.format(key))
        for term in term_list:
            mod_term = term.replace('+', '')
            if key in mod_term:
                if 'g2' in mod_term:
                    mod_term = mod_term.replace('g2*{0}'.format(key), 'odeA')
                elif 'lap' in mod_term:
                    mod_term = mod_term.replace('lap*{0}'.format(key), 'odeB')
                elif regexp1.search(mod_term):
                    searched = regexp1.search(mod_term)
                    ctrl_key = f'{searched.group(1)}{searched.group(2)}'
                    if ctrl_key in ctrl_key_list:
                        ctrl_constant = regexp1.sub("1", mod_term)
                        ctrl_constants_dict[ctrl_key][key] += eval(ctrl_constant) * np.eye(n_state_pts)
                    mod_term = '0'
                elif regexp2.search(mod_term):
                    searched = regexp2.search(mod_term)
                    ctrl_key = f'{searched.group(1)}{searched.group(2)}'
                    if ctrl_key in ctrl_key_list:
                        ctrl_constant = regexp2.sub("1", mod_term)
                        ctrl_constants_dict[ctrl_key][key] += eval(ctrl_constant) * np.eye(n_state_pts)
                    mod_term = '0'
                else:
                    mod_term = mod_term.replace('{0}'.format(key),
                                                r'np.hstack((np.zeros((n_state_pts,1)),'
                                                r'np.eye(n_state_pts),'
                                                r'np.zeros((n_state_pts,1))))')
                ode_terms[key] += eval(mod_term)
        if row_matrix is None:
            row_matrix = ode_terms[key]
        else:
            row_matrix = np.hstack((row_matrix, ode_terms[key]))
    ctrl_mats = {}
    for key1 in ctrl_key_list:
        ctrl_mat = None
        for key, val in ctrl_constants_dict[key1].items():
            if ctrl_mat is None:
                ctrl_mat = val
            else:
                ctrl_mat = np.hstack((ctrl_mat, val))
        ctrl_mats[key1] = ctrl_mat
    return row_matrix, ctrl_mats


def alg_list_to_row_v2(term_list, plant_pars):
    pars = SimpleNamespace(**plant_pars)
    algA = pars.state_dz[[0, -1]]
    ld2 = pars.ld2
    n_state_pts = pars.n_state_pts
    model_order = pars.model_order
    pi = np.pi
    alg_terms = {}
    for k_p in range(model_order):
        if k_p == 0:
            alg_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts + 2))
        else:
            alg_terms['pr{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts + 2))
            alg_terms['pi{0:0=3d}'.format(k_p)] = np.zeros((2, n_state_pts + 2))
    row_matrix = None
    for key, item in alg_terms.items():
        # print(key)
        for term in term_list:
            mod_term = term.replace('+', '')
            if key in mod_term:
                if 'g2' in mod_term:
                    mod_term = mod_term.replace('g2*{0}'.format(key), 'algA')
                else:
                    evec1 = np.array([1, 0]).reshape(-1, 1)
                    evec2 = np.array([0, 1]).reshape(-1, 1)
                    mod_term = mod_term.replace('{0}'.format(key),
                                                r'np.hstack((evec1,np.zeros((2,n_state_pts)),evec2))')
                # print(mod_term)
                alg_terms[key] += eval(mod_term)
        if row_matrix is None:
            row_matrix = alg_terms[key]
        else:
            row_matrix = np.hstack((row_matrix, alg_terms[key]))
    return row_matrix


def pars_to_state_space(plant_pars):
    pars = SimpleNamespace(**plant_pars)
    n_state_pts = pars.n_state_pts
    plate_gap = pars.plate_gap
    ld2 = pars.ld2
    model_order = pars.model_order
    w_ctrl_order = pars.w_ctrl_order
    d_ctrl_order = pars.d_ctrl_order
    ctrl_key_list = pars.ctrl_key_list
    state_pts, state_dz, state_dz2, state_ig = utils.colloc(n_state_pts, True, True, plate_gap=plate_gap,
                                                            shifted=False)
    ode_list, alg_list = ss_eqn_gen(model_order, w_ctrl_order, d_ctrl_order)
    mini_plant_pars = dict(n_state_pts=n_state_pts,
                           plate_gap=plate_gap,
                           ld2=ld2,
                           model_order=model_order,
                           w_ctrl_order=w_ctrl_order,
                           d_ctrl_order=d_ctrl_order,
                           state_pts=state_pts,
                           state_dz=state_dz,
                           state_dz2=state_dz2,
                           state_ig=state_ig,
                           ctrl_key_list=ctrl_key_list)
    a_mat = None
    for mini_list in ode_list:
        if a_mat is None:
            a_mat, _ = ode_list_to_row_v2(mini_list, ctrl_key_list, mini_plant_pars)
        else:
            a_row, _ = ode_list_to_row_v2(mini_list, ctrl_key_list, mini_plant_pars)
            a_mat = np.vstack((a_mat, a_row))
    for mini_list in alg_list:
        a_mat = np.vstack((a_mat, alg_list_to_row_v2(mini_list, mini_plant_pars)))
    new_mat = a_mat
    n_eqns = (model_order - 1) * 2 + 1
    n_d = n_state_pts * n_eqns
    n_a = 2 * n_eqns
    n_tot = n_d + n_a
    alg_cols = [0]
    for i in range(n_a - 1):
        if i % 2 == 0:
            alg_cols.append(alg_cols[-1] + n_state_pts + 1)
        else:
            alg_cols.append(alg_cols[-1] + 1)
    alg_bools = np.zeros(n_tot)
    alg_bools[alg_cols] = 1
    alg_bools = np.array(alg_bools, dtype=bool)
    a11 = new_mat[:n_d, ~alg_bools]
    a12 = new_mat[:n_d, alg_bools]
    a21 = new_mat[n_d:, ~alg_bools]
    a22 = new_mat[n_d:, alg_bools]
    inv22a21 = lstsq(a22, a21)[0]
    new_mat = a11 - a12 @ inv22a21

    ctrl_mat = np.zeros((len(ode_list), len(ode_list)))
    key_names = ['pr000']
    name_to_var = {key_names[0]: 0}
    ind = 1
    pi = np.pi
    for i in range(1, model_order):
        key_names.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
        name_to_var[f'pr{i:0=3d}'] = ind
        name_to_var[f'pi{i:0=3d}'] = ind + 1
        ind += 2
    for ind1, e1 in enumerate(ode_list):
        for e2 in e1:
            if 'wr001' in e2:
                for key, val in name_to_var.items():
                    if key in e2:
                        ctrl_mat[ind1, val] += eval(e2.replace(f"{key}*wr001", "1"))
    return new_mat, ctrl_mat


def pars_to_state_space_v2(plant_pars, return_inv_mat=False):
    pars = SimpleNamespace(**plant_pars)
    n_state_pts = pars.n_state_pts
    plate_gap = pars.plate_gap
    ld2 = pars.ld2
    model_order = pars.model_order
    w_ctrl_order = pars.w_ctrl_order
    d_ctrl_order = pars.d_ctrl_order
    ctrl_key_list = pars.ctrl_key_list
    state_pts, state_dz, state_dz2, state_ig = utils.colloc(n_state_pts, True, True, plate_gap=plate_gap,
                                                            shifted=False)
    ode_list, alg_list = ss_eqn_gen(model_order, w_ctrl_order, d_ctrl_order)
    mini_plant_pars = dict(n_state_pts=n_state_pts,
                           plate_gap=plate_gap,
                           ld2=ld2,
                           model_order=model_order,
                           w_ctrl_order=w_ctrl_order,
                           d_ctrl_order=d_ctrl_order,
                           state_pts=state_pts,
                           state_dz=state_dz,
                           state_dz2=state_dz2,
                           state_ig=state_ig,
                           ctrl_key_list=ctrl_key_list)
    a_mat = None
    for mini_list in ode_list:
        if a_mat is None:
            a_mat, _ = ode_list_to_row_v2(mini_list, ctrl_key_list, mini_plant_pars)
        else:
            a_row, _ = ode_list_to_row_v2(mini_list, ctrl_key_list, mini_plant_pars)
            a_mat = np.vstack((a_mat, a_row))
    for mini_list in alg_list:
        a_mat = np.vstack((a_mat, alg_list_to_row_v2(mini_list, mini_plant_pars)))
    new_mat = a_mat
    n_eqns = (model_order - 1) * 2 + 1
    n_d = n_state_pts * n_eqns
    n_a = 2 * n_eqns
    n_tot = n_d + n_a
    alg_cols = [0]
    for i in range(n_a - 1):
        if i % 2 == 0:
            alg_cols.append(alg_cols[-1] + n_state_pts + 1)
        else:
            alg_cols.append(alg_cols[-1] + 1)
    alg_bools = np.zeros(n_tot)
    alg_bools[alg_cols] = 1
    alg_bools = np.array(alg_bools, dtype=bool)
    a11 = new_mat[:n_d, ~alg_bools]
    a12 = new_mat[:n_d, alg_bools]
    a21 = new_mat[n_d:, ~alg_bools]
    a22 = new_mat[n_d:, alg_bools]
    try:
        inv22a21 = lstsq(a22, a21)[0]
    except ValueError:
        inv22a21 = lstsq(a22, a21)[0]
    new_mat = a11 - a12 @ inv22a21

    ctrl_mats = {}
    for ctrl_key in ctrl_key_list:
        ctrl_mat = np.zeros((len(ode_list), len(ode_list)))
        key_names = ['pr000']
        name_to_var = {key_names[0]: 0}
        ind = 1
        pi = np.pi
        for i in range(1, model_order):
            key_names.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
            name_to_var[f'pr{i:0=3d}'] = ind
            name_to_var[f'pi{i:0=3d}'] = ind + 1
            ind += 2
        for ind1, e1 in enumerate(ode_list):
            for e2 in e1:
                if ctrl_key in e2:
                    for key, val in name_to_var.items():
                        if key in e2:
                            ctrl_mat[ind1, val] += eval(e2.replace(f"{key}*{ctrl_key}", "1"))
        ctrl_mats[ctrl_key] = ctrl_mat
    if return_inv_mat:
        return new_mat, ctrl_mats, inv22a21
    else:
        return new_mat, ctrl_mats
