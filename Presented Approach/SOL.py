import numpy as np
import utils
import math
from functools import partial
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import solve_continuous_are
import seaborn as sns

class Library():
    def __init__(self, chosen_bases, CMAC_setting, measure_dim, m):
        self.chosen_bases = chosen_bases
        self.n = measure_dim
        self.m = m

        self.CMAC_setting = CMAC_setting

        self.CMAC_size = 1
        for i in self.CMAC_setting['num_of_receptors']: self.CMAC_size = self.CMAC_size * i
        self.CMAC_function = np.zeros((self.CMAC_size))
        self.CMAC_functionP = np.zeros((self.CMAC_size, self.n))

        self.grid_x0 = np.linspace(self.CMAC_setting['domain_of_interest'][0][0],
                                   self.CMAC_setting['domain_of_interest'][0][1],
                                   self.CMAC_setting['num_of_receptors'][0])
        self.grid_x1 = np.linspace(self.CMAC_setting['domain_of_interest'][1][0],
                                   self.CMAC_setting['domain_of_interest'][1][1],
                                   self.CMAC_setting['num_of_receptors'][1])

        self.polyn_order=0
        for i,base in enumerate(self.chosen_bases):
            if 'X^' in base:
                self.polyn_order=int(base[2:])
                self.chosen_bases[i]='X^'

        # Construct a matrix to store different combinations of different orders of polynomials.
        # Each row corresponds to one basis.
        self.array_of_pows=[]
        self.combine(self.n)
        self.array_of_pows=np.reshape(self.array_of_pows,(-1,self.n))
        self.array_of_pows = self.array_of_pows[np.sum(self.array_of_pows,axis=1).argsort()]
        self.array_of_pows=np.flip(self.array_of_pows,axis=1)
        self.len_polyn=self.array_of_pows.shape[0]





        # library of bases
        self.lib = {'1': lambda x: [1], \
                    'x': lambda x: x, \
                    'x^2': lambda x: x ** 2, \
                    'x^3': lambda x: x ** 3, \
                    'sinx': lambda x: np.sin(x), \
                    '(sinx)^2': lambda x: np.sin(x) ** 2, \
                    'cosx': lambda x: np.cos(x), \
                    '(cosx)^2': lambda x: np.cos(x) ** 2, \
                    'xx': lambda x: self.build_product_xx(x), \
                    'xxx': lambda x: self.build_product_xxx(x), \
                    'xsinx^2': lambda x: x[1] * (np.sin(x[0]) ** 2), \
                    'CMAC': lambda x: self.build_CMAC(x),\
                    'X^':lambda x: self.build_polyn(x)}
        # library of the corresponding gradients
        self.plib = {'1': lambda x: x * 0, \
                     'x': lambda x: np.diag(x ** 0), \
                     'x^2': lambda x: np.diag(2 * x), \
                     'x^3': lambda x: np.diag(3 * (x ** 2)), \
                     'sinx': lambda x: np.diag(np.cos(x)), \
                     '(sinx)^2': lambda x: np.diag(np.multiply(2 * np.sin(x), np.cos(x))), \
                     'cosx': lambda x: np.diag(-np.sin(x)), \
                     '(cosx)^2': lambda x: np.diag(np.multiply(-2 * np.cos(x), np.sin(x))), \
                     'xx': lambda x: self.build_pproduct_xx(x), \
                     'xxx': lambda x: self.build_pproduct_xxx(x), \
                     'xsinx^2': lambda x: np.array([[x[1] * np.sin(2 * x[0]), (np.sin(x[0]) ** 2)]]), \
                     'CMAC': lambda x: self.build_P_CMAC(x),\
                     'X^':lambda x: self.build_ppolyn(x)}
        # library of the corresponding labels
        self.lib_labels = {'1': '1', \
                           'x': self.build_lbl('x'), \
                           'x^2': self.build_lbl('x^2'), \
                           'x^3': self.build_lbl('x^3'), \
                           'sinx': self.build_lbl('sinx'), \
                           '(sinx)^2': self.build_lbl('(sinx)^2'), \
                           'cosx': self.build_lbl('cosx'), \
                           '(cosx)^2': self.build_lbl('(cosx)^2'), \
                           'xx': self.build_lbl_product_xx('xx'), \
                           'xxx': self.build_lbl_product_xxx('xxx'), \
                           'xsinx^2': ['x_2sinx_1^2'], \
                           'CMAC': self.build_lbl_CMAC(),\
                           'X^':self.build_lbl_polyn()}
        self.lib_dims = {'1': 1, \
                         'x': self.n, \
                         'x^2': self.n, \
                         'x^3': self.n, \
                         'sinx': self.n, \
                         '(sinx)^2': self.n, \
                         'cosx': self.n, \
                         '(cosx)^2': self.n, \
                         'xx': (self.n ** 2 - self.n) / 2, \
                         'xxx': len(self.build_lbl_product_xxx('xxx')), \
                         'xsinx^2': 1, \
                         'CMAC': self.CMAC_size,\
                         'X^':self.len_polyn}

        self._Phi_lbl = []
        for i in self.chosen_bases:
            self._Phi_lbl.extend(self.lib_labels[i])
        self._Phi_dim = len(self._Phi_lbl)
        # reserve the memeory required to evaluate Phi
        self._Phi_res = np.zeros((self._Phi_dim))
        # reserve the memeory required to evaluate pPhi
        self._pPhi_res = np.zeros((self._Phi_dim, self.n))

    def combine(self,n,list=[]):
        if n == 0:
            # print(list)
            if sum(list)<=self.polyn_order:
                self.array_of_pows=np.concatenate((self.array_of_pows,np.array(list)),axis=0)
            return
        for i in range(self.polyn_order+1):
            list.append(i)
            self.combine(n - 1,list)
            list.pop()
    def build_polyn(self,x):
        multiplied=np.ones(self.len_polyn)
        for i in range(self.n):
            multiplied=np.multiply(multiplied,np.power(x[i],self.array_of_pows[:,i]))
        return multiplied
    def build_ppolyn(self,x):
        gradient=np.ones((self.len_polyn,self.n))
        for j in range(self.n):
            multiplied=self.array_of_pows[:, j]
            for i in range(self.n):
                if j==i:
                    multiplied = np.multiply(multiplied, np.power(x[i], np.abs(self.array_of_pows[:, j] - np.ones(self.len_polyn))))
                else:
                    multiplied = np.multiply(multiplied, np.power(x[i], self.array_of_pows[:, i]))
            gradient[:,j]=multiplied
        return gradient

    def build_lbl_polyn(self):
        lbl_list=[]
        for j in range(self.len_polyn):
            lbl=''
            for i in range(self.n):
                if int(self.array_of_pows[j,i]) != 0:
                    if int(self.array_of_pows[j, i]) == 1:
                        lbl = lbl + 'x({})'.format(i + 1)
                    else:
                        lbl=lbl+'x({})^{}'.format(i+1,int(self.array_of_pows[j,i]))
            if len(lbl)==0: lbl='1'
            lbl_list.append(lbl)

        return lbl_list



    def build_product_xx(self, x):
        function = np.zeros((int((self.n ** 2 - self.n) / 2)))
        ind = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                function[ind] = x[i] * x[j]
                ind += 1
        return function

    def build_pproduct_xx(self, x):
        g = np.zeros((int((self.n ** 2 - self.n) / 2), self.n))
        ind = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                g[ind][i] = x[j]
                g[ind][j] = x[i]
                ind += 1
        return g

    def build_lbl_product_xx(self, func_name):
        lbl = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                index1 = func_name.find('x')
                index2 = func_name.find('x', index1 + 1)
                lbl.append(
                    func_name[:index1 + 1] + '({})'.format(i + 1) + func_name[index1 + 1:index2 + 1] + '({})'.format(
                        j + 1) + func_name[index2 + 1:])
        return lbl

    def build_product_xxx(self, x):
        n = len(x)
        function = np.zeros((self.lib_dims['xxx']))
        ind = 0
        for i in range(n):
            for j in range(i, n):
                for k in range(j + 1, n):
                    function[ind] = x[i] * x[j] * x[k]
                    ind += 1
        return function

    def build_pproduct_xxx(self, x):
        n = len(x)
        g = np.zeros((self.lib_dims['xxx'], n))
        for i in range(self.lib_dims['xxx']):
            for j in range(n):
                if j in self.product_table_xxx[i]:
                    if self.product_table_xxx[i][0] == self.product_table_xxx[i][1] == j:
                        g[i][j] = 2 * x[j]
                    else:
                        g[i][j] = 1

                    for ind in self.product_table_xxx[i]:
                        if ind != j:
                            g[i][j] = g[i][j] * x[ind]
        return g

    def build_lbl_product_xxx(self, func_name):
        lbl = []
        self.product_xxx_length = 0
        self.product_table_xxx = []
        for i in range(self.n):
            for j in range(i, self.n):
                for k in range(j + 1, self.n):
                    lbl.append('x({})x({})x({})'.format(i + 1, j + 1, k + 1))
                    self.product_table_xxx.append(np.array([i, j, k]))
        return lbl

    def build_CMAC(self, x):
        for i in range(self.CMAC_setting['num_of_receptors'][0]):
            for j in range(self.CMAC_setting['num_of_receptors'][1]):
                self.CMAC_function[i * j + j] = np.exp(
                    -(((x[0] - self.grid_x0[i]) ** 2) / (2 * self.CMAC_setting['variance'][0]) \
                      + ((x[1] - self.grid_x1[j]) ** 2) / (2 * self.CMAC_setting['variance'][1])))

        return self.CMAC_function

    def build_P_CMAC(self, x):

        for i in range(self.CMAC_setting['num_of_receptors'][0]):
            for j in range(self.CMAC_setting['num_of_receptors'][1]):
                self.CMAC_functionP[i * j + j, 0] = -2 * (x[0] - self.grid_x0[i]) / (
                            2 * self.CMAC_setting['variance'][0]) * np.exp(
                    -(((x[0] - self.grid_x0[i]) ** 2) / (2 * self.CMAC_setting['variance'][0]) \
                      + ((x[1] - self.grid_x1[j]) ** 2) / (2 * self.CMAC_setting['variance'][1])))

                self.CMAC_functionP[i * j + j, 1] = -2 * (x[1] - self.grid_x1[i]) / (
                            2 * self.CMAC_setting['variance'][0]) * np.exp(
                    -(((x[0] - self.grid_x0[i]) ** 2) / (2 * self.CMAC_setting['variance'][0]) \
                      + ((x[1] - self.grid_x1[j]) ** 2) / (2 * self.CMAC_setting['variance'][1])))
        return self.CMAC_functionP

    def build_lbl_CMAC(self):
        lbl = []
        for i in range(self.CMAC_setting['num_of_receptors'][0]):
            for j in range(self.CMAC_setting['num_of_receptors'][1]):
                lbl.append('CMAC({},{})'.format(self.grid_x0[i], self.grid_x1[j]))

        return lbl

    def build_lbl(self, func_name):
        lbl = []
        for i in range(self.n):
            index = func_name.find('x')
            lbl.append(func_name[:index + 1] + '({})'.format(i + 1) + func_name[index + 1:])
        return lbl

    def _Phi_(self, x):
        i = 0
        for key in self.chosen_bases:
            temp = int(self.lib_dims[key])
            self._Phi_res[i:i + temp] = self.lib[key](x)
            i += temp
        return self._Phi_res

    def _pPhi_(self, x):
        i = 0
        for key in self.chosen_bases:
            temp = int(self.lib_dims[key])
            self._pPhi_res[i:i + temp, :] = self.plib[key](x)
            i += temp
        return self._pPhi_res


class Control():
    def __init__(self, h, Objective, Lib, P_init):
        self.Objective = Objective
        self.Lib = Lib
        self.Qb = np.zeros((self.Lib._Phi_dim, self.Lib._Phi_dim))
        self.Qb[1:self.Lib.n + 1, 1:self.Lib.n + 1] = self.Objective.Q
        self.P = P_init
        self.h = h
        self.update_P = 1

    def integrate_P_dot(self, x, Wt, k, sparsify):
        self.Phi = self.Lib._Phi_(x)
        self.pPhi = self.Lib._pPhi_(x)
        dp_dt = partial(self.P_dot, x=x, Wt=Wt)
        sol = solve_ivp(dp_dt, [0, k * self.h], self.P.flatten(), method='LSODA', t_eval=None, rtol=10, atol=10,
                        dense_output=False, events=None, vectorized=False, min_step=0.01, max_step=0.02)
        if sol.status == -1:
            self.P *= 0
            print('Reset occuried...!')
        else:
            self.P = sol.y[..., -1].reshape((self.Lib._Phi_dim, self.Lib._Phi_dim))


        if (sparsify):
            absPk = np.absolute(self.P)
            maxP = np.amax(absPk)
            small_index = absPk < (0.001 * maxP)  # np.logical_and(absPk<0.1 , absPk<(0.0001*maxP))
            self.P[small_index] = 0


    def P_dot(self, t, P, x, Wt):
        P = P.reshape((self.Lib._Phi_dim, self.Lib._Phi_dim))
        W = Wt[:, :self.Lib._Phi_dim]
        SIGMA = np.zeros((self.Lib._Phi_dim, self.Lib._Phi_dim))

        P_pPhi_W = np.matmul(np.matmul(P, self.pPhi), W)
        for im in range(self.Lib.m):
            P_pPhi_Wcj_Phi = np.matmul(np.matmul(P, self.pPhi),
                                       np.matmul(Wt[:, self.Lib._Phi_dim * (im + 1):self.Lib._Phi_dim * (im + 2)],
                                                 self.Phi))
            SIGMA += 1 / self.Objective.R[im, im] * np.outer(P_pPhi_Wcj_Phi, P_pPhi_Wcj_Phi)
        out = ((self.Qb - SIGMA + P_pPhi_W + P_pPhi_W.T - self.Objective.gamma * P) * self.update_P)
        return out.flatten()

    def calculate(self, x, Wt, u_lim):
        u = np.zeros((self.Lib.m))
        for im in range(self.Lib.m):
            u[im] = -(1 / self.Objective.R[im, im]) * np.matmul(self.Phi,
                                                                np.matmul(np.matmul(self.P, self.Lib._pPhi_(x)),
                                                                          np.matmul(Wt[:, self.Lib._Phi_dim * (
                                                                                      im + 1):self.Lib._Phi_dim * (
                                                                                      im + 2)], self.Phi)))
        u = np.clip(u, -u_lim[0], u_lim[0])
        return u

    def value(self, x):
        self.Phi = self.Lib._Phi_(x)
        return np.matmul(np.matmul(self.Phi, self.P), self.Phi)

    def regulate(self, u_des, u):
        self.Objective.R[0, 0] = (abs(u) / abs(u_des)) * self.Objective.R[0, 0]
        return self.Objective.R


class Objective():
    def __init__(self, Q, R, gamma):
        self.gamma = gamma
        self.Q = Q
        self.R = R

    def stage_cost(self, x, u):
        return np.matmul(np.matmul(x, Q), x) + np.matmul(np.matmul(u, R), u)


class Database():
    def __init__(self, db_dim, Theta_dim, output_dir_path, Lib, load=True, save=True):
        self.db_dim = db_dim
        self.output_dir_path = output_dir_path
        self.Lib = Lib
        self._Phi_dim = self.Lib._Phi_dim
        self.Theta_dim = Theta_dim
        self.save = save
        self.load = load
        self.db_index = 0

        if load & os.path.exists(self.output_dir_path + '/db_Theta.npy'):
            self.db_Theta = np.load(self.output_dir_path + '/db_Theta.npy')
            self.db_X_dot = np.load(self.output_dir_path + '/db_X_dot.npy')
            self.db_overflow = np.load(self.output_dir_path + '/db_overflow.npy')
            print(self.db_X_dot[1, :10].T)
            print('Theta_dict:', self.db_Theta)
            self.db_index = np.load(output_dir_path + '/db_index.npy')
        else:
            self.Lib = Lib
            self.db_Theta = np.zeros((Theta_dim, self.db_dim))
            self.db_X_dot = np.zeros((self.Lib.n, self.db_dim))
            self.db_overflow = False
            self.db_index = 0

    def add(self, x, x_dot, u):
        self.db_X_dot[:, self.db_index] = x_dot
        _Phi_ = self.Lib._Phi_(x)
        self.db_Theta[:self._Phi_dim, self.db_index] = _Phi_
        for im in range(self.Lib.m):
            self.db_Theta[self._Phi_dim * (im + 1):self._Phi_dim * (im + 2), self.db_index] = _Phi_ * u[im]
        self.db_index += 1
        if self.db_index > (self.db_dim - 1):
            self.db_overflow = True
            self.db_index = 0

    def read(self):
        if self.db_overflow:
            db = [self.db_Theta, self.db_X_dot]
        else:
            db = [self.db_Theta[:, :self.db_index], self.db_X_dot[:, :self.db_index]]
        return db

    def DB_save(self):
        if self.save:
            np.save(self.output_dir_path + '/db_Theta.npy', self.db_Theta)
            np.save(self.output_dir_path + '/db_X_dot.npy', self.db_X_dot)
            np.save(self.output_dir_path + '/db_index.npy', self.db_index)
            np.save(self.output_dir_path + '/db_overflow.npy', self.db_overflow)


class Knowledge_base():
    def __init__(self, u_lim, measurments={'height': 0, 'roll': 0, 'pitch': 0, 'yaw': 0},
                 u={'pwm1': 0, 'pwm2': 0, 'pwm3': 0, 'pwm4': 0}):
        self.measurments = measurments
        self.u = u
        self.mode = set()
        self.u_pre = u  # previous values of the input
        self.mode_pre = self.mode  # previous states
        self.h_trig = 0.1
        self.ang_trig = 3
        self.control_step = 200
        self.u_lim = u_lim

    def mode_update(self, sample, u):
        self.mode = set()
        self.measurments['height'] = sample[2]
        self.measurments['roll'] = sample[9]
        self.measurments['pitch'] = sample[10]
        self.measurments['yaw'] = sample[11]

        self.u['pwm1'] = u[0]
        self.u['pwm2'] = u[1]
        self.u['pwm3'] = u[2]
        self.u['pwm4'] = u[3]
        self.u_temp = self.u  # just a copy that we need later to shape the control

        if self.measurments['height'] < -self.h_trig:
            self.mode.add('low_h')
        elif abs(self.measurments['height']) <= self.h_trig:
            self.mode.add('fine_h')
        elif self.measurments['height'] > self.h_trig:
            self.mode.add('high_h')

        if self.measurments['roll'] > self.ang_trig:
            self.mode.add('roll_p')
        elif self.measurments['roll'] < -self.ang_trig:
            self.mode.add('roll_n')

        if self.measurments['pitch'] > self.ang_trig:
            self.mode.add('pitch_p')
        elif self.measurments['pitch'] < -self.ang_trig:
            self.mode.add('pitch_n')

        if self.measurments['yaw'] > self.ang_trig:
            self.mode.add('yaw_p')
        elif self.measurments['yaw'] < -self.ang_trig:
            self.mode.add('yaw_n')
        print(self.mode)

    def enforce(self):
        if ('low_h' in self.mode):
            for key in self.u:
                if (self.u[key] - self.u_pre[key]) <= 0: self.u_temp[key] = self.u[key] + self.control_step
        elif ('high_h' in self.mode):
            for key in self.u:
                if (self.u[key] - self.u_pre[key]) >= 0: self.u_temp[key] = self.u[key] - self.control_step

        if ('roll_p' in self.mode):
            if (self.u['pwm1'] - self.u_pre['pwm1']) <= 0: self.u_temp['pwm1'] = self.u['pwm1'] + self.control_step / 2
            if (self.u['pwm2'] - self.u_pre['pwm2']) <= 0: self.u_temp['pwm2'] = self.u['pwm2'] + self.control_step / 2
            if (self.u['pwm3'] - self.u_pre['pwm3']) >= 0: self.u_temp['pwm3'] = self.u['pwm3'] - self.control_step / 2
            if (self.u['pwm4'] - self.u_pre['pwm4']) >= 0: self.u_temp['pwm4'] = self.u['pwm4'] - self.control_step / 2

        if ('pitch_p' in self.mode):
            if (self.u['pwm1'] - self.u_pre['pwm1']) <= 0: self.u_temp['pwm1'] = self.u['pwm1'] + self.control_step / 2
            if (self.u['pwm2'] - self.u_pre['pwm2']) >= 0: self.u_temp['pwm2'] = self.u['pwm2'] - self.control_step / 2
            if (self.u['pwm3'] - self.u_pre['pwm3']) >= 0: self.u_temp['pwm3'] = self.u['pwm3'] - self.control_step / 2
            if (self.u['pwm4'] - self.u_pre['pwm4']) <= 0: self.u_temp['pwm4'] = self.u['pwm4'] + self.control_step / 2

        if ('yaw_p' in self.mode):
            if (self.u['pwm1'] - self.u_pre['pwm1']) <= 0: self.u_temp['pwm1'] = self.u['pwm1'] + self.control_step / 2
            if (self.u['pwm2'] - self.u_pre['pwm2']) >= 0: self.u_temp['pwm2'] = self.u['pwm2'] - self.control_step / 2
            if (self.u['pwm3'] - self.u_pre['pwm3']) <= 0: self.u_temp['pwm3'] = self.u['pwm3'] + self.control_step / 2
            if (self.u['pwm4'] - self.u_pre['pwm4']) >= 0: self.u_temp['pwm4'] = self.u['pwm4'] - self.control_step / 2
        self.u_pre = self.u
        for key in self.u:
            self.u[key] = np.clip(self.u_temp[key], -self.u_lim, self.u_lim)
        return ([self.u['pwm1'], self.u['pwm2'], self.u['pwm3'], self.u['pwm4']])


class SysID():
    def __init__(self, select_ID_algorithm, Database, Weights, Lib):
        self.ID_alg = select_ID_algorithm
        self.DB = Database
        self.Weights = Weights

        self.Lib = Lib
        self.Theta = np.zeros((self.DB.Theta_dim))
        self.P_rls = np.zeros((self.Lib.n, self.DB.Theta_dim, self.DB.Theta_dim))
        for i in range(self.Lib.n):
            self.P_rls[i] = np.eye(self.DB.Theta_dim) * 1000

        if self.ID_alg['RLS'] & (self.DB.load) & (os.path.exists(self.DB.output_dir_path + '/Weights.npy')):
            self.Weights = np.load(self.DB.output_dir_path + '/Weights.npy')
            self.P_rls = np.load(self.DB.output_dir_path + '/P_rls.npy')

    def update(self, x, x_dot, u):
        if self.ID_alg['SINDy']:
            lam = 0.2
            if self.DB.db_overflow:
                self.Weights = (utils.SINDy(self.DB.db_X_dot, self.DB.db_Theta, lam))
            else:
                self.Weights = (
                    utils.SINDy(self.DB.db_X_dot[:, :self.DB.db_index], self.DB.db_Theta[:, :self.DB.db_index], lam))
        elif self.ID_alg['RLS']:
            _Phi_ = self.Lib._Phi_(x)
            self.Theta[:self.Lib._Phi_dim] = _Phi_
            for im in range(self.Lib.m):
                self.Theta[self.Lib._Phi_dim * (im + 1):self.Lib._Phi_dim * (im + 2)] = _Phi_ * u[im]
            # for i in range(self.Lib.n):
            self.Weights, self.P_rls[0] = utils.RLS(self.Theta, x_dot, self.Weights, self.P_rls[0])
        elif self.ID_alg['Least Squares']:
            lam = 0.5
            if self.DB.db_overflow:
                self.Weights = (utils.LLS(self.DB.db_X_dot, self.DB.db_Theta))
            else:
                self.Weights = (
                    utils.LLS(self.DB.db_X_dot[:, :self.DB.db_index], self.DB.db_Theta[:, :self.DB.db_index]))
        elif self.ID_alg['Gradient Descent']:
            _Phi_ = self.Lib._Phi_(x)
            self.Theta[:self.Lib._Phi_dim] = _Phi_
            for im in range(self.Lib.m):
                self.Theta[self.Lib._Phi_dim * (im + 1):self.Lib._Phi_dim * (im + 2)] = _Phi_ * u[im]
            # for i in range(self.Lib.n):
            self.Weights = utils.Gradient_descent(self.Theta, x_dot, self.Weights, 0.5e-4)  # 0.2 for Gaussaian 7e-4

        return self.Weights

    def evaluate(self, x, u):
        _Phi_ = self.Lib._Phi_(x)
        self.Theta[:self.Lib._Phi_dim] = _Phi_
        for im in range(self.Lib.m):
            self.Theta[self.Lib._Phi_dim * (im + 1):self.Lib._Phi_dim * (im + 2)] = _Phi_ * u[im]
        return np.matmul(self.Weights, self.Theta)

    def save(self):
        if self.ID_alg['RLS']:
            if (self.DB.save):
                np.save(self.DB.output_dir_path + '/Weights.npy', self.Weights)
                np.save(self.DB.output_dir_path + '/P_rls.npy', self.P_rls)

    def convert_cpp(self, w):
        ws = '{\n'
        for i in range(np.shape(w)[0]):
            ws = ws + '{'
            for j in range(np.shape(w)[1]):
                ws = ws + '{:7.3f}'.format(w[i, j])
                if j < ((np.shape(w)[1]) - 1):
                    ws = ws + ', '
            if i < ((np.shape(w)[0]) - 1):
                ws = ws + '},\n'
            else:
                ws = ws + '}\n'
        ws = ws + '}'
        return ws


class SimResults():
    def __init__(self, t, Lib, DB, SysID, Ctrl, output_dir_path, select={'states': 1, 'value': 1, 'P': 1, 'error': 1}):
        self.t = t
        len_t = len(t)
        self.Lib = Lib
        self.DB = DB
        self.SysID = SysID
        self.Ctrl = Ctrl
        self.x_s_history = np.zeros((self.Lib.n, len_t))
        self.u_history = np.zeros((self.Lib.m, len_t))
        self.P_history = np.zeros((len_t, self.Lib._Phi_dim, self.Lib._Phi_dim))
        self.P_norm_history = np.zeros((len_t, 1, 1))
        self.V_history = np.zeros((len_t))
        self.error_history = np.zeros((len_t))
        self.runtime_history = np.zeros((2,len_t))

        self.select = select
        self.output_dir_path = output_dir_path
        self.pallet = ['r', 'g', 'b', 'm', '#E67E22', '#1F618D']

    def record(self, i, x, u, P, V, error, x_ref, runtime):
        # x[0]=utils.rad_regu(x[0])
        temp = np.copy(x)
        temp = x_ref - temp
        self.x_s_history[:, i] = temp
        self.u_history[:, i] = u
        self.P_norm_history[i] = np.linalg.norm(P)
        self.P_history[i] = P
        self.V_history[i] = V
        self.error_history[i] = error
        self.runtime_history[:,i] = runtime * 1000

    def graph(self, j, i):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PLOT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # plot: the components of matrix 'Pk' vs. 'time' #######################################
        sns.set_theme()
        px = self.Lib._Phi_dim
        if self.select['P']:
            fig = plt.figure()
            for ii in range(px):
                for jj in range(px):
                    plt.plot(self.t[:i], self.P_history[:i, ii, jj], 'g')

            plt.savefig(self.output_dir_path + '/fig_P{}.png'.format(j))
            plt.close(fig)
            plt.show()

        # plot the 'control' + 'states' of the system vs. 'time' ################################
        if self.select['states']:
            fig1 = plt.figure()
            for im in range(self.Lib.m):
                plt.plot(self.t[:i], self.u_history[im, :i], 'c--')
            for ii in range(self.Lib.n):
                plt.plot(self.t[:i], self.x_s_history[ii, :i], self.pallet[ii % len(self.pallet)])

            plt.legend(["Control", "Angle", "Angular Velocity", "Position", "Velocity"], loc=1)
            plt.xlabel('t (sec)')
            plt.ylabel('States and Control')
            # plt.tight_layout()
            plt.ylim((-10, 10))

            plt.grid(color='k', linestyle=':', linewidth=1)
            plt.savefig(self.output_dir_path + '/fig_states_control{}.pdf'.format(j), format='pdf')
            plt.close(fig1)
            plt.show()

            # plot the 'value' + 'parameters' and error of the system vs. 'time' ################################
            fig0, axs = plt.subplots(3, 1)
            b1, = axs[0].plot(self.t[:i], self.V_history[:i], 'b')
            # axs[0].set_xlabel('time')
            axs[0].set_ylabel('Value')
            axs[0].grid(color='k', linestyle=':', linewidth=1)

            for ii in range(px):
                for jj in range(px):
                    axs[1].plot(self.t[:i], self.P_history[:i, ii, jj], 'g')
            # axs[0].set_xlabel('time')
            axs[1].set_ylabel('Parameters')
            axs[1].grid(color='k', linestyle=':', linewidth=1)

            b1, = axs[2].plot(self.t[:i], self.error_history[:i], 'r')
            # axs[0].set_xlabel('time')
            axs[2].set_ylabel('Error')
            axs[2].set_ylim([0, 100])
            axs[2].grid(color='k', linestyle=':', linewidth=1)

            plt.tight_layout()
            plt.savefig(self.output_dir_path + '/fig_states_control_Value_Param_Error{}.pdf'.format(j), format='pdf')
            plt.close(fig0)
            plt.show()

            # plot the 'value' + 'parameters' and error of the system vs. 'time' ################################
            fig9, axs = plt.subplots(2, 1)
            for ii in range(px):
                for jj in range(px):
                    axs[0].plot(self.t[:i], self.P_history[:i, ii, jj])
            # axs[0].set_xlabel('time')
            axs[0].set_ylabel('Parameters')
            axs[0].yaxis.grid()

            axs[1].plot(self.t[:i], self.error_history[:i])
            # axs[0].set_xlabel('time')
            axs[1].set_ylabel('Model Error')
            axs[1].yaxis.grid()

            plt.tight_layout()
            plt.savefig(self.output_dir_path + '/fig_states_control_Value_Param_Error{}.pdf'.format(j), format='pdf')
            plt.close(fig9)
            plt.show()

        # plot: the pridiction error ###########################################################
        if self.select['error']:
            fig2 = plt.figure()
            plt.plot(self.t[:i], self.error_history[:i], 'g')
            plt.ylim((0, 200))
            plt.savefig(self.output_dir_path + '/fig_error{}.png'.format(j))
            plt.close(fig2)
            plt.show()

        # plot: Value ###########################################################
        if self.select['value']:
            fig3 = plt.figure()
            plt.plot(self.t[:i], self.V_history[:i], 'b')
            plt.tight_layout()

            plt.savefig(self.output_dir_path + '/fig_value{}.png'.format(j))
            plt.close(fig3)
            plt.show()
        if self.select['runtime']:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            ax0=sns.histplot(self.runtime_history[0,:i],bins=100,kde=True,ax=axs[0],color="y")
            ax0.set_title('Runtime: Sys. Identification')
            ax0.set_xlim([0,6])
            ax=sns.histplot(self.runtime_history[1,:i],bins=100,kde=True,ax=axs[1],color="y")
            ax.set_title('Runtime: Control')
            ax.set_xlabel('t(msec)')
            ax.set_xlim([0,6])
            plt.tight_layout()
            plt.savefig(self.output_dir_path+'/fig_runtime{}.pdf'.format(j),format='pdf')
            plt.close(fig)
            plt.show()

    def printout(self, j):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PRINT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # print: identified system
        print('Episode {}:'.format(j + 1))
        if self.DB.db_overflow:
            print('Number of samples in database : ', self.DB.db_dim)
        else:
            print('Number of samples in database : ', self.DB.db_index)

        # print('Initial and final step values:')
        # print(initial_stage_value)
        # print(final_stage_value)
        chosen_basis_label = self.Lib._Phi_lbl
        for ii in range(self.Lib.n):
            handle_str = 'x_dot({}) = '.format(ii + 1)
            for jj in range(self.DB.Theta_dim):
                if (self.SysID.Weights[ii, jj] != 0) & (abs(self.SysID.Weights[ii, jj]) > 0.01):
                    if jj < self.Lib._Phi_dim:
                        handle_str = handle_str + (
                            ' {:7.3f}*{} '.format(self.SysID.Weights[ii, jj], chosen_basis_label[jj]))
                    elif jj >= self.Lib._Phi_dim:
                        handle_str = handle_str + (' {:7.3f}*{}*u{} '.format(self.SysID.Weights[ii, jj],
                                                                             chosen_basis_label[jj % self.Lib._Phi_dim],
                                                                             jj // self.Lib._Phi_dim))
            print(handle_str)
        # print: obtained value function
        handle_str = 'V(x) = '
        for ii in range(self.Lib._Phi_dim):
            for jj in range(ii + 1):
                if (self.Ctrl.P[ii, jj] != 0):
                    if (ii == jj):
                        handle_str = handle_str + '{:7.3f}*{}^2'.format(self.Ctrl.P[ii, jj], chosen_basis_label[jj])
                    else:
                        handle_str = handle_str + '{:7.3f}*{}*{}'.format(2 * self.Ctrl.P[ii, jj],
                                                                         chosen_basis_label[ii], chosen_basis_label[jj])
        print(handle_str)
        print("% of non-zero elements in P: {:4.1f} %".format(
            100 * np.count_nonzero(self.Ctrl.P) / (self.Lib._Phi_dim ** 2)))
