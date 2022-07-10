"""
This module contains the simulation optimization algorithm for the RNN inspired simulation approach for large-scale inventory optimization problems 
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

import os
import numpy as np
from time import time
from datetime import datetime
from rnnisa.utils.tool_function import print_run_time, my_dump


class SimOpt():
    def __init__(self, data_path, rep_num, step_size, regula_para, stop_thresh, positive_flag,
                 cost_f, grad_f, step_bound=None, step_size_ratio=1.0, stop_thresh_ratio=1.0, decay_mode=1,
                 print_grad=False):

        print('Optimization parameters:', 'rep_num', rep_num, 'regula_para',
              format(regula_para, '.3e'), 'positive_flag', positive_flag, '\nstep_bound', step_bound,
              'decay_mode', decay_mode)
        self.__rep_num = rep_num
        self.__data_path = data_path
        self.__step_size = step_size
        self.__regula_para = regula_para # regularization parameter
        self.__stop_thresh = stop_thresh #stopping threshold
        self.__positive_flag = positive_flag
        self.__decay_mode = decay_mode
        self.__grad_f = grad_f
        self.__cost_f = cost_f
        if step_bound is None:
            self.__step_bound1 = None
            self.__step_bound2 = None
        else:
            self.__step_bound1 = step_bound[0]
            self.__step_bound2 = step_bound[1]
        self.__step_size_ratio = step_size_ratio
        self.__stop_thresh_ratio = stop_thresh_ratio
        self.__print_grad = print_grad


    def FISTA(self, I_S_0, selected_location=None):
        print('FISTA:', 'step_size', format(self.__step_size, '.3e'),
              'stop_thresh', format(self.__stop_thresh, '.3e'))
        print('Initial Point', I_S_0)
        r = 3
        regula_para2 = self.__regula_para
        opt_history = []
        t_s_FISTA = time()
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        I_S_former = I_S
        print('FISTA start at:', datetime.now().strftime('%Y-%m-%d %H:%M'))
        cost_x = self.__cost_f(I_S, self.__rep_num)
        opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para, np.count_nonzero(I_S)))
        _print_opt_info(cost_x, I_S, 0, self.__regula_para)
        former_cost = cost_x  # + np.sum(np.abs(I_S)) * regul_factor
        k = 0
        y = I_S
        while True:
            k += 1
            step_k = self.__step_size * 51 / (k ** self.__decay_mode + 50)  # 51/(int(decaying_step)*(k**2)+50)#51 / (k + 50)#
            cost_y, grad_mean = self.__grad_f(y, self.__rep_num)
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            if self.__print_grad:
                print('grad max:', format(np.max(grad_mean), '.3e'))
                print('grad min:', format(np.min(grad_mean), '.3e'))
            I_S = prox((y - step_k * grad_mean), step_k * regula_para2)
            if self.__positive_flag: I_S = np.maximum(I_S, 0)
            if self.__step_bound1 is not None: I_S = cal_step_bound(I_S_former, I_S, self.__step_bound1)
            y = I_S + (k / (k + r)) * (I_S - I_S_former)
            cost_x = self.__cost_f(I_S, self.__rep_num)
            opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para, np.count_nonzero(I_S)))
            _print_opt_info(cost_x, I_S, k, self.__regula_para)
            current_cost = cost_x  # + np.sum(np.abs(I_S)) * regul_factor
            if abs(current_cost - former_cost) < self.__stop_thresh * former_cost:
                break
            else:
                former_cost = current_cost
                I_S_former = I_S
        print('number of non-zero:', np.count_nonzero(I_S))
        print_run_time('FISTA', t_s_FISTA)
        path = os.path.join(self.__data_path, 'history_FISTA_' + str(I_S_0.shape[1]) +
                            'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        print('FISTA terminated at', datetime.now().strftime('%Y-%m-%d %H-%M'))
        return I_S, k


    def SGD(self, I_S_0, selected_location=None):
        stop_thresh = self.__stop_thresh * self.__stop_thresh_ratio
        step_size = self.__step_size * self.__step_size_ratio
        print('SGD:', 'step_size', format(step_size, '.3e'),
              'stop_thresh', format(stop_thresh, '.3e'))
        print('Initial Point', I_S_0)
        t_s_SGD = time()
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        epoch_num = 0
        former_cost = 0
        opt_history = []
        while True:
            epoch_num += 1
            avg_cost, grad_mean = self.__grad_f(I_S, self.__rep_num)
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            if self.__print_grad:
                print('grad max:', format(np.max(grad_mean), '.3e'))
                print('grad min:', format(np.min(grad_mean), '.3e'))
            _print_opt_info(avg_cost, I_S, epoch_num)
            opt_history.append((avg_cost, epoch_num, np.count_nonzero(I_S)))
            if abs(avg_cost - former_cost) < stop_thresh * former_cost:
                break
            else:
                former_cost = avg_cost
            I_S_former = I_S
            I_S = I_S - step_size * grad_mean * 101 / (epoch_num + 100)  # 201 / (epoch_num + 200)
            if self.__positive_flag: I_S = np.maximum(I_S, 0)
            if self.__step_bound2 is not None: I_S = cal_step_bound(I_S_former, I_S, self.__step_bound2)

        print('number of non-zero:', np.count_nonzero(I_S))
        print_run_time('SGD', t_s_SGD)
        path = os.path.join(self.__data_path, 'history_SGD_decay_' + str(I_S_0.shape[1]) +
                            'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        print('optimization terminated at', datetime.now().strftime('%Y-%m-%d %H-%M'))
        return I_S


    def SSGD(self, I_S_0, max_epoch=np.inf):
        print('Initial Point: ', I_S_0)
        t_s_SSGD = time()
        I_S = I_S_0
        epoch_num = 0
        former_cost = 0
        opt_history = []
        while True:
            avg_cost, grad_mean = self.__grad_f(I_S, self.__rep_num)
            _print_opt_info(avg_cost, I_S, epoch_num, self.__regula_para)
            opt_history.append((avg_cost, epoch_num, avg_cost + np.sum(np.abs(I_S)) * self.__regula_para,
                                np.count_nonzero(I_S)))
            if epoch_num == max_epoch: break
            current_cost = avg_cost  # + np.sum(np.abs(I_S)) * regul_factor
            if abs(current_cost - former_cost) < self.__stop_thresh * former_cost:  # stopping_threshold:  #
                break
            else:
                former_cost = current_cost
            grad2 = grad_mean + self.__regula_para * np.sign(I_S)
            I_S = I_S - self.__step_size * grad2 * 50 / (epoch_num ** self.__decay_mode + 50)  # 201
            if self.__positive_flag: I_S = np.maximum(I_S, 0)
            epoch_num += 1

        print('number of non-zero: ', np.count_nonzero(I_S))
        print_run_time('SGD_subgradient', t_s_SSGD)
        path = os.path.join(self.__data_path, 'history_SGD_subgradient_' + str(I_S_0.shape[1])
                            + 'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        return I_S


    def two_stage_procedure(self, I_S_0, selected_location=None):
        t_s = time()
        I_S_1, _ = self.FISTA(I_S_0=I_S_0, selected_location=selected_location)
        selected_location = np.where(np.abs(I_S_1) <= 0, 0, 1)  # np.where(I_S >= 1, 1, 0)
        I_S_2 = self.SGD(I_S_0=I_S_1, selected_location=selected_location)
        print_run_time('Two Stage Procedure', t_s)
        return I_S_1, I_S_2




def _print_opt_info(cost, I_S, epoch_num, regul_factor=None):
    if regul_factor is not None:
        print(',(', format(cost, '.3e'), ',', format(cost + np.sum(np.abs(I_S)) * regul_factor, '.3e'), ',',
              np.count_nonzero(I_S), ',', epoch_num, ')')
    else:
        print(',(', format(cost, '.3e'), ',', epoch_num, ',',
              np.count_nonzero(I_S), ')')



def prox(x, t):
    x = np.where(np.abs(x) > t, x, 0)
    x = np.where(x > t, x - t, x)
    x = np.where(x < -t, x + t, x)
    return x



def cal_step_bound(x_former, x, bound_info):
    upper = np.maximum(bound_info[0], bound_info[1] * x_former)#np.maximum(7.5, 0.15 * x_former)
    lower = np.minimum(bound_info[2], bound_info[3] * x_former)#np.minimum(-15, -0.23 * x_former)
    return x_former + np.maximum(lower, np.minimum(x - x_former, upper))




















