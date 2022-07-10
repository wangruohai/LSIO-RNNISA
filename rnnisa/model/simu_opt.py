"""
This module contains the simulation optimization algorithm for the RNN based method

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
        # print('max holding: ', np.max(self.holding_cost), 'min holding: ', np.min(self.holding_cost))
        # regula_para2 = regul_factor * (self.holding_cost / np.max(self.holding_cost)) / (self.manu_lead / np.max(self.manu_lead))
        # regula_para2 = regul_factor  / (self.manu_lead / np.max(self.manu_lead))
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
        # step_k = step_size
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
        print('number of negative:', np.sum(np.where(I_S < 0, 1, 0)))
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
        print('number of negative:', np.sum(np.where(I_S < 0, 1, 0)))
        # print('number of selected location: ', np.sum(np.where(I_S >= 1, 1, 0)))
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
        # print('decaying step size')
        # while np.any(np.abs(I_S - I_S_former)>1):
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
        print('number of negative: ', np.sum(np.where(I_S < 0, 1, 0)))
        print('number of selected location: ', np.sum(np.where(I_S >= 1, 1, 0)))
        print_run_time('SGD_subgradient', t_s_SSGD)
        path = os.path.join(self.__data_path, 'history_SGD_subgradient_' + str(I_S_0.shape[1])
                            + 'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        return I_S


    def bisection_search(self, I_S_0, selected_location=None):
        t_s = time()
        init_step_size=self.__step_size
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        cost_eval_num = 0
        grad_eval_num = 0
        print('Optimization start at: ', datetime.now().strftime('%Y-%m-%d %H:%M'))
        return_flag = False
        while True:
            avg_cost, grad_mean = self.__grad_f(I_S, self.__rep_num)
            grad_eval_num += 1
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            print(',(', avg_cost, ',', cost_eval_num, ',', grad_eval_num, ')')
            delta_Is = -init_step_size * grad_mean
            temp_k = 1.0
            while True:
                temp_I_S = I_S + temp_k * delta_Is
                temp_I_S = np.maximum(temp_I_S, 0)
                avg_cost2 = self.__cost_f(temp_I_S, self.__rep_num)
                cost_eval_num += 1
                if avg_cost2 < avg_cost:
                    I_S = temp_I_S
                    break
                else:
                    temp_k = temp_k * 0.5
                if temp_k < self.__stop_thresh:
                    return_flag = True
                    break
            if return_flag:
                break

        print('number of non-zero: ', np.count_nonzero(I_S))
        print('number of selected location: ', np.sum(np.where(I_S >= 1, 1, 0)))
        print('Time used for bisection_search: %6.2f minutes' % ((time() - t_s) / 60))
        return I_S


    def ACC_SSGD(self, I_S_0, max_epoch=np.inf, selected_location=None):
        r = 3
        print('Initial Point: ', I_S_0)
        opt_history = []
        t_start = time()
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        I_S_former = I_S
        print('ACC_SSGD start at: ', datetime.now().strftime('%Y-%m-%d %H:%M'))
        cost_x = self.__cost_f(I_S, self.__rep_num)
        opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para, np.count_nonzero(I_S)))
        _print_opt_info(cost_x, I_S, 0, self.__regula_para)

        former_cost = cost_x
        k = 0
        step_t = self.__step_size
        y = I_S
        while True:
            k += 1
            cost_y, grad_mean = self.__grad_f(y, self.__rep_num)
            grad_mean = grad_mean + self.__regula_para * np.sign(I_S)
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            I_S = y - step_t * grad_mean  # * 51 / (k + 50)
            if self.__positive_flag: I_S = np.maximum(I_S, 0)
            y = I_S + (k / (k + r)) * (I_S - I_S_former)
            cost_x = self.__cost_f(I_S, self.__rep_num)
            opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para,
                                np.count_nonzero(I_S)))
            _print_opt_info(cost_x, I_S, k, self.__regula_para)
            if k == max_epoch: break
            if abs(cost_x - former_cost + np.sum(np.abs(I_S)) * self.__regula_para - np.sum(
                    np.abs(I_S_former)) * self.__regula_para) < self.__stop_thresh:  # stopping_threshold * former_cost:#
                break
            else:
                former_cost = cost_x
                I_S_former = I_S
        print('number of non-zero: ', np.count_nonzero(I_S))
        print('number of negative: ', np.sum(np.where(I_S < 0, 1, 0)))
        print('Run Time for ACC_SSGD: %6.2f minutes' % ((time() - t_start) / 60))
        path = os.path.join(self.__data_path, 'history_ACC_SSGD_' + str(I_S_0.shape[1]) +
                            'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        return I_S


    def ISTA(self, I_S_0, selected_location=None):
        print('Initial Point: ', I_S_0)
        regula_para2 = self.__regula_para
        # print('max holding: ', np.max(self.holding_cost), 'min holding: ', np.min(self.holding_cost))
        # regula_para2 = regul_factor * (self.holding_cost / np.max(self.holding_cost)) / (self.manu_lead / np.max(self.manu_lead))
        # regula_para2 = regul_factor  / (self.manu_lead / np.max(self.manu_lead))
        opt_history = []
        t_start = time()
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        I_S_former = I_S

        print('ISTA start at: ', datetime.now().strftime('%Y-%m-%d %H:%M'))
        cost_x, grad_mean = self.__grad_f(I_S, self.__rep_num)
        if selected_location is not None:
            grad_mean = np.multiply(grad_mean, selected_location)
        opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para, np.count_nonzero(I_S)))
        _print_opt_info(cost_x, I_S, 0, self.__regula_para)

        former_cost = cost_x
        k = 0
        step_t = self.__step_size
        while True:
            k += 1
            I_S = prox((I_S_former - step_t * grad_mean), step_t * regula_para2)
            if self.__positive_flag: I_S = np.maximum(I_S, 0)
            cost_x, grad_mean = self.__grad_f(I_S, self.__rep_num)
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            opt_history.append((cost_x, cost_x + np.sum(np.abs(I_S)) * self.__regula_para,
                                np.count_nonzero(I_S)))
            _print_opt_info(cost_x, I_S, k, self.__regula_para)
            # if abs(cost_x-former_cost) < stopping_threshold*former_cost:
            if abs(cost_x - former_cost + np.sum(np.abs(I_S)) * self.__regula_para - np.sum(
                    np.abs(I_S_former)) * self.__regula_para) < self.__stop_thresh:  # stopping_threshold * former_cost:#
                break
            else:
                former_cost = cost_x
                I_S_former = I_S

        print('number of non-zero: ', np.count_nonzero(I_S))
        print('number of negative: ', np.sum(np.where(I_S < 0, 1, 0)))
        print('Run Time for ISTA: %6.2f minutes' % ((time() - t_start) / 60))
        path = os.path.join(self.__data_path, 'history_ISTA_' + str(I_S_0.shape[1]) +
                            'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        return I_S, k


    def FISTA_line_search(self, I_S_0,  selected_location=None):
        print('Initial Point: ', I_S_0)
        regula_para2 = self.__regula_para
        # print('max holding: ', np.max(self.holding_cost), 'min holding: ', np.min(self.holding_cost))
        # regula_para2 = regul_factor * (self.holding_cost / np.max(self.holding_cost)) / (self.manu_lead / np.max(self.manu_lead))
        # regula_para2 = regul_factor  / (self.manu_lead / np.max(self.manu_lead))
        opt_history = []
        t_start = time()
        if selected_location is None:
            I_S = I_S_0
        else:
            I_S = np.multiply(I_S_0, selected_location)
        I_S_former = I_S
        cost_eval_num = 0
        grad_eval_num = 0
        print('Optimization start at: ', datetime.now().strftime('%Y-%m-%d %H:%M'))
        cost_x = self.__cost_f(I_S, self.__rep_num)
        cost_eval_num += 1
        opt_history.append((cost_x, cost_eval_num, grad_eval_num,
                            cost_x + np.sum(np.abs(I_S)) * self.__regula_para,
                            np.count_nonzero(I_S)))
        _print_opt_info(cost_x, I_S, 0, self.__regula_para)
        former_cost = np.inf
        return_flag = False
        k = 0
        step_t = self.__step_size
        while True:
            k = k + 1
            y = I_S + ((k - 2) / (k + 1)) * (I_S - I_S_former)
            cost_y, grad_mean = self.__grad_f(y, self.__rep_num)
            grad_eval_num += 1
            if selected_location is not None:
                grad_mean = np.multiply(grad_mean, selected_location)
            # print('grad: ',grad_mean)
            # print('I_S: ',I_S)
            # print('max grad: ', np.max(grad_mean), 'min grad: ', np.min(grad_mean))
            # grad_mean = np.maximum(grad_mean,-10 / step_t)
            # grad_mean = np.minimum(grad_mean, 20 / step_t)
            temp_I_S = prox((y - step_t * grad_mean), step_t * regula_para2)
            if self.__positive_flag: temp_I_S = np.maximum(temp_I_S, 0)
            if self.__step_bound1 is not None: temp_I_S = cal_step_bound(I_S, temp_I_S, self.__step_bound1)
            cost_x = self.__cost_f(temp_I_S, self.__rep_num)
            cost_eval_num += 1
            while cost_x > (cost_y + np.dot((temp_I_S - y), grad_mean.T)[0, 0] + 0.5 * np.sum(
                    np.square(temp_I_S - y)) / step_t):
                step_t = 0.6 * step_t  # 0.6
                # if np.max(np.abs(grad_mean))*step_t < 0.1:
                if step_t < self.__stop_thresh * self.__step_size:
                    return_flag = True
                    break
                temp_I_S = prox((y - step_t * grad_mean), step_t * regula_para2)
                if self.__positive_flag: temp_I_S = np.maximum(temp_I_S, 0)
                if self.__step_bound1 is not None: temp_I_S = cal_step_bound(I_S, temp_I_S, self.__step_bound1)
                cost_x = self.__cost_f(temp_I_S, self.__rep_num)
                cost_eval_num += 1
            if former_cost < cost_x:
                k = 0
                step_t = self.__step_size
                former_cost = np.inf
            else:
                I_S_former = I_S
                I_S = temp_I_S
                former_cost = cost_x
                opt_history.append((cost_x, cost_eval_num, grad_eval_num,
                                    cost_x + np.sum(np.abs(I_S)) * self.__regula_para,
                                    np.count_nonzero(I_S)))
                _print_opt_info(cost_x, I_S, k, self.__regula_para)
            if return_flag:
                break

        opt_history.append((cost_x, cost_eval_num, grad_eval_num,
                            cost_x + np.sum(np.abs(I_S)) * self.__regula_para,
                            np.count_nonzero(I_S)))
        _print_opt_info(cost_x, I_S, k, self.__regula_para)
        print('number of non-zero: ', np.count_nonzero(I_S))
        print('number of negative: ', np.sum(np.where(I_S < 0, 1, 0)))
        print('Run Time for FISTA_line_search: %6.2f minutes' % ((time() - t_start) / 60))
        path = os.path.join(self.__data_path, 'history_FISTA_search_' + str(I_S_0.shape[1]) +
                            'nodes_' + datetime.now().strftime('%Y-%m-%d %H-%M') + '.pkl')
        my_dump(opt_history, path)
        print('FISTA_line_search terminated at', datetime.now().strftime('%Y-%m-%d %H-%M'))

        return I_S


    def two_stage_procedure(self, I_S_0, selected_location=None):
        t_s = time()
        """
        # I_S=sim.FISTA_line_search(I_S_0=I_S0, sample_num=40, init_step_size=0.001, regul_factor=3000,
        #                           stopping_threshold=0.001)
        # _ = sim.FISTA_line_search(I_S_0=I_S0, sample_num=10, init_step_size=0.0000004, regul_factor=10000000,
        #                             stopping_threshold=0.001,            #                             positive_flag=True)
        I_S_1 = self.FISTA_line_search(I_S_0=I_S_0, selected_location=selected_location)
        """
        I_S_1, _ = self.FISTA(I_S_0=I_S_0, selected_location=selected_location)
        selected_location = np.where(np.abs(I_S_1) <= 0, 0, 1)  # np.where(I_S >= 1, 1, 0)
        I_S_2 = self.SGD(I_S_0=I_S_1, selected_location=selected_location)
        print_run_time('Two Stage Procedure', t_s)
        return I_S_1, I_S_2




def _print_opt_info(cost, I_S, epoch_num, regul_factor=None):
    # print(',(', cost_x, ',', cost_x + np.sum(np.abs(I_S)) * regul_factor, ',',
    #       np.sum(np.where(np.abs(I_S) < self.equal_tolerance, 0, 1)), ')')
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




















