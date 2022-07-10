"""
This module contains the simulation class for application when the lead time is random
Author:
    Tan Wang
"""

import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # this
import numpy as np
from random import seed, normalvariate
from multiprocessing import Pool, cpu_count
from scipy.sparse import diags,dia_matrix

from warnings import filterwarnings
filterwarnings('ignore')

CORE_NUM = cpu_count()


class Simulation():
    def __init__(self, data_type, duration, data_path, network_name, delivery_cycle=0, penalty_factor=10.0):
        self.__duration = duration
        self.__data_type = data_type
        print('Data Type:', data_type)
        print('penalty_factor:', penalty_factor)
        self.prepare_data(data_path, network_name, penalty_factor, data_type, delivery_cycle)
        self.__seed_num = 0

    def prepare_data(self, data_path, network_name, penalty_factor, data_type, delivery_cycle):
        import networkx as nx
        from scipy.sparse import eye
        from rnnisa.utils.tool_function import my_load

        def count_layer(B):
            B.eliminate_zeros()
            B = B.tocsr()
            B = B.astype(self.__data_type)
            temp = eye(B.shape[0], dtype=self.__data_type)
            temp = temp.tocsr()
            # result = 0
            maxlayer = 0
            for i in range(B.shape[0]):
                # print(i)
                temp = B * temp
                temp.eliminate_zeros()
                if temp.nnz == 0:
                    maxlayer = i
                    break
                # result = result + temp.nnz
            # print('nnz number: ', result)
            # print("max layer: ", maxlayer + 1)
            return maxlayer + 1

        path = os.path.join(data_path, network_name)
        G = my_load(path)

        # self.G = G
        if type(G) == list:
            G = G[0]
        self.__B = nx.adjacency_matrix(G, weight='weight')
        # self.B_one=nx.adjacency_matrix(G, weight=1)
        self.__stage_num = count_layer(self.__B)
        print("stage num:", self.__stage_num)
        self.__nodes_num = self.__B.shape[0]
        print('nodes number:', self.__nodes_num)

        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type)
        self.__hold_coef = np.expand_dims(self.__hold_coef, axis=0)
        # self.holding_cost = np.array([[1,6,20,8,4,50,3,13,4,12]])
        self.__lead_time = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))
        # self.manu_lead = np.array([2, 4, 6, 4, 3, 2, 3, 3, 2, 2])
        # self.purchase_lead = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))

        self.__D_mean = np.zeros((self.__duration, self.__nodes_num))  # , dtype=int
        self.__std = np.zeros_like(self.__D_mean)
        in_degrees = G.in_degree()
        in_degree_values = np.array([v for k, v in in_degrees])
        demand_node = np.where(in_degree_values == 0)[0]
        i = 0
        t_range = range(self.__duration)
        for nd in list(G.nodes()):
            if i in demand_node:
                self.__D_mean[t_range, i] = G.nodes[nd]['mean']
                self.__std[t_range, i] = G.nodes[nd]['std']
            i += 1
        """
        seed(1)  # just for generate numbers
        for t in range(self.duration):
            self.D_mean[t, demand_node] = [randint(0, 100) for _ in demand_node]
        seed()
        self.std = self.D_mean / 2
        self.std = self.std.astype(int)
        """
        # np.random.seed(1)
        # for i in demand_node:
        #     self.D_mean[range(self.duration):,i]=np.random.randint(low=0, high=100, size=self.duration)
        # self.std = self.D_mean/2

        # temp_D_mean = np.sum(self.D_mean, axis=0)
        # demand_set = np.where(temp_D_mean > 0)
        # self.demand_set = list(demand_set[0])
        self.__demand_set = demand_node
        print('number of demand node:', len(self.__demand_set))

        self.penalty_coef = data_type(penalty_factor) * self.__hold_coef
        self.__B = self.__B.astype(data_type)
        self.__B_T = self.__B.T
        self.__B_T = self.__B_T.tocsr()
        E = eye(self.__nodes_num, dtype=data_type)
        self.__E = E.tocsr()
        E_B_T = (self.__E - self.__B).T
        self.__E_B_T = E_B_T.tocsr()
        self.__E_B_T.eliminate_zeros()

        self.__zero = data_type(0.0)
        self.__one_minus = data_type(-1.0)
        self.__one = data_type(1.0)
        if data_type == np.float32:
            self.__equal_tole = data_type(1e-5)
        else:
            self.__equal_tole = data_type(1e-11)

        # item_list = list(G.nodes())
        # self.raw_material_node = np.zeros((1, self.nodes_num), dtype=data_type)
        out_degrees = G.out_degree()
        out_degree_values = np.expand_dims(np.array([v for k, v in out_degrees]), axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)
        # i = 0
        # for nd in item_list:
        # if G.out_degree(nd) == 0:
        # self.raw_material_node[0, i] = self.one
        # i = i + 1
        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])
        self.__raw_item_diag = diags(self.__raw_material_node[0])

        idx_mau = np.nonzero(1 - self.__raw_material_node)[1]
        self.__B_indices_list = {i: self.__B[i].indices for i in
                                 idx_mau}  # [self.B[i].indices for i in range(self.nodes_num)]

        time_stamp = np.zeros((self.__duration, self.__nodes_num), dtype=int)#time_stamp_m
        # time_stamp_p = np.zeros_like(time_stamp_m)
        range_t = np.array(list(range(self.__duration)))
        time_stamp[:, :] = self.__lead_time
        # time_stamp_p[:, :] = self.lead_time
        time_stamp[:, :] = time_stamp[:, :] + np.expand_dims(range_t, axis=1)
        # time_stamp_p[:, :] = time_stamp_p[:, :] + np.expand_dims(range_t, axis=1)
        self.__time_stamp = time_stamp
        # self.time_stamp_p = time_stamp_p
        self.__time_stamp_truncated = np.minimum(time_stamp, self.__duration)

        # if delivery_cycle is None:
        #     delivery_cycle = order_time * np.ones(self.__nodes_num, dtype=int)
        if type(delivery_cycle) == int:
            delivery_cycles = delivery_cycle * np.ones(self.__nodes_num, dtype=int)
        else:
            delivery_cycles = my_load(os.path.join(data_path, delivery_cycle))
            print('max delivery cycle:', np.max(delivery_cycles))

        self.__delivery_shift = np.zeros_like(time_stamp)
        for t in range(self.__duration):
            self.__delivery_shift[t, self.__demand_set] = t - delivery_cycles[self.__demand_set]
        self.__delivery_shift = np.maximum(-1, self.__delivery_shift)

    def reset_seed(self):
        self.__seed_num = 0

    def cut_seed(self, num):
        self.__seed_num -= num

    def get_demand_set(self):
        return self.__demand_set

    def evaluate_cost(self, I_S, eval_num=30):#, print_flag=False
        process_num = min(CORE_NUM, eval_num)
        if self.__nodes_num == 500000: process_num = min(process_num, 30)
        if self.__nodes_num == 100000: process_num = min(process_num, 50)
        # I_S_list = [I_S for _ in range(eval_num)]
        I_S_list = [(I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__stage_num
                     , self.__lead_time, self.__data_type, self.__B_indices_list, self.__hold_coef, self.penalty_coef,
                     self.__raw_material_node, self.__B, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(simulate_only_parallel, I_S_list)
            # parall_result = pool.map(self.simulate_only, I_S_list)
        # result = list(zip(*result))
        # cost = np.mean(result[0])  # np.array([result[0] for result in parall_result])
        cost = np.mean(result)
        # cost2 = np.mean(result[1])  # np.array([result[1] for result in parall_result])
        # filled_demand = np.sum(result[2], axis=0)
        # D_sum = np.sum(result[3], axis=0)
        # cost1 = 0
        # cost2 = 0
        # filled_demand = 0
        # D_sum = 0
        # for result in parall_result:
        #     cost1 = cost1 + result[0]
        #     cost2 = cost2 + result[1]
        #     filled_demand = filled_demand + result[2]
        #     D_sum = D_sum + result[3]
        # cost1 = cost1 / eval_num
        # cost2 = cost2 / eval_num
        # fill_rate = filled_demand / D_sum
        # if print_flag:
        #     print('total_cost: ', cost)
        #     # print('total_holding_cost: ', cost2)
        #     # print('fill_rate: ', fill_rate[0, self.demand_set])
        return cost

    def evaluate_cost_gradient(self, I_S, eval_num=30, mean_flag=True):
        process_num = min(CORE_NUM, eval_num)
        # I_S_list = [I_S for _ in range(eval_num)]
        I_S_list = [(I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num
                     , self.__lead_time, self.__data_type, self.__B_indices_list, self.__equal_tole
                     , self.__hold_coef, self.penalty_coef, self.__mau_item_diag, self.__raw_material_node, self.__B
                     , self.__B_T, self.__E_B_T, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(simulate_and_bp_parallel, I_S_list)
            # parall_result = pool.map(self.simulate_and_bp, I_S_list)
            # parall_result = pool.map_async(self.simulate_and_bp, I_S_list).get()
        result = list(zip(*result))  # result = np.array(parall_result).T
        cost_result = np.array(result[0])  # np.array([result[0] for result in parall_result])
        grad_result = np.squeeze(result[1])  # np.squeeze(np.array([result[1] for result in parall_result]))
        if mean_flag:
            cost_result = np.mean(cost_result)  # np.sum(cost_result) / eval_num
            grad_result = np.mean(grad_result, axis=0,
                                  keepdims=True)  # np.expand_dims(np.sum(grad_result, axis=0),axis=0) / eval_num
        return cost_result, grad_result



def generate_random_demand_for_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                        zero, rand_seed):
    if rand_seed is not None:
        seed(rand_seed)
    D = np.zeros((duration, nodes_num), dtype=data_type)
    D_order = np.zeros((duration + 1, nodes_num), dtype=data_type)
    for t in range(duration):
        D_order[t, demand_set] = [normalvariate(D_mean[t, i], std[t, i]) for i in demand_set]
        D[t, demand_set] = D_order[delivery_shift[t, demand_set], demand_set]
        # for i in demand_set:
        #     D_order[t, i] = normalvariate(D_mean[t, i], std[t, i])
        #     if t >= delivery_cycle[i]:
        #         D[t, i] = D_order[int(t - delivery_cycle[i]), i]
    # D[self.order_time:self.duration, :] = D_order[0:(self.duration - self.order_time), :]
    D_order = np.maximum(zero, D_order)
    D = np.maximum(zero, D)
    return D, D_order


def simulate_and_bp_parallel(args):
    (I_S, duration, nodes_num, zero, one, one_minus, stage_num, lead_time, data_type
     , B_indices_list, equal_tolerance, holding_cost, penalty_cost, mau_item_diag, raw_material_node
     , B, B_T, E_B_T, D_mean, std, demand_set, delivery_shift, rand_seed) = args

    D, D_order = generate_random_demand_for_parallel(duration, nodes_num, data_type, D_mean, std, demand_set,
                                                     delivery_shift, zero, rand_seed)

    # t_s = time()
    # initialize
    M_buffer = np.zeros((1, nodes_num), dtype=data_type)
    W_t = np.zeros((duration, nodes_num), dtype=data_type)
    P = np.zeros((duration, nodes_num), dtype=data_type)
    D_queue = np.zeros((1, nodes_num), dtype=data_type)
    W_qty = np.zeros((1, nodes_num), dtype=data_type)
    I_t = I_S + 0  # self.initial_inv+0#
    I_position = I_S + 0  # self.initial_inv+0#
    cost = data_type(0.0)

    d_It_d_Yt = []
    d_Dq_d_Yt = []
    d_O_d_Ipformer = []
    d_W_d_Mqty_item = []
    d_M_d_man_o = []
    d_M_d_r_r = []
    d_r_r_d_I = []
    d_r_r_d_r_n = []
    d_P_d_O_item = []
    d_P_d_O_lead = []
    Mbuf_flag = []
    for i in range(duration):
        d_M_d_man_o.append(np.zeros((1, nodes_num), dtype=data_type))
        d_M_d_r_r.append({})
        d_P_d_O_item.append([])
        d_P_d_O_lead.append({})
        d_W_d_Mqty_item.append([])
        d_O_d_Ipformer.append([])

    for t in range(1, duration + 1):
        I_position = I_position - D_order[t - 1, :]
        temp_O_t = -np.minimum(zero, (I_position - I_S))
        flag = np.where((I_position - I_S) < 0, one_minus, zero)
        d_O_d_Ipformer[t - 1].insert(0, diags(flag[0]))
        for i in range(stage_num - 1):
            temp_I_position = I_position - temp_O_t * B
            temp_O_t = -np.minimum(zero, (temp_I_position - I_S))
            flag = np.where((temp_I_position - I_S) < 0, one_minus, zero)
            d_O_d_Ipformer[t - 1].insert(0, diags(flag[0]))
        O_t = temp_O_t + 0
        I_position = I_position - O_t * B + O_t
        # print('test:',I_S)
        temp_I_t = I_t - D_queue - D[t - 1] + W_t[t - 1] + P[t - 1]
        I_t = np.maximum(zero, temp_I_t)
        flag = np.where(temp_I_t >= 0, one, zero)
        d_It_d_Yt.append(diags(flag[0]))
        D_queue = -np.minimum(zero, temp_I_t)
        flag = np.where(temp_I_t < 0, one_minus, zero)
        d_Dq_d_Yt.append(diags(flag[0]))
        # W_qty = W_qty - W_t[t - 1]
        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_buffer
        idx_purch = np.nonzero(purchase_order)
        idx_purch = idx_purch[1]
        idx_mau = np.nonzero(mau_order)
        idx_mau = idx_mau[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed
        temp_resource_rate[np.isnan(temp_resource_rate)] = one  # *2
        temp1 = one / (resource_needed)
        temp2 = -np.multiply(temp_resource_rate, temp1)
        resource_rate = np.minimum(one, temp_resource_rate)
        flag2 = np.where(temp_resource_rate < 1, one, zero)
        d_r_r_d_I.append(
            diags(flag2[0]) * dia_matrix((temp1[0], [0]), shape=(nodes_num, nodes_num)))
        d_r_r_d_r_n.append(
            diags(flag2[0]) * dia_matrix((temp2[0], [0]), shape=(nodes_num, nodes_num)))
        for index in idx_purch:
            temp_lead = max(1, int(round(
                normalvariate(lead_time[index], 0.05 * lead_time[index]))))
            time_stamp = t - 1 + temp_lead
            # time_stamp = t - 1 + self.purchase_lead[index]
            if time_stamp < duration:
                P[time_stamp, index] = purchase_order[0, index]
                d_P_d_O_item[time_stamp].append(index)
                d_P_d_O_lead[time_stamp][index] = temp_lead

        M_actual = np.zeros((1, nodes_num), dtype=data_type)
        for index in idx_mau:
            col = B_indices_list[index]
            min_rate = resource_rate[0, col].min()
            if min_rate > 0:
                M_actual[0, index] = min_rate * mau_order[0, index]
                col2 = col[np.abs(resource_rate[0, col] - min_rate) < equal_tolerance]
                k_len = len(col2)
                k = data_type(1 / k_len) * mau_order[0, index]
                d_M_d_r_r[t - 1][index] = (k, col2)
                d_M_d_man_o[t - 1][0, index] = min_rate
                time_stamp = t - 1 + lead_time[index]
                if time_stamp < duration:
                    W_t[time_stamp, index] = M_actual[0, index]
                    d_W_d_Mqty_item[time_stamp].append(index)
        M_buffer = mau_order - M_actual
        Mbuf_flag.append(np.where(M_buffer > 0, one, zero))
        I_t = I_t - M_actual * B
        # W_qty = W_qty + M_actual
        cost = cost + np.sum(np.multiply((I_t + W_qty), holding_cost)) + np.sum(
            np.multiply(D_queue, penalty_cost))
    d_S = np.zeros((1, nodes_num), dtype=data_type)
    # d_Wqty = self.holding_cost + 0
    d_It = holding_cost + 0
    d_Dback = penalty_cost + 0
    d_Ipt = np.zeros((1, nodes_num), dtype=data_type)
    d_Mt_buffer = np.zeros((1, nodes_num), dtype=data_type)
    d_O = []
    d_W_d_Mq = []
    for i in range(duration):
        d_W_d_Mq.append(np.zeros((1, nodes_num), dtype=data_type))
        d_O.append(np.zeros((1, nodes_num), dtype=data_type))

    for tt in range(1, duration + 1):
        t = duration + 1 - tt
        d_Mact = - d_It * B_T  # + d_Wqty
        temp = np.multiply(d_Mt_buffer, Mbuf_flag[t - 1])
        d_Mq = d_Mact - temp + d_W_d_Mq[t - 1]
        d_mau_o = temp + np.multiply(d_Mq, d_M_d_man_o[t - 1])
        d_res_r = np.zeros((1, nodes_num), dtype=data_type)
        for index in d_M_d_r_r[t - 1]:
            temp_k = d_M_d_r_r[t - 1][index][0] * d_Mq[0, index]
            for c_num in d_M_d_r_r[t - 1][index][1]:
                d_res_r[0, c_num] = d_res_r[0, c_num] + temp_k

        d_It = d_It + d_res_r * d_r_r_d_I[t - 1]
        d_res_n = d_res_r * d_r_r_d_r_n[t - 1]
        d_mau_o = d_mau_o + d_res_n * B_T
        d_O[t - 1] = d_O[t - 1] + d_mau_o * mau_item_diag
        d_Yt = d_It * d_It_d_Yt[t - 1] + d_Dback * d_Dq_d_Yt[t - 1]
        d_O[t - 1] = d_O[t - 1] + d_Ipt * E_B_T
        d_temp_O = d_O[t - 1] + 0
        for i in range(stage_num - 1):
            d_S = d_S - d_temp_O * d_O_d_Ipformer[t - 1][i]
            d_temp_Ipt = d_temp_O * d_O_d_Ipformer[t - 1][i]
            d_Ipt = d_Ipt + d_temp_Ipt
            d_temp_O = -d_temp_Ipt * B_T
        d_S = d_S - d_temp_O * d_O_d_Ipformer[t - 1][stage_num - 1]
        d_Ipt = d_Ipt + d_temp_O * d_O_d_Ipformer[t - 1][stage_num - 1]
        if t > 1:
            d_Mt_buffer = d_mau_o + 0
            d_It = d_Yt + holding_cost
            d_Dback = -d_Yt + penalty_cost
            d_Wt = d_Yt + 0  # - d_Wqty
            # d_Wqty = d_Wqty + self.holding_cost
            d_P = d_Yt + 0
            for index in d_W_d_Mqty_item[t - 1]:
                lead = lead_time[index]
                d_W_d_Mq[t - 1 - lead][0, index] = d_Wt[0, index]
            for index in d_P_d_O_item[t - 1]:
                lead = d_P_d_O_lead[t - 1][index]  # self.purchase_lead[index]
                d_O[t - 1 - lead][0, index] = d_O[t - 1 - lead][0, index] + d_P[0, index]
        else:
            d_S = d_S + d_Yt
            d_S = d_S + d_Ipt
    gradient = np.array(d_S)
    return cost, gradient


def simulate_only_parallel(args):
    (I_S, duration, nodes_num, zero, one, stage_num, lead_time, data_type, B_indices_list, holding_cost,
     penalty_cost, raw_material_node, B, D_mean, std, demand_set,
     delivery_shift, rand_seed) = args

    D, D_order = generate_random_demand_for_parallel(duration, nodes_num, data_type, D_mean, std, demand_set,
                                                     delivery_shift, zero, rand_seed)

    # t_s = time()
    # initialize
    M_buffer = np.zeros((1, nodes_num), dtype=data_type)
    W_t = np.zeros((duration, nodes_num), dtype=data_type)
    P = np.zeros((duration, nodes_num), dtype=data_type)
    D_queue = np.zeros((1, nodes_num), dtype=data_type)
    W_qty = np.zeros((1, nodes_num), dtype=data_type)
    I_t = I_S + 0  # self.initial_inv+0#
    I_position = I_S + 0  # self.initial_inv+0#
    cost = data_type(0.0)
    # cost2 = data_type(0.0)
    # filled_demand = 0

    for t in range(1, duration + 1):
        I_position = I_position - D_order[t - 1, :]
        temp_O_t = -np.minimum(zero, (I_position - I_S))

        for i in range(stage_num - 1):
            temp_I_position = I_position - temp_O_t * B
            temp_O_t = -np.minimum(zero, (temp_I_position - I_S))

        O_t = temp_O_t + 0
        I_position = I_position - O_t * B + O_t
        # print('test:',I_S)
        temp_I_t = I_t - D_queue - D[t - 1] + W_t[t - 1] + P[t - 1]
        I_t = np.maximum(zero, temp_I_t)

        D_queue = -np.minimum(zero, temp_I_t)

        # W_qty = W_qty - W_t[t - 1]
        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_buffer
        idx_purch = np.nonzero(purchase_order)
        idx_purch = idx_purch[1]
        idx_mau = np.nonzero(mau_order)
        idx_mau = idx_mau[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed
        temp_resource_rate[np.isnan(temp_resource_rate)] = one

        resource_rate = np.minimum(one, temp_resource_rate)

        for index in idx_purch:
            temp_lead = max(1, int(round(
                normalvariate(lead_time[index], 0.05 * lead_time[index]))))
            time_stamp = t - 1 + temp_lead  # time_stamp = t - 1 + self.purchase_lead[index]
            if time_stamp < duration:
                P[time_stamp, index] = purchase_order[0, index]

        M_actual = np.zeros((1, nodes_num), dtype=data_type)
        for index in idx_mau:
            col = B_indices_list[index]
            min_rate = resource_rate[0, col].min()
            if min_rate > 0:
                M_actual[0, index] = min_rate * mau_order[0, index]
                time_stamp = t - 1 + lead_time[index]
                if time_stamp < duration:
                    W_t[time_stamp, index] = M_actual[0, index]

        M_buffer = mau_order - M_actual

        I_t = I_t - M_actual * B
        # W_qty = W_qty + M_actual
        cost = cost + np.sum(np.multiply((I_t + W_qty), holding_cost)) + np.sum(
            np.multiply(D_queue, penalty_cost))
        # cost2 = cost2 + np.sum(np.multiply((I_t + W_qty), holding_cost))
        # filled_demand = filled_demand + D[t - 1] - D_queue
    # D_sum = np.sum(D, axis=0)
    return cost#, cost2, filled_demand, D_sum



















