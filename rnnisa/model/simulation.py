"""
This module contains the code of simulation and gradient computation for the RNN inspired simulation approach for large-scale inventory optimization problems 
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach"

Author:
    Tan Wang
"""

import os
import numpy as np
from random import seed, normalvariate
from multiprocessing import Pool, cpu_count
from scipy.sparse import diags
from warnings import filterwarnings

filterwarnings('ignore')

CORE_NUM = cpu_count()


class Simulation():
    def __init__(self, data_type, duration, data_path, network_name, delivery_cycle=0, penalty_factor=10.0):
        self.__duration = duration
        self.__data_type = data_type
        self._prepare_data(data_path, network_name, penalty_factor, data_type, delivery_cycle)
        self.__seed_num = 0
        self._print_info()

    def _prepare_data(self, data_path, network_name, penalty_factor, data_type, delivery_cycle):
        import networkx as nx
        from scipy.sparse import eye
        from rnnisa.utils.tool_function import my_load

        def count_layer(B):
            B.eliminate_zeros()
            B = B.tocsr()
            B = B.astype(self.__data_type)
            temp = eye(B.shape[0], dtype=self.__data_type)
            temp = temp.tocsr()
            
            maxlayer = 0
            for i in range(B.shape[0]):
                
                temp = B * temp
                temp.eliminate_zeros()
                if temp.nnz == 0:
                    maxlayer = i
                    break
            return maxlayer + 1

        G = my_load(os.path.join(data_path, network_name))

        if type(G) == list:
            G = G[0]
        self.__B = nx.adjacency_matrix(G, weight='weight')
        self.__stage_num = count_layer(self.__B)
        self.__nodes_num = self.__B.shape[0]

        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type)
        self.__hold_coef = np.expand_dims(self.__hold_coef, axis=0)
        self.__lead_time = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))

        self.__D_mean = np.zeros((self.__duration, self.__nodes_num))  
        self.__std = np.zeros_like(self.__D_mean)
        in_degree_values = np.array([v for k, v in G.in_degree()])
        demand_node = np.where(in_degree_values == 0)[0]
        i = 0
        for nd in list(G.nodes()):
            if i in demand_node:
                self.__D_mean[range(self.__duration), i] = G.nodes[nd]['mean']
                self.__std[range(self.__duration), i] = G.nodes[nd]['std']
            i += 1

        self.__demand_set = demand_node

        self.__penalty_coef = data_type(penalty_factor) * self.__hold_coef
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

        out_degree_values = np.expand_dims(np.array([v for k, v in G.out_degree()]), axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)

        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])

        idx_mau = np.nonzero(1 - self.__raw_material_node)[1]
        self.__B_indices_list = {i: self.__B[i].indices for i in
                                 idx_mau}
        time_stamp = np.zeros((self.__duration, self.__nodes_num), dtype=int)

        time_stamp[:, :] = self.__lead_time
        time_stamp[:, :] = time_stamp[:, :] + np.expand_dims(np.array(list(range(self.__duration))), axis=1)
        self.__time_stamp = time_stamp
        self.__time_stamp_truncated = np.minimum(time_stamp, self.__duration)

        if type(delivery_cycle) == int:
            delivery_cycles = delivery_cycle * np.ones(self.__nodes_num, dtype=int)
        else:
            delivery_cycles = my_load(os.path.join(data_path, delivery_cycle))

        self.__delivery_shift = np.zeros_like(time_stamp)
        for t in range(self.__duration):
            self.__delivery_shift[t, self.__demand_set] = t - delivery_cycles[self.__demand_set]
        self.__delivery_shift = np.maximum(-1, self.__delivery_shift)

    def _print_info(self):
        print('Data Type:', self.__data_type)
        print('nodes number:', self.__nodes_num)

    def reset_seed(self):
        self.__seed_num = 0

    def cut_seed(self, num):
        self.__seed_num -= num

    def get_demand_set(self):
        return self.__demand_set

    def _generate_random_demand(self, rand_seed):
        if rand_seed is not None:
            seed(rand_seed)
        D = np.zeros((self.__duration, self.__nodes_num), dtype=self.__data_type)
        D_order = np.zeros((self.__duration + 1, self.__nodes_num), dtype=self.__data_type)
        D_mean = self.__D_mean
        std = self.__std
        demand_set = self.__demand_set
        delivery_shift = self.__delivery_shift
        for t in range(self.__duration):
            D_order[t, demand_set] = [normalvariate(D_mean[t, i], std[t, i]) for i in demand_set]
            D[t, demand_set] = D_order[delivery_shift[t, demand_set], demand_set]
            
        D_order = np.maximum(self.__zero, D_order)
        D = np.maximum(self.__zero, D)
        return D, D_order

    def simulate_traditional(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        zero = self.__zero
        stage_num = self.__stage_num
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated

        B = self.__B.toarray()
        nonzero = np.nonzero
        maximum = np.maximum
        minimum = np.minimum
        zeros_like = np.zeros_like

        D, D_order = self._generate_random_demand(rand_seed)

        M_backlog = np.zeros(self.__nodes_num, dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        D_backlog = zeros_like(M_backlog)
        I_t = np.squeeze(I_S) + zero
        I_position = np.squeeze(I_S) + zero
        cost = zero
        O_t = zeros_like(M_backlog)
        temp_I_position = zeros_like(M_backlog)
        purchase_order = zeros_like(M_backlog)
        mau_order = zeros_like(M_backlog)
        temp_I = zeros_like(M_backlog)
        node_range = range(self.__nodes_num)
        for t in range(duration):

            I_position[node_range] = I_position[node_range] - D_order[t, node_range]
            O_t[node_range] = -minimum(zero, (I_position[node_range] - I_S[0, node_range]))

            for _ in range(stage_num - 1):
                temp_I_position[node_range] = I_position[node_range] + zero
                for i in node_range:
                    if raw_material_node[0, i] < 1:
                        temp_I_position[node_range] = temp_I_position[node_range] - O_t[i] * B[i, node_range]
                O_t[node_range] = -minimum(zero, (temp_I_position[node_range] - I_S[0, node_range]))

            I_position[node_range] = I_position[node_range] + O_t[node_range]
            temp_I[node_range] = I_t[node_range] - D_backlog[node_range] - D[t, node_range] + P[
                t, node_range]
            I_t[node_range] = maximum(zero, temp_I[node_range])
            D_backlog[node_range] = -minimum(zero, temp_I[node_range])
            purchase_order[node_range] = O_t[node_range] * raw_material_node[0, node_range]
            mau_order[node_range] = O_t[node_range] - purchase_order[node_range] + M_backlog[node_range]
            idx_purch = nonzero(purchase_order)[0]
            idx_mau = nonzero(mau_order)[0]
            for i in idx_mau:
                I_position[node_range] = I_position[node_range] - O_t[i] * B[i, node_range]
            resource_needed = zeros_like(M_backlog)
            for i in idx_mau:
                resource_needed[node_range] = resource_needed[node_range] + mau_order[i] * B[i, node_range]

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[idx_purch]
            M_actual = zeros_like(M_backlog)
            M_backlog = zeros_like(M_actual)
            for index in idx_mau:
                min_rate = 1
                for j in node_range:
                    if B[index, j] > 0:
                        min_rate = min(min_rate, I_t[j] / resource_needed[j])
                if min_rate > 0:
                    M_actual[index] = min_rate * mau_order[index]
                    M_backlog[index] = (1 - min_rate) * mau_order[index]
                else:
                    M_backlog[index] = mau_order[index] + zero
            for i in idx_mau:
                if M_actual[i] > 0:
                    I_t[node_range] = I_t[node_range] - M_actual[i] * B[i, node_range]

            P[time_stamp[t, idx_mau], idx_mau] = M_actual[idx_mau]

            cost = cost + sum(
                I_t[node_range] * holding_cost[0, node_range] + D_backlog[node_range] * penalty_cost[0, node_range])

        if print_flag:
            print('total_cost: ', cost)
        return cost

    def simulate(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        zero = self.__zero
        one = self.__one
        stage_num = self.__stage_num
        B = self.__B
        B_indices_list = self.__B_indices_list
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated
        maximum = np.maximum
        minimum = np.minimum
        np_isnan = np.isnan
        nonzero = np.nonzero
        zeros_like = np.zeros_like
        np_sum = np.sum
        np_multiply = np.multiply
        np_array = np.array

        D, D_order = self._generate_random_demand(rand_seed)

        M_backlog = np.zeros((1, self.__nodes_num), dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        D_backlog = zeros_like(M_backlog)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        for t in range(duration):
            I_position -= D_order[t, :]  
            O_t = -minimum(zero, (I_position - I_S))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - O_t * B
                O_t = -minimum(zero, (temp_I_position - I_S))
            I_position += O_t - O_t * B  
            temp_I_t = I_t - D_backlog - D[t] + P[t] 
            I_t = maximum(zero, temp_I_t)
            D_backlog = -minimum(zero, temp_I_t)
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = mau_order * B
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[np_isnan(temp_resource_rate)] = one
            resource_rate = minimum(one, temp_resource_rate)

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]

            M_actual = zeros_like(M_backlog)  
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]

            M_backlog = mau_order - M_actual
            I_t -= M_actual * B  
            
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))  
            
        if print_flag:
            print('total_cost: ', cost)
            
        return cost

    def simulate_dense(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        zero = self.__zero
        one = self.__one
        stage_num = self.__stage_num
        B = self.__B.toarray()
        B_indices_list = self.__B_indices_list
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated  
        
        maximum = np.maximum
        minimum = np.minimum
        np_isnan = np.isnan
        nonzero = np.nonzero
        zeros_like = np.zeros_like
        np_sum = np.sum
        np_multiply = np.multiply
        np_dot = np.dot
        np_array = np.array

        D, D_order = self._generate_random_demand(rand_seed)
        
        M_backlog = np.zeros((1, self.__nodes_num), dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        
        D_backlog = zeros_like(M_backlog)
        
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - np_dot(O_t, B)
                O_t = -minimum(zero, (temp_I_position - I_S))
            I_position = I_position - np_dot(O_t, B) + O_t

            temp_I_t = I_t - D_backlog - D[t] + P[t] 
            I_t = maximum(zero, temp_I_t)
            D_backlog = -minimum(zero, temp_I_t)
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = np_dot(mau_order, B)
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[np_isnan(temp_resource_rate)] = one
            resource_rate = minimum(one, temp_resource_rate)

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]

            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]

            M_backlog = mau_order - M_actual
            I_t = I_t - np_dot(M_actual, B)
            
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))
        if print_flag:
            print('total_cost: ', cost)
        return cost

    def simulate_and_bp(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        nodes_num = self.__nodes_num
        zero = self.__zero
        one = self.__one
        one_minus = self.__one_minus
        stage_num = self.__stage_num
        
        lead_time = self.__lead_time
        data_type = self.__data_type
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        mau_item_diag = self.__mau_item_diag
        raw_material_node = self.__raw_material_node
        B = self.__B
        B_T = self.__B_T
        E_B_T = self.__E_B_T
        time_stamp = self.__time_stamp_truncated  
        
        maximum = np.maximum
        minimum = np.minimum
        where = np.where
        nonzero = np.nonzero
        np_abs = np.abs
        zeros_like = np.zeros_like
        np_sum = np.sum
        np_multiply = np.multiply
        np_array = np.array

        D, D_order = self._generate_random_demand(rand_seed)
        
        M_backlog = np.zeros((1, nodes_num), dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        
        D_backlog = zeros_like(M_backlog)
        
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_It_d_Yt = []
        d_Dback_d_Yt = []
        d_O_d_Ipformer = [[] for _ in range(duration)]
        
        d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]  
        d_M_d_r_r = [{} for _ in range(duration)]
        d_r_r_d_I = []
        d_r_r_d_r_n = []
        
        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            flag = where((I_position - I_S) < 0, one_minus, zero)
            d_O_d_Ipformer[t].insert(0, diags(flag[0]))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - O_t * B
                O_t = -minimum(zero, (temp_I_position - I_S))
                flag = where((temp_I_position - I_S) < 0, one_minus, zero)
                d_O_d_Ipformer[t].insert(0, diags(flag[0]))
            I_position = I_position - O_t * B + O_t

            temp_I_t = I_t - D_backlog - D[t] + P[t] 
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_It_d_Yt.append(diags(flag[0]))
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t <= 0, one_minus, zero)
            d_Dback_d_Yt.append(diags(flag[0]))
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = mau_order * B
            temp_resource_rate = I_t / resource_needed
            
            temp_resource_rate[resource_needed == 0] = one
            temp1 = one / resource_needed
            temp1[resource_needed == 0] = one
            temp2 = -np_multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            d_r_r_d_I.append(
                diags(np_multiply(flag2, temp1)[
                          0]))  
            d_r_r_d_r_n.append(
                diags(np_multiply(flag2, temp2)[
                          0]))  

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                               < equal_tolerance] for i in range(len(idx_mau))]
            d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i]) for i
                            in range(len(idx_mau)) if min_rate[i] > 0}
            d_M_d_man_o[t][0, idx_mau] = min_rate + zero
            
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
            M_backlog = mau_order - M_actual
            I_t = I_t - M_actual * B
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))  
        d_S = zeros_like(M_backlog)
        
        d_It = holding_cost + zero
        d_Dback = penalty_cost + zero
        d_Ipt = zeros_like(M_backlog)
        d_Mt_backlog = zeros_like(M_backlog)
        
        d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)
        
        d_P_d_Mq = zeros_like(d_O)

        for t in range(duration - 1, -1, -1):
            d_Mact = - d_It * B_T  
            d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]  
            d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])  
            d_res_r = zeros_like(M_backlog)
            
            for index in d_M_d_r_r[t]:
                temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
                col2_list = d_M_d_r_r[t][index][1]
                d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k
                 
            d_It = d_It + d_res_r * d_r_r_d_I[t]
            d_res_n = d_res_r * d_r_r_d_r_n[t]
            d_mau_o = d_mau_o + d_res_n * B_T
            d_O[t] = d_O[t] + d_mau_o * mau_item_diag
            d_Yt = d_It * d_It_d_Yt[t] + d_Dback * d_Dback_d_Yt[t]
            d_O[t] = d_O[t] + d_Ipt * E_B_T
            d_temp_O = d_O[t] + zero
            for i in range(stage_num - 1):
                d_temp_Ipt = d_temp_O * d_O_d_Ipformer[t][i]
                d_S = d_S - d_temp_Ipt
                d_Ipt = d_Ipt + d_temp_Ipt
                d_temp_O = -d_temp_Ipt * B_T
            temp_d_Ipt = d_temp_O * d_O_d_Ipformer[t][stage_num - 1]
            d_S = d_S - temp_d_Ipt
            d_Ipt = d_Ipt + temp_d_Ipt
            if t > 0:
                d_Mt_backlog = d_mau_o + zero
                d_It = d_Yt + holding_cost
                d_Dback = -d_Yt + penalty_cost
                
                d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
                d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]

                d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
                d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]

            else:
                d_S = d_S + d_Yt
                d_S = d_S + d_Ipt
        
        if print_flag:
            _print_cost_grad_info(cost, d_S) 

        return cost, d_S

    def simulate_and_bp_dense(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        nodes_num = self.__nodes_num
        zero = self.__zero
        one = self.__one
        one_minus = self.__one_minus
        stage_num = self.__stage_num
        
        lead_time = self.__lead_time
        data_type = self.__data_type
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        mau_item_diag = self.__mau_item_diag
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated  
        
        B = self.__B.toarray()
        B_T = self.__B_T.toarray()
        E_B_T = self.__E_B_T.toarray()
        minimum = np.minimum
        maximum = np.maximum
        where = np.where
        nonzero = np.nonzero
        np_abs = np.abs
        zeros_like = np.zeros_like
        np_sum = np.sum
        np_multiply = np.multiply
        np_array = np.array
        np_dot = np.dot

        D, D_order = self._generate_random_demand(rand_seed)
        
        M_backlog = np.zeros((1, nodes_num), dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        
        D_backlog = zeros_like(M_backlog)
        
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero
        d_It_d_Yt = []
        d_Dback_d_Yt = []
        d_O_d_Ipformer = [[] for _ in range(duration)]
        d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]  
        d_M_d_r_r = [{} for _ in range(duration)]
        d_r_r_d_I = []
        d_r_r_d_r_n = []
        
        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            flag = where((I_position - I_S) < 0, one_minus, zero)
            d_O_d_Ipformer[t].insert(0, diags(flag[0]))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - np_dot(O_t, B)
                O_t = -minimum(zero, (temp_I_position - I_S))
                flag = where((temp_I_position - I_S) < 0, one_minus, zero)
                d_O_d_Ipformer[t].insert(0, diags(flag[0]))
            I_position = I_position - np_dot(O_t, B) + O_t

            temp_I_t = I_t - D_backlog - D[t] + P[t]
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_It_d_Yt.append(diags(flag[0]))
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dback_d_Yt.append(diags(flag[0]))
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = np_dot(mau_order, B)
            temp_resource_rate = I_t / resource_needed
            
            temp_resource_rate[resource_needed == 0] = one
            temp1 = one / resource_needed
            temp1[resource_needed == 0] = one
            temp2 = -np_multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            d_r_r_d_I.append(
                diags(np_multiply(flag2, temp1)[
                          0]))  
            d_r_r_d_r_n.append(
                diags(np_multiply(flag2, temp2)[
                          0]))  

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                               < equal_tolerance] for i in range(len(idx_mau))]
            d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i])
                            for i in range(len(idx_mau)) if min_rate[i] > 0}
            d_M_d_man_o[t][0, idx_mau] = min_rate + zero
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]

            M_backlog = mau_order - M_actual
            
            I_t = I_t - np_dot(M_actual, B)
            
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))
        d_S = zeros_like(M_backlog)
        
        d_It = holding_cost + zero
        d_Dback = penalty_cost + zero
        d_Ipt = zeros_like(M_backlog)
        d_Mt_backlog = zeros_like(M_backlog)
        
        d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)
        
        d_P_d_Mq = zeros_like(d_O)

        for t in range(duration - 1, -1, -1):
            
            d_Mact = - np_dot(d_It, B_T)  
            
            d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]  
            d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])  
            d_res_r = zeros_like(M_backlog)
            
            for index in d_M_d_r_r[t]:
                temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
                col2_list = d_M_d_r_r[t][index][1]
                d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k
                
            d_It = d_It + d_res_r * d_r_r_d_I[t]
            d_res_n = d_res_r * d_r_r_d_r_n[t]
            d_mau_o = d_mau_o + np_dot(d_res_n, B_T)
            d_O[t] = d_O[t] + d_mau_o * mau_item_diag
            d_Yt = d_It * d_It_d_Yt[t] + d_Dback * d_Dback_d_Yt[t]
            d_O[t] = d_O[t] + np_dot(d_Ipt, E_B_T)
            d_temp_O = d_O[t] + zero
            for i in range(stage_num - 1):
                d_temp_Ipt = d_temp_O * d_O_d_Ipformer[t][i]
                d_S = d_S - d_temp_Ipt
                d_Ipt = d_Ipt + d_temp_Ipt
                d_temp_O = -np_dot(d_temp_Ipt, B_T)
            temp_d_Ipt = d_temp_O * d_O_d_Ipformer[t][stage_num - 1]
            d_S = d_S - temp_d_Ipt
            d_Ipt = d_Ipt + temp_d_Ipt
            if t > 0:
                d_Mt_backlog = d_mau_o + zero
                d_It = d_Yt + holding_cost
                d_Dback = -d_Yt + penalty_cost
                
                d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
                d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]

                d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
                d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]

            else:
                d_S = d_S + d_Yt
                d_S = d_S + d_Ipt
        
        if print_flag:
            _print_cost_grad_info(cost, d_S)

        return cost, d_S

    def simulate_and_IPA(self, I_S, rand_seed=None, print_flag=False):
        from scipy.sparse import csr_matrix, lil_matrix
        def set_row_lil(A, row_idx, new_row_csr):
            A.rows[row_idx] = new_row_csr.indices.tolist()
            A.data[row_idx] = new_row_csr.data.tolist()
            return A

        duration = self.__duration
        stage_num = self.__stage_num
        data_type = self.__data_type
        zero = self.__zero
        one = self.__one
        one_minus = self.__one_minus
        E = self.__E
        B = self.__B
        B_T = self.__B_T
        E_B_T = self.__E_B_T
        raw_material_node = self.__raw_material_node
        mau_item_diag = self.__mau_item_diag
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp
        
        matrix_shape1 = (self.__nodes_num, self.__nodes_num)
        vector_shape = (1, self.__nodes_num)
        np_sum = np.sum
        multiply = np.multiply
        minimum = np.minimum
        maximum = np.maximum
        np_abs = np.abs
        zeros_like = np.zeros_like
        isnan = np.isnan
        where = np.where
        nonzero = np.nonzero
        np_array = np.array
        nan = np.nan

        D, D_order = self._generate_random_demand(rand_seed)
        
        M_backlog = np.zeros(vector_shape, dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        
        D_backlog = zeros_like(M_backlog)
        
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_I = E + zero
        d_Dbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        
        d_P = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_Iposition = E + zero
        d_Mbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        d_cost = zeros_like(M_backlog)

        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            flag = where(I_position - I_S < 0, one_minus, zero)
            d_O = diags(flag[0]) * (d_Iposition - E)
            for _ in range(stage_num - 1):
                temp_I_position = I_position - O_t * B
                d_temp_Iposition = d_Iposition - B_T * d_O
                O_t = -minimum(zero, (temp_I_position - I_S))
                flag = where(temp_I_position - I_S < 0, one_minus, zero)
                d_O = diags(flag[0]) * (d_temp_Iposition - E)

            I_position = I_position - O_t * B + O_t
            d_Iposition = d_Iposition + E_B_T * d_O
            temp_I_t = I_t - D_backlog - D[t] + P[t]
            d_tempI = d_I - d_Dbacklog + d_P[t].tocsr() 
            
            d_P[t] = nan
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_I = diags(flag[0]) * d_tempI
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dbacklog = diags(flag[0]) * d_tempI
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            d_mau_o = d_Mbacklog + mau_item_diag * d_O
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = mau_order * B
            d_res_n = B_T * d_mau_o
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[isnan(temp_resource_rate)] = one
            temp1 = one / resource_needed
            temp2 = -multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            dia_1 = diags(multiply(flag2, temp1)[0])
            dia_2 = diags(multiply(flag2, temp2)[0])
            d_res_rate = dia_1 * d_I + dia_2 * d_res_n

            P[minimum(duration, time_stamp[t, idx_purch]), idx_purch] = purchase_order[0, idx_purch]
            d_O_getrow = d_O.getrow
            for index in idx_purch:
                if time_stamp[t, index] < duration:
                    d_P[time_stamp[t, index]] = set_row_lil(d_P[time_stamp[t, index]], index, d_O_getrow(index))

            M_actual = zeros_like(M_backlog)  
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[minimum(duration, time_stamp[t, idx_mau]), idx_mau] = M_actual[0, idx_mau]

            d_Mact = lil_matrix(matrix_shape1, dtype=data_type)
            for i in range(len(idx_mau)):

                if min_rate[i] > 0:
                    index = idx_mau[i]
                    col = B_indices_list[index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                    k_len = len(col2)
                    d_min_rate = csr_matrix(([data_type(1.0 / k_len)] * k_len, ([0] * k_len, col2)), shape=vector_shape,
                                            dtype=data_type) * d_res_rate
                    
                    temp = mau_order[0, index] * d_min_rate + min_rate[i] * d_mau_o[index]
                    d_Mact = set_row_lil(d_Mact, index, temp)

                    if time_stamp[t, index] < duration:
                        d_P[time_stamp[t, index]].rows[index] = d_Mact.rows[index]  
                        d_P[time_stamp[t, index]].data[index] = d_Mact.data[index]  
            d_Mact = d_Mact.tocsr()
            M_backlog = mau_order - M_actual
            d_Mbacklog = d_mau_o - d_Mact
            I_t = I_t - M_actual * B
            d_I = d_I - B_T * d_Mact
            
            cost = cost + np_sum(multiply(I_t, holding_cost)) + np_sum(
                multiply(D_backlog, penalty_cost))  
            
            d_cost = d_cost + holding_cost * d_I + penalty_cost * d_Dbacklog
        
        if print_flag:
            _print_cost_grad_info(cost, d_cost)
        return cost, d_cost

    def simulate_and_IPA_dense(self, I_S, rand_seed=None, print_flag=False):
        from scipy.sparse import csr_matrix, lil_matrix
        def set_row_lil(A, row_idx, new_row_csr):
            A.rows[row_idx] = new_row_csr.indices.tolist()
            A.data[row_idx] = new_row_csr.data.tolist()
            return A

        duration = self.__duration
        stage_num = self.__stage_num
        data_type = self.__data_type
        zero = self.__zero
        one = self.__one
        one_minus = self.__one_minus
        E = self.__E
        B = self.__B.toarray()
        B_T = self.__B_T.toarray()
        
        E_B_T = (E - B).T
        raw_material_node = self.__raw_material_node
        mau_item_diag = self.__mau_item_diag
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp
        
        matrix_shape1 = (self.__nodes_num, self.__nodes_num)
        vector_shape = (1, self.__nodes_num)
        np_sum = np.sum
        multiply = np.multiply
        minimum = np.minimum
        maximum = np.maximum
        np_abs = np.abs
        zeros_like = np.zeros_like
        isnan = np.isnan
        where = np.where
        nonzero = np.nonzero
        np_array = np.array
        np_dot = np.dot
        nan = np.nan

        D, D_order = self._generate_random_demand(rand_seed)
        
        M_backlog = np.zeros(vector_shape, dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        
        D_backlog = zeros_like(M_backlog)
        
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_I = E + zero
        d_Dbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        
        d_P = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_Iposition = E + zero
        d_Mbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        d_cost = zeros_like(M_backlog)

        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            flag = where(I_position - I_S < 0, one_minus, zero)
            d_O = diags(flag[0]) * (d_Iposition - E)
            for _ in range(stage_num - 1):
                temp_I_position = I_position - np_dot(O_t, B)
                d_temp_Iposition = d_Iposition - B_T * d_O
                O_t = -minimum(zero, (temp_I_position - I_S))
                flag = where(temp_I_position - I_S < 0, one_minus, zero)
                d_O = diags(flag[0]) * (d_temp_Iposition - E)

            I_position = I_position - np_dot(O_t, B) + O_t
            d_Iposition = d_Iposition + E_B_T * d_O
            temp_I_t = I_t - D_backlog - D[t] + P[t] 
            d_tempI = d_I - d_Dbacklog + d_P[t].tocsr() 
            
            d_P[t] = nan
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_I = diags(flag[0]) * d_tempI
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dbacklog = diags(flag[0]) * d_tempI
            
            purchase_order = multiply(O_t, raw_material_node)
            mau_order = O_t - purchase_order + M_backlog
            d_mau_o = d_Mbacklog + mau_item_diag * d_O
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = np_dot(mau_order, B)
            d_res_n = B_T * d_mau_o
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[isnan(temp_resource_rate)] = one
            temp1 = one / resource_needed
            temp2 = -multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            dia_1 = diags(multiply(flag2, temp1)[0])
            dia_2 = diags(multiply(flag2, temp2)[0])
            
            d_res_rate = dia_1 * d_I + dia_2 * d_res_n

            P[minimum(duration, time_stamp[t, idx_purch]), idx_purch] = purchase_order[0, idx_purch]
            d_O = csr_matrix(d_O)
            d_O_getrow = d_O.getrow
            for index in idx_purch:
                
                if time_stamp[t, index] < duration:
                    
                    d_P[time_stamp[t, index]] = set_row_lil(d_P[time_stamp[t, index]], index, d_O_getrow(index))
                    
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[minimum(duration, time_stamp[t, idx_mau]), idx_mau] = M_actual[0, idx_mau]

            d_Mact = lil_matrix(matrix_shape1, dtype=data_type)
            for i in range(len(idx_mau)):

                if min_rate[i] > 0:
                    index = idx_mau[i]
                    col = B_indices_list[index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                    k_len = len(col2)
                    d_min_rate = csr_matrix(([data_type(1.0 / k_len)] * k_len, ([0] * k_len, col2)), shape=vector_shape,
                                            dtype=data_type) * d_res_rate
                    
                    temp = csr_matrix(mau_order[0, index] * d_min_rate + min_rate[i] * d_mau_o[index])
                    d_Mact = set_row_lil(d_Mact, index, temp)
                    
                    if time_stamp[t, index] < duration:
                        
                        d_P[time_stamp[t, index]].rows[index] = d_Mact.rows[index]  
                        d_P[time_stamp[t, index]].data[index] = d_Mact.data[index]  
            d_Mact = d_Mact.tocsr()
            M_backlog = mau_order - M_actual
            d_Mbacklog = d_mau_o - d_Mact
            I_t = I_t - np_dot(M_actual, B)
            d_I = d_I - B_T * d_Mact
            
            cost = cost + np_sum(multiply(I_t, holding_cost)) + np_sum(
                multiply(D_backlog, penalty_cost))  
            
            d_cost = d_cost + holding_cost * d_I + penalty_cost * d_Dbacklog
        gradient = np_array(d_cost)
        if print_flag:
            _print_cost_grad_info(cost, gradient)
        return cost, gradient

    def _get_tf_B(self):
        import tensorflow as tf
        temp_B = self.__B.tocoo()
        indices = np.mat([temp_B.row, temp_B.col]).transpose()
        idx_mau = np.nonzero(1 - self.__raw_material_node)[1]
        tf_B = tf.SparseTensor(indices, temp_B.data, temp_B.shape)
        tf_B_sparse_split = tf.sparse.split(sp_input=tf_B, num_split=self.__nodes_num, axis=0)
        tf_B_indices_list = {i: tf_B_sparse_split[i].indices for i in idx_mau}
        tf_B = tf.sparse.reorder(tf_B)
        return tf_B, tf_B_indices_list

    def simulate_tf(self, I_S, GPU_flag, dense_flag=False, rand_seed=None, print_flag=False):
        import tensorflow as tf
        duration = self.__duration
        stage_num = self.__stage_num
        one = self.__one
        one_small = np.int8(1)
        zero = self.__zero
        equal_tolerance = self.__equal_tole
        raw_material_node = self.__raw_material_node
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp_truncated
        vector_shape = [1, self.__nodes_num]
        tf_maximum = tf.maximum
        tf_minimum = tf.minimum
        reduce_sum = tf.reduce_sum
        where = tf.where
        reduce_min = tf.reduce_min
        gather_nd = tf.gather_nd
        update = tf.tensor_scatter_nd_update
        scatter_nd = tf.scatter_nd

        D, D_order = self._generate_random_demand(rand_seed)
        P_values = np.zeros((duration + 1, duration, self.__nodes_num), dtype=np.int8)
        cost = zero

        if GPU_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            physical_devices = tf.config.list_physical_devices('GPU')
            for device_gpu in physical_devices:
                tf.config.experimental.set_memory_growth(device_gpu, True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.experimental.set_synchronous_execution(enable=False)

        tf_B, tf_B_indices_list = self._get_tf_B()
        if dense_flag:
            tf_matmul = tf.matmul
            tf_B = tf.sparse.to_dense(tf_B)
        else:
            tf_matmul = tf.sparse.sparse_dense_matmul
  
        tf_I_S = tf.convert_to_tensor(I_S, self.__data_type)
        M_backlog = tf.zeros(vector_shape, dtype=self.__data_type)

        D_backlog = tf.zeros_like(M_backlog)

        P_history = tf.zeros([duration, self.__nodes_num], dtype=self.__data_type)

        I_t = tf_I_S + zero
        I_position = tf_I_S + zero
        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = - tf_minimum(zero, (I_position - tf_I_S))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - tf_matmul(O_t, tf_B)
                O_t = - tf_minimum(zero, (temp_I_position - tf_I_S))
            I_position = I_position + O_t - tf_matmul(O_t, tf_B)
            temp_I_t = I_t - D_backlog - D[t] + reduce_sum(P_values[t] * P_history, axis=0)
                       
            I_t = tf_maximum(zero, temp_I_t)
            D_backlog = -tf_minimum(zero, temp_I_t)
            
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = where(purchase_order > 0)[:, 1].numpy()
            idx_mau = where(mau_order > 0).numpy()  

            resource_needed = tf_matmul(mau_order, tf_B)
            resource_needed = tf_maximum(equal_tolerance, resource_needed)
            resource_rate = I_t / resource_needed
            resource_rate = tf_minimum(one, resource_rate)

            min_rate = scatter_nd(idx_mau, [reduce_min(gather_nd(resource_rate,
                                                                 tf_B_indices_list[index])) for index in idx_mau[:, 1]],
                                  vector_shape)
            M_act = min_rate * mau_order

            P_history = update(P_history, [[t]], purchase_order+M_act)

            idx_mau2 = where(M_act > 0)[:, 1].numpy()  
            P_values[time_stamp[t, idx_purch], t, idx_purch] = one_small
            P_values[time_stamp[t, idx_mau2], t, idx_mau2] = one_small
            M_backlog = mau_order - M_act

            I_t = I_t - tf_matmul(M_act, tf_B)
            
            cost = cost + reduce_sum(I_t * holding_cost) + reduce_sum(
                D_backlog * penalty_cost)
        cost = cost.numpy()
        if print_flag:
            print('total_cost: ', cost)
        return cost

    def simulate_and_bp_tf(self, I_S, GPU_flag, dense_flag=False, rand_seed=None, print_flag=False):
        import tensorflow as tf
        duration = self.__duration
        stage_num = self.__stage_num
        one = self.__one
        one_small = np.int8(1)
        zero = self.__zero
        equal_tolerance = self.__equal_tole
        raw_material_node = self.__raw_material_node
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp_truncated
        vector_shape = [1, self.__nodes_num]
        tf_maximum = tf.maximum
        tf_minimum = tf.minimum
        reduce_sum = tf.reduce_sum
        where = tf.where
        reduce_min = tf.reduce_min
        gather_nd = tf.gather_nd
        update = tf.tensor_scatter_nd_update
        scatter_nd = tf.scatter_nd

        D, D_order = self._generate_random_demand(rand_seed)
        P_values = np.zeros((duration + 1, duration, self.__nodes_num), dtype=np.int8)
        cost = zero

        if GPU_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            physical_devices = tf.config.list_physical_devices('GPU')
            for device_gpu in physical_devices:
                tf.config.experimental.set_memory_growth(device_gpu, True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.experimental.set_synchronous_execution(enable=False)


        tf_B, tf_B_indices_list = self._get_tf_B()
        if dense_flag:
            tf_matmul = tf.matmul
            tf_B = tf.sparse.to_dense(tf_B)
        else:
            tf_matmul = tf.sparse.sparse_dense_matmul

        tf_I_S = tf.convert_to_tensor(I_S, self.__data_type)
        M_backlog = tf.zeros(vector_shape, dtype=self.__data_type)

        D_backlog = tf.zeros_like(M_backlog)
        P_history = tf.zeros([duration, self.__nodes_num], dtype=self.__data_type)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tf_I_S)
            I_t = tf_I_S + zero
            I_position = tf_I_S + zero
            for t in range(duration):
                I_position = I_position - D_order[t, :]
                O_t = - tf_minimum(zero, (I_position - tf_I_S))
                for _ in range(stage_num - 1):
                    temp_I_position = I_position - tf_matmul(O_t, tf_B)
                    O_t = - tf_minimum(zero, (temp_I_position - tf_I_S))
                I_position = I_position + O_t - tf_matmul(O_t, tf_B)
                temp_I_t = I_t - D_backlog - D[t] + reduce_sum(P_values[t] * P_history, axis=0) 
                I_t = tf_maximum(zero, temp_I_t)
                D_backlog = -tf_minimum(zero, temp_I_t)
                
                purchase_order = O_t * raw_material_node
                mau_order = O_t - purchase_order + M_backlog
                with tape.stop_recording():
                    idx_purch = where(purchase_order > 0)[:, 1].numpy()
                    idx_mau = where(mau_order > 0).numpy()  

                resource_needed = tf_matmul(mau_order, tf_B)
                resource_needed = tf_maximum(equal_tolerance, resource_needed)
                resource_rate = I_t / resource_needed
                resource_rate = tf_minimum(one, resource_rate)

                min_rate = scatter_nd(idx_mau, [reduce_min(gather_nd(resource_rate,
                                                                     tf_B_indices_list[index])) for index in
                                                idx_mau[:, 1]],
                                      vector_shape)
                M_act = min_rate * mau_order

                P_history = update(P_history, [[t]], purchase_order+M_act)

                with tape.stop_recording():
                    idx_mau2 = where(M_act > 0)[:, 1].numpy()  
                    P_values[time_stamp[t, idx_purch], t, idx_purch] = one_small
                    P_values[time_stamp[t, idx_mau2], t, idx_mau2] = one_small
                M_backlog = mau_order - M_act

                I_t = I_t - tf_matmul(M_act, tf_B)
                
                cost = cost + reduce_sum(I_t * holding_cost) + reduce_sum(
                    D_backlog * penalty_cost)

        gradient = tape.gradient(cost, tf_I_S)
        gradient = gradient.numpy()
        cost = cost.numpy()
        if print_flag:
            _print_cost_grad_info(cost, gradient)
        return cost, gradient

    def evaluate_cost(self, I_S, eval_num=30):
        process_num = min(CORE_NUM, eval_num)
        if self.__nodes_num == 500000: process_num = min(process_num, 30)
        if self.__nodes_num == 100000: process_num = min(process_num, 50)

        I_S_list = [(I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__stage_num
                     , self.__data_type, self.__B_indices_list, self.__hold_coef, self.__penalty_coef,
                     self.__raw_material_node, self.__B, self.__time_stamp_truncated, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_only_parallel, I_S_list)

        cost = np.mean(result)

        return cost

    def evaluate_cost_gradient(self, I_S, eval_num=30, mean_flag=True):
        process_num = min(CORE_NUM, eval_num)
        I_S_list = [(I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num
                     , self.__lead_time, self.__data_type, self.__B_indices_list, self.__equal_tole
                     , self.__hold_coef, self.__penalty_coef, self.__mau_item_diag, self.__raw_material_node, self.__B
                     , self.__B_T, self.__E_B_T, self.__time_stamp_truncated, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_and_bp_parallel, I_S_list)
            
            
        result = list(zip(*result))
        cost_result = np.array(result[0])
        grad_result = np.squeeze(result[1])
        if mean_flag:
            cost_result = np.mean(cost_result)
            grad_result = np.mean(grad_result, axis=0,
                                  keepdims=True)
        return cost_result, grad_result



def _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                     zero, rand_seed):
    if rand_seed is not None:
        seed(rand_seed)
    D = np.zeros((duration, nodes_num), dtype=data_type)
    D_order = np.zeros((duration + 1, nodes_num), dtype=data_type)
    for t in range(duration):
        D_order[t, demand_set] = [normalvariate(D_mean[t, i], std[t, i]) for i in demand_set]
        D[t, demand_set] = D_order[delivery_shift[t, demand_set], demand_set]

    D_order = np.maximum(zero, D_order)
    D = np.maximum(zero, D)
    return D, D_order


def _simulate_and_bp_parallel(args):
    (I_S, duration, nodes_num, zero, one, one_minus, stage_num, lead_time, data_type
     , B_indices_list, equal_tole, hold_coef, penalty_coef, mau_item_diag, raw_material_node
     , B, B_T, E_B_T, time_stamp, D_mean, std, demand_set, delivery_shift, rand_seed) = args
    minimum = np.minimum
    maximum = np.maximum
    where = np.where

    nonzero = np.nonzero
    np_abs = np.abs
    zeros_like = np.zeros_like
    np_sum = np.sum
    np_multiply = np.multiply
    np_array = np.array

    D, D_order = _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                                  zero, rand_seed)

    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)

    D_backlog = zeros_like(M_backlog)

    I_t = I_S + zero
    I_position = I_S + zero
    cost = zero

    d_It_d_Yt = []
    d_Dback_d_Yt = []
    d_O_d_Ipformer = [[] for _ in range(duration)]

    d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]
    d_M_d_r_r = [{} for _ in range(duration)]
    d_r_r_d_I = []
    d_r_r_d_r_n = []


    for t in range(duration):
        I_position = I_position - D_order[t, :]
        O_t = -minimum(zero, (I_position - I_S))
        flag = where((I_position - I_S) < 0, one_minus, zero)
        d_O_d_Ipformer[t].insert(0, diags(flag[0]))
        for _ in range(stage_num - 1):
            temp_I_position = I_position - O_t * B
            O_t = -minimum(zero, (temp_I_position - I_S))
            flag = where((temp_I_position - I_S) < 0, one_minus, zero)
            d_O_d_Ipformer[t].insert(0, diags(flag[0]))
        I_position = I_position - O_t * B + O_t

        temp_I_t = I_t - D_backlog - D[t] + P[t]
        I_t = maximum(zero, temp_I_t)
        flag = where(temp_I_t > 0, one, zero)
        d_It_d_Yt.append(diags(flag[0]))
        D_backlog = -minimum(zero, temp_I_t)
        flag = where(temp_I_t <= 0, one_minus, zero)
        d_Dback_d_Yt.append(diags(flag[0]))

        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_backlog
        idx_purch = nonzero(purchase_order)[1]
        idx_mau = nonzero(mau_order)[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed

        temp_resource_rate[resource_needed == 0] = one

        temp1 = one / resource_needed
        temp1[resource_needed == 0] = one
        temp2 = -np_multiply(temp_resource_rate, temp1)
        resource_rate = minimum(one, temp_resource_rate)
        flag2 = where(temp_resource_rate < 1, one, zero)
        d_r_r_d_I.append(
            diags(np_multiply(flag2, temp1)[
                      0]))
        d_r_r_d_r_n.append(
            diags(np_multiply(flag2, temp2)[
                      0]))

        P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
        M_actual = zeros_like(M_backlog)
        min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
        col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                           < equal_tole] for i in range(len(idx_mau))]
        d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i]) for i
                        in range(len(idx_mau)) if min_rate[i] > 0}
        d_M_d_man_o[t][0, idx_mau] = min_rate + zero

        P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
        M_backlog = mau_order - M_actual

        I_t = I_t - M_actual * B

        cost = cost + np_sum(np_multiply(I_t, hold_coef)) + np_sum(
            np_multiply(D_backlog, penalty_coef))
    d_S = zeros_like(M_backlog)

    d_It = hold_coef + zero
    d_Dback = penalty_coef + zero
    d_Ipt = zeros_like(M_backlog)
    d_Mt_backlog = zeros_like(M_backlog)

    d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)

    d_P_d_Mq = zeros_like(d_O)

    for t in range(duration - 1, -1, -1):
        d_Mact = - d_It * B_T
        d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]
        d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])
        d_res_r = zeros_like(M_backlog)
        for index in d_M_d_r_r[t]:
            temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
            col2_list = d_M_d_r_r[t][index][1]
            d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k

        d_It = d_It + d_res_r * d_r_r_d_I[t]
        d_res_n = d_res_r * d_r_r_d_r_n[t]
        d_mau_o = d_mau_o + d_res_n * B_T
        d_O[t] = d_O[t] + d_mau_o * mau_item_diag
        d_Yt = d_It * d_It_d_Yt[t] + d_Dback * d_Dback_d_Yt[t]
        d_O[t] = d_O[t] + d_Ipt * E_B_T
        d_temp_O = d_O[t] + zero
        for i in range(stage_num - 1):
            d_temp_Ipt = d_temp_O * d_O_d_Ipformer[t][i]
            d_S = d_S - d_temp_Ipt
            d_Ipt = d_Ipt + d_temp_Ipt
            d_temp_O = -d_temp_Ipt * B_T
        temp_d_Ipt = d_temp_O * d_O_d_Ipformer[t][stage_num - 1]
        d_S = d_S - temp_d_Ipt
        d_Ipt = d_Ipt + temp_d_Ipt
        if t > 0:
            d_Mt_backlog = d_mau_o + zero
            d_It = d_Yt + hold_coef
            d_Dback = -d_Yt + penalty_coef

            d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
            d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]

            d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
            d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]

        else:
            d_S = d_S + d_Yt
            d_S = d_S + d_Ipt
    return cost, d_S


def _simulate_only_parallel(args):
    (I_S, duration, nodes_num, zero, one, stage_num, data_type, B_indices_list, hold_coef,
     penalty_coef, raw_material_node, B, time_stamp, D_mean, std, demand_set,
     delivery_shift, random_seed) = args

    minimum = np.minimum
    maximum = np.maximum
    np_isnan = np.isnan
    nonzero = np.nonzero
    zeros_like = np.zeros_like
    np_sum = np.sum
    np_multiply = np.multiply
    np_array = np.array

    D, D_order = _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                                  zero, random_seed)

    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)

    D_backlog = zeros_like(M_backlog)

    I_t = I_S + zero
    I_position = I_S + zero
    cost = zero


    for t in range(duration):
        I_position -= D_order[t, :]
        O_t = -minimum(zero, (I_position - I_S))
        for _ in range(stage_num - 1):
            temp_I_position = I_position - O_t * B
            O_t = -minimum(zero, (temp_I_position - I_S))
        I_position += O_t - O_t * B
        temp_I_t = I_t - D_backlog - D[t] + P[t]
        I_t = maximum(zero, temp_I_t)
        D_backlog = -minimum(zero, temp_I_t)

        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_backlog
        idx_purch = nonzero(purchase_order)[1]
        idx_mau = nonzero(mau_order)[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed
        temp_resource_rate[np_isnan(temp_resource_rate)] = one
        resource_rate = minimum(one, temp_resource_rate)


        P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]

        M_actual = zeros_like(M_backlog)
        min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]

        P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]

        M_backlog = mau_order - M_actual
        I_t -= M_actual * B

        cost = cost + np_sum(np_multiply(I_t, hold_coef)) + np_sum(
            np_multiply(D_backlog, penalty_coef))

    return cost


def _print_cost_grad_info(cost, gradient):
    print('total_cost: ', cost)
    delta_S = np.ones_like(gradient)
    print('gradient of item 666: ', gradient[0, 666])
    print('cost change: ', np.sum(delta_S * gradient))
