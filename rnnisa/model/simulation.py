"""
This module contains the simulation class for paper

Author:
    Tan Wang
"""

import os
print('OS')
# os.environ["MKL_NUM_THREADS"] = "1"
# # os.environ["NUMEXPR_NUM_THREADS"] = "1"
# # os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"  # this
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

        G = my_load(os.path.join(data_path, network_name))

        # self.G = G
        if type(G) == list:
            G = G[0]
        self.__B = nx.adjacency_matrix(G, weight='weight')
        # self.B_one=nx.adjacency_matrix(G, weight=1)
        self.__stage_num = count_layer(self.__B)
        # print("stage num:", self.__stage_num)
        self.__nodes_num = self.__B.shape[0]

        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type)
        self.__hold_coef = np.expand_dims(self.__hold_coef, axis=0)
        # self.holding_cost = np.array([[1,6,20,8,4,50,3,13,4,12]])
        self.__lead_time = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))
        # self.manu_lead = np.array([2, 4, 6, 4, 3, 2, 3, 3, 2, 2])
        # self.purchase_lead = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))

        self.__D_mean = np.zeros((self.__duration, self.__nodes_num))  # ,dtype=int
        self.__std = np.zeros_like(self.__D_mean)
        in_degree_values = np.array([v for k, v in G.in_degree()])
        demand_node = np.where(in_degree_values == 0)[0]
        i = 0
        for nd in list(G.nodes()):
            if i in demand_node:
                self.__D_mean[range(self.__duration), i] = G.nodes[nd]['mean']
                self.__std[range(self.__duration), i] = G.nodes[nd]['std']
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

        # item_list = list(G.nodes())
        # self.raw_material_node = np.zeros((1, self.nodes_num), dtype=data_type)
        out_degree_values = np.expand_dims(np.array([v for k, v in G.out_degree()]), axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)
        # i = 0
        # for nd in item_list:
        # if G.out_degree(nd) == 0:
        # self.raw_material_node[0, i] = self.one
        # i = i + 1
        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])
        # self.__raw_item_diag = diags(self.__raw_material_node[0])

        idx_mau = np.nonzero(1 - self.__raw_material_node)[1]
        self.__B_indices_list = {i: self.__B[i].indices for i in
                                 idx_mau}  # [self.B[i].indices for i in range(self.nodes_num)]

        time_stamp = np.zeros((self.__duration, self.__nodes_num), dtype=int)#time_stamp_m
        # time_stamp_p = np.zeros_like(time_stamp_m)
        time_stamp[:, :] = self.__lead_time
        # time_stamp_p[:, :] = self.lead_time
        time_stamp[:, :] = time_stamp[:, :] + np.expand_dims(np.array(list(range(self.__duration))), axis=1)
        # time_stamp_p[:, :] = time_stamp_p[:, :] + np.expand_dims(range_t, axis=1)
        self.__time_stamp = time_stamp
        # self.time_stamp_p = time_stamp_p
        self.__time_stamp_truncated = np.minimum(time_stamp, self.__duration)
        # self.time_stamp_p_limited = np.minimum(time_stamp_p, self.duration)

        # if delivery_cycle is None:
        #     delivery_cycle = order_time * np.ones(self.nodes_num, dtype=int)
        if type(delivery_cycle) == int:
            delivery_cycles = delivery_cycle * np.ones(self.__nodes_num, dtype=int)
        else:
            delivery_cycles = my_load(os.path.join(data_path, delivery_cycle))
            print('max delivery cycle:', np.max(delivery_cycles))

        self.__delivery_shift = np.zeros_like(time_stamp)
        for t in range(self.__duration):
            self.__delivery_shift[t, self.__demand_set] = t - delivery_cycles[self.__demand_set]
        self.__delivery_shift = np.maximum(-1, self.__delivery_shift)

    def _print_info(self):
        print('Data Type:', self.__data_type)
        print('nodes number:', self.__nodes_num)
        print('number of demand node:', len(self.__demand_set))
        print('penalty_factor:', np.min(self.__penalty_coef / self.__hold_coef))

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
            # for i in demand_set:
            #     D_order[t, i] = normalvariate(D_mean[t, i], std[t, i])
            #     if t >= delivery_cycle[i]:
            #         D[t, i] = D_order[int(t - delivery_cycle[i]), i]
        # D[self.order_time:self.duration, :] = D_order[0:(self.duration - self.order_time), :]
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
        # time_stamp_p = self.time_stamp_p_limited
        B = self.__B.toarray()
        nonzero = np.nonzero
        maximum = np.maximum
        minimum = np.minimum
        zeros_like = np.zeros_like

        D, D_order = self._generate_random_demand(rand_seed)
        # t_s = time()
        # initialize
        M_backlog = np.zeros(self.__nodes_num, dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
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
            # print('period: ', t + 1)
            # O_t = np_zeros(nodes_num, dtype=data_type)
            I_position[node_range] = I_position[node_range] - D_order[t, node_range]
            O_t[node_range] = -minimum(zero, (I_position[node_range] - I_S[0, node_range]))
            # for i in node_range:
            # I_position[i]=I_position[i]-D_order[t, i]
            # O_t[i]=-min(zero,(I_position[i] - I_S[0,i]))
            for _ in range(stage_num - 1):
                # temp_I_position=np_zeros(nodes_num, dtype=data_type)
                temp_I_position[node_range] = I_position[node_range] + zero
                for i in node_range:
                    if raw_material_node[0, i] < 1:
                        # for j in node_range:
                        # if B[i, j] > 0:
                        # temp_I_position[j]=temp_I_position[j]- O_t[i] * B[i,j]
                        temp_I_position[node_range] = temp_I_position[node_range] - O_t[i] * B[i, node_range]
                # O_t = np_zeros(nodes_num, dtype=data_type)
                # for i in range(nodes_num):
                # O_t[i]=-min(zero, (temp_I_position[i] - I_S[0,i]))
                O_t[node_range] = -minimum(zero, (temp_I_position[node_range] - I_S[0, node_range]))
            # W_qty = W_qty - W_t[t - 1]
            # purchase_order=np_zeros(nodes_num,data_type)
            # mau_order=np_zeros(nodes_num,data_type)
            I_position[node_range] = I_position[node_range] + O_t[node_range]
            temp_I[node_range] = I_t[node_range] - D_backlog[node_range] - D[t, node_range] + P[
                t, node_range] #+ W_t[t, node_range]
            I_t[node_range] = maximum(zero, temp_I[node_range])
            D_backlog[node_range] = -minimum(zero, temp_I[node_range])
            purchase_order[node_range] = O_t[node_range] * raw_material_node[0, node_range]
            mau_order[node_range] = O_t[node_range] - purchase_order[node_range] + M_backlog[node_range]
            idx_purch = nonzero(purchase_order)[0]
            idx_mau = nonzero(mau_order)[0]
            for i in idx_mau:
                # I_position[i] = I_position[i] + O_t[i]
                # if raw_material_node[0, i] < 1:
                # for j in node_range:
                # if B[i, j] > 0:
                # I_position[j] = I_position[j] - O_t[i] * B[i, j]
                I_position[node_range] = I_position[node_range] - O_t[i] * B[i, node_range]
                # temp = I_t[i] - D_backlog[i] - D[t - 1, i] + W_t[t - 1, i] + P[t - 1, i]
                # I_t[i] = max(zero, temp)
                # D_backlog[i] = -min(zero, temp)
                # purchase_order[i]=O_t[i]*raw_material_node[0,i]
                # mau_order[i] = O_t[i] - purchase_order[i] + M_buffer[i]
            resource_needed = zeros_like(M_backlog)
            for i in idx_mau:
                # for j in node_range:
                # if B[i, j] > 0:
                # resource_needed[j] = resource_needed[j] + mau_order[i] * B[i,j]
                resource_needed[node_range] = resource_needed[node_range] + mau_order[i] * B[i, node_range]
            """
            resource_rate=np_zeros(nodes_num,data_type)
            for i in range(nodes_num):
                if resource_needed[i]!=zero:
                    temp_resource_rate = I_t[i] / resource_needed[i]
                else:
                    temp_resource_rate=one
                resource_rate[i]=min(one,temp_resource_rate)
            """
            # P[minimum(duration, t + purchase_lead[idx_purch]), idx_purch] = purchase_order[idx_purch]
            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[idx_purch]
            # for index in idx_purch:
            # time_stamp = t - 1 + purchase_lead[index]
            # if time_stamp < duration:
            # P[time_stamp, index] = purchase_order[index]
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
                    # time_stamp = t - 1 + manu_lead[index]
                    # if time_stamp < duration:
                    # W_t[time_stamp, index] = M_actual[index]
                else:
                    M_backlog[index] = mau_order[index] + zero

            for i in idx_mau:
                if M_actual[i] > 0:
                    # for j in node_range:
                    # if B[i,j]>0:
                    # I_t[j] = I_t[j] - M_actual[i] * B[i,j]
                    I_t[node_range] = I_t[node_range] - M_actual[i] * B[i, node_range]
            # W_qty = W_qty + M_actual
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[idx_mau]
            # W_t[minimum(duration, t + manu_lead[idx_mau]), idx_mau] = M_actual[idx_mau]
            cost = cost + sum(
                I_t[node_range] * holding_cost[0, node_range] + D_backlog[node_range] * penalty_cost[0, node_range])
            # for i in node_range:
            # cost=cost+I_t[i]*holding_cost[0,i]+D_backlog[i]*penalty_cost[0,i]
        if print_flag:
            print('total_cost: ', cost)
        return cost

    """
    def simulate_traditional3(self, I_S, test_flag=True):
        #considering sparisity
        duration=self.duration
        zero=self.zero
        #one=self.one
        nodes_num=self.nodes_num
        stage_num=self.stage_num
        purchase_lead=self.purchase_lead
        manu_lead=self.manu_lead
        data_type=self.data_type
        matrix_shape2 = (duration, nodes_num)
        B_indices_list=self.B_indices_list
        holding_cost=self.holding_cost
        penalty_cost=self.penalty_cost
        raw_material_node=self.raw_material_node
        maximum=np.maximum
        nonzero=np.nonzero
        np_zeros=np.zeros
        np_sum=np.sum
        B=self.B.todense()

        D, D_order = self.generate_random_demand(test_flag)
        t_s = time.time()
        # initialize
        M_buffer = np_zeros(nodes_num, dtype=data_type)
        W_t = np_zeros(matrix_shape2, dtype=data_type)
        P = np_zeros(matrix_shape2, dtype=data_type)
        D_queue = np_zeros(nodes_num, dtype=data_type)
        #W_qty = np_zeros(vector_shape, dtype=data_type)
        I_t = np.squeeze(I_S) + zero
        I_position = np.squeeze(I_S) + zero
        cost = zero
        cost2 = zero
        filled_demand = np_zeros(nodes_num, dtype=data_type)

        for t in range(duration):
            print('period: ', t+1)
            O_t = np_zeros(nodes_num, dtype=data_type)
            for i in range(nodes_num):
                I_position[i]=I_position[i]-D_order[t, i]
                O_t[i]=-min(zero,(I_position[i] - I_S[0,i]))
            for _ in range(stage_num - 1):
                temp_I_position=np_zeros(nodes_num, dtype=data_type)
                for i in range(nodes_num):
                    temp_I_position[i] = I_position[i] + zero
                for i in range(nodes_num):
                    for j in B_indices_list[i]:
                        temp_I_position[j]=temp_I_position[j]-O_t[i] * B[i,j]
                O_t = np_zeros(nodes_num, dtype=data_type)
                for i in range(nodes_num):
                    O_t[i]=-min(zero, (temp_I_position[i] - I_S[0,i]))
            for i in range(nodes_num):
                I_position[i]=I_position[i]+O_t[i]
                for j in B_indices_list[i]:
                    I_position[j] = I_position[j] - O_t[i] * B[i,j]
            #print('sum:',np.sum(I_position-np.squeeze(I_S)))
            # print('test:',I_S)
            for i in range(nodes_num):
                temp=I_t[i] - D_queue[i] - D[t,i] + W_t[t,i] + P[t,i]
                I_t[i]=max(zero, temp)
                D_queue[i] = -min(zero,temp)

            # W_qty = W_qty - W_t[t - 1]
            purchase_order=np_zeros(nodes_num,data_type)
            mau_order=np_zeros(nodes_num,data_type)
            for i in range(nodes_num):
                purchase_order[i]=O_t[i]*raw_material_node[0,i]
                mau_order[i] = O_t[i] - purchase_order[i] + M_buffer[i]
            idx_purch = nonzero(purchase_order)[0]
            idx_mau = nonzero(mau_order)[0]
            resource_needed=np_zeros(nodes_num,data_type)
            for i in range(nodes_num):
                for j in B_indices_list[i]:
                    resource_needed[j] = resource_needed[j] + mau_order[i] * B[i,j]

            #resource_rate=np_zeros(nodes_num,data_type)
            #for i in range(nodes_num):
                #if resource_needed[i]!=zero:
                    #temp_resource_rate = I_t[i] / resource_needed[i]
                #else:
                    #temp_resource_rate=one
                #resource_rate[i]=min(one,temp_resource_rate)

            for index in idx_purch:
                time_stamp = t + purchase_lead[index]
                if time_stamp < duration:
                    P[time_stamp, index] = purchase_order[index]

            M_actual = np_zeros(nodes_num, dtype=data_type)
            for index in idx_mau:
                #col = B_indices_list[index]
                #min_rate = resource_rate[col].min()
                min_rate = 1
                for j in B_indices_list[index]:
                    min_rate = min(min_rate, I_t[j] / resource_needed[j])
                if min_rate > 0:
                    M_actual[index] = min_rate * mau_order[index]
                    time_stamp = t + manu_lead[index]
                    if time_stamp < duration:
                        W_t[time_stamp, index] = M_actual[index]
            M_buffer=np_zeros(nodes_num, dtype=data_type)
            for i in range(nodes_num):
                M_buffer[i] = mau_order[i] - M_actual[i]
                for j in B_indices_list[i]:
                    I_t[j] = I_t[j] - M_actual[i] * B[i,j]

            #I_t = I_t - M_actual * B
            # W_qty = W_qty + M_actual
            for i in range(nodes_num):
                cost=cost+I_t[i]*holding_cost[0,i]+D_queue[i]*penalty_cost[0,i]
                cost2 = cost2+I_t[i]*holding_cost[0,i]
                filled_demand[i]=filled_demand[i]+D[t,i]-D_queue[i]
            #cost = cost + np_sum(np_multiply((I_t + W_qty), holding_cost)) + np_sum(
                #np_multiply(D_queue, penalty_cost))
            #cost2 = cost2 + np_sum(np_multiply((I_t + W_qty), holding_cost))
            #filled_demand = filled_demand + D[t - 1] - D_queue
        D_sum = np_sum(D, axis=0)
        if test_flag:
            print('total_cost2: ', cost2)
            print_cost_info(t_s, cost)

        return cost, cost2, filled_demand, D_sum
    """

    def simulate(self, I_S, rand_seed=None, print_flag=False):
    # def simulate_only(self, I_S_seed, print_flag=False):
    #     I_S = I_S_seed[0]
    #     rand_seed = I_S_seed[1]
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
        # time_stamp_p = self.time_stamp_p_limited
        maximum = np.maximum
        minimum = np.minimum
        np_isnan = np.isnan
        nonzero = np.nonzero
        zeros_like = np.zeros_like
        np_sum = np.sum
        np_multiply = np.multiply
        np_array = np.array

        D, D_order = self._generate_random_demand(rand_seed)
        # t_s = time()
        # initialize
        M_backlog = np.zeros((1, self.__nodes_num), dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero
        # cost2 = zero
        # filled_demand = 0

        for t in range(duration):
            I_position -= D_order[t, :]  # I_position = I_position - D_order[t, :]
            O_t = -minimum(zero, (I_position - I_S))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - O_t * B
                O_t = -minimum(zero, (temp_I_position - I_S))
            I_position += O_t - O_t * B  # I_position = I_position - O_t * B + O_t
            temp_I_t = I_t - D_backlog - D[t] + P[t] #+ W_t[t]
            I_t = maximum(zero, temp_I_t)
            D_backlog = -minimum(zero, temp_I_t)
            # W_qty = W_qty - W_t[t - 1]
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = mau_order * B
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[np_isnan(temp_resource_rate)] = one
            resource_rate = minimum(one, temp_resource_rate)

            # P[minimum(duration, t - 1 + purchase_lead[idx_purch]), idx_purch] = purchase_order[0, idx_purch]
            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            # for index in idx_purch:
            # time_stamp = t - 1 + purchase_lead[index]
            # if time_stamp < duration:
            # P[time_stamp, index] = purchase_order[0, index]
            M_actual = zeros_like(M_backlog)  # M_actual[:]=zero
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            # W_t[minimum(duration, t - 1 + manu_lead[idx_mau]), idx_mau] = M_actual[0, idx_mau]
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
            """
            for index in idx_mau:
                col = B_indices_list[index]
                min_rate = resource_rate[0, col].min()
                if min_rate > 0:
                    M_actual[0, index] = min_rate * mau_order[0, index]
                    time_stamp = t - 1 + manu_lead[index]
                    if time_stamp < duration:
                        W_t[time_stamp, index] = M_actual[0, index]
            """
            M_backlog = mau_order - M_actual
            I_t -= M_actual * B  # I_t = I_t - M_actual * B
            # W_qty = W_qty + M_actual
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))  # np_multiply((I_t + W_qty), holding_cost)
            # cost2 = cost2 + np_sum(np_multiply(I_t, holding_cost))
            # filled_demand = filled_demand - D_queue + D[t]
        # D_sum = np_sum(D, axis=0)

        if print_flag:
            print('total_cost: ', cost)
            # print('total_holding_cost: ', cost2)
        return cost#, cost2, filled_demand, D_sum

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
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
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
        # t_s = time()
        # initialize
        M_backlog = np.zeros((1, self.__nodes_num), dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
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

            temp_I_t = I_t - D_backlog - D[t] + P[t] #+ W_t[t]
            I_t = maximum(zero, temp_I_t)
            D_backlog = -minimum(zero, temp_I_t)
            # W_qty = W_qty - W_t[t - 1]
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = np_dot(mau_order, B)
            temp_resource_rate = I_t / resource_needed
            temp_resource_rate[np_isnan(temp_resource_rate)] = one
            resource_rate = minimum(one, temp_resource_rate)

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            """
            for index in idx_purch:
                time_stamp = t - 1 + purchase_lead[index]
                if time_stamp < duration:
                    P[time_stamp, index] = purchase_order[0, index]
            """
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
            """
            for index in idx_mau:
                col = B_indices_list[index]
                min_rate = resource_rate[0, col].min()
                if min_rate > 0:
                    M_actual[0, index] = min_rate * mau_order[0, index]
                    time_stamp = t - 1 + manu_lead[index]
                    if time_stamp < duration:
                        W_t[time_stamp, index] = M_actual[0, index]
            """
            M_backlog = mau_order - M_actual
            I_t = I_t - np_dot(M_actual, B)
            # W_qty = W_qty + M_actual
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
        # purchase_lead = self.lead_time
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
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
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
        # t_s = time()
        # initialize
        M_backlog = np.zeros((1, nodes_num), dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_It_d_Yt = []
        d_Dback_d_Yt = []
        d_O_d_Ipformer = [[] for _ in range(duration)]
        # d_P_d_Mqty_item = [[] for _ in range(duration+1)]
        # d_P_d_O_item = [[] for _ in range(duration+1)]
        d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]  # []
        d_M_d_r_r = [{} for _ in range(duration)]
        d_r_r_d_I = []
        d_r_r_d_r_n = []
        # d_P_d_O_item = [[] for _ in range(duration)]
        # Mbuf_flag = []

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

            temp_I_t = I_t - D_backlog - D[t] + P[t] #+ W_t[t]
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_It_d_Yt.append(diags(flag[0]))
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t <= 0, one_minus, zero)
            d_Dback_d_Yt.append(diags(flag[0]))
            # W_qty = W_qty - W_t[t - 1]
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = mau_order * B
            temp_resource_rate = I_t / resource_needed
            # temp_resource_rate[np.isnan(temp_resource_rate)] = one
            temp_resource_rate[resource_needed == 0] = one
            temp1 = one / resource_needed
            temp1[resource_needed == 0] = one
            temp2 = -np_multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            d_r_r_d_I.append(
                diags(np_multiply(flag2, temp1)[
                          0]))  # diags(flag2[0])*sp.dia_matrix((temp1[0], [0]), shape=(nodes_num, nodes_num)))
            d_r_r_d_r_n.append(
                diags(np_multiply(flag2, temp2)[
                          0]))  # diags(flag2[0])*sp.dia_matrix((temp2[0], [0]), shape=(nodes_num, nodes_num)))

            # P[minimum(duration, t + purchase_lead[idx_purch]), idx_purch] = purchase_order[0, idx_purch]
            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            # for index in idx_purch:
            # time_stamp = t - 1 + purchase_lead[index]
            # if time_stamp < duration:
            # P[time_stamp, index] = purchase_order[0, index]
            # d_P_d_O_item[time_stamp].append(index)
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                               < equal_tolerance] for i in range(len(idx_mau))]
            d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i]) for i
                            in range(len(idx_mau)) if min_rate[i] > 0}
            d_M_d_man_o[t][0, idx_mau] = min_rate + zero
            # W_t[minimum(duration, t + manu_lead[idx_mau]), idx_mau] = M_actual[0, idx_mau]
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
            """
            M_actual = np_zeros(vector_shape, dtype=data_type)
            for index in idx_mau:
                col = B_indices_list[index]
                min_rate = resource_rate[0, col].min()
                if min_rate > 0:
                    M_actual[0, index] = min_rate * mau_order[0, index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate) < equal_tolerance]
                    k = data_type(1.0 / len(col2)) * mau_order[0, index]
                    d_M_d_r_r[t - 1][index] = (k, col2)
                    d_M_d_man_o[t - 1][0, index] = min_rate
                    time_stamp = t - 1 + manu_lead[index]
                    if time_stamp < duration:
                        W_t[time_stamp, index] = M_actual[0, index]
                        d_W_d_Mqty_item[time_stamp].append(index)
            """
            M_backlog = mau_order - M_actual
            # Mbuf_flag.append(where(M_backlog > 0, one, zero))
            I_t = I_t - M_actual * B
            # W_qty = W_qty + M_actual
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))  # np_sum(np_multiply((I_t + W_qty), holding_cost))
        d_S = zeros_like(M_backlog)
        # d_Wqty = self.holding_cost + self.zero
        d_It = holding_cost + zero
        d_Dback = penalty_cost + zero
        d_Ipt = zeros_like(M_backlog)
        d_Mt_backlog = zeros_like(M_backlog)
        # d_O = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
        d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)
        # d_W_d_Mq = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
        d_P_d_Mq = zeros_like(d_O)

        for t in range(duration - 1, -1, -1):
            # t = duration - tt
            d_Mact = - d_It * B_T  # + d_Wqty
            # temp = np_multiply(d_Mt_backlog, Mbuf_flag[t - 1])
            d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]  # d_Mact - temp + d_W_d_Mq[t - 1]
            d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])  # temp + np_multiply(d_Mq, d_M_d_man_o[t - 1])
            d_res_r = zeros_like(M_backlog)
            # d_M_d_r_r_key=np.array(list(d_M_d_r_r[t - 1].keys()))
            for index in d_M_d_r_r[t]:
                temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
                col2_list = d_M_d_r_r[t][index][1]
                d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k
                # for c_num in d_M_d_r_r[t - 1][index][1]:
                # d_res_r[0, c_num] = d_res_r[0, c_num] + temp_k
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
                # d_Wt = d_Yt + zero  # - d_Wqty
                # d_Wqty = d_Wqty + self.holding_cost
                # d_P = d_Yt + zero
                d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
                d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]
                """
                for index in d_W_d_Mqty_item[t - 1]:
                    #lead = manu_lead[index]
                    d_W_d_Mq[t - 1 - manu_lead[index]][0, index] = d_Yt[0, index]#d_Wt[0, index]
                """
                d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
                d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]
                """
                for index in d_P_d_O_item:#d_P_d_O_item[t - 1]:
                    #lead = purchase_lead[index]
                    d_O[t - 1 - purchase_lead[index]][0, index] = d_Yt[0, index]#d_P[0, index]#d_O[t - 1 - lead][0, index] + d_P[0, index]
                """
            else:
                d_S = d_S + d_Yt
                d_S = d_S + d_Ipt
        # gradient = d_S  # np_array(d_S)
        if print_flag:
            _print_cost_grad_info(cost, d_S) #gradient

        return cost, d_S#gradient

    def simulate_and_bp_dense(self, I_S, rand_seed=None, print_flag=False):
        duration = self.__duration
        nodes_num = self.__nodes_num
        zero = self.__zero
        one = self.__one
        one_minus = self.__one_minus
        stage_num = self.__stage_num
        # purchase_lead = self.lead_time
        lead_time = self.__lead_time
        data_type = self.__data_type
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        mau_item_diag = self.__mau_item_diag
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
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
        # t_s = time()
        # initialize
        M_backlog = np.zeros((1, nodes_num), dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_It_d_Yt = []
        d_Dback_d_Yt = []
        d_O_d_Ipformer = [[] for _ in range(duration)]
        # d_W_d_Mqty_item = [[] for _ in range(duration)]
        d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]  # []
        d_M_d_r_r = [{} for _ in range(duration)]
        d_r_r_d_I = []
        d_r_r_d_r_n = []
        # d_P_d_O_item = [[] for _ in range(duration)]
        # Mbuf_flag = []

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

            temp_I_t = I_t - D_backlog - D[t] + P[t]# + W_t[t]
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_It_d_Yt.append(diags(flag[0]))
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dback_d_Yt.append(diags(flag[0]))
            # W_qty = W_qty - W_t[t - 1]
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = nonzero(purchase_order)[1]
            idx_mau = nonzero(mau_order)[1]

            resource_needed = np_dot(mau_order, B)
            temp_resource_rate = I_t / resource_needed
            # temp_resource_rate[np.isnan(temp_resource_rate)] = one
            temp_resource_rate[resource_needed == 0] = one
            temp1 = one / resource_needed
            temp1[resource_needed == 0] = one
            temp2 = -np_multiply(temp_resource_rate, temp1)
            resource_rate = minimum(one, temp_resource_rate)
            flag2 = where(temp_resource_rate < 1, one, zero)
            d_r_r_d_I.append(
                diags(np_multiply(flag2, temp1)[
                          0]))  # diags(flag2[0])*sp.dia_matrix((temp1[0], [0]), shape=(nodes_num, nodes_num)))
            d_r_r_d_r_n.append(
                diags(np_multiply(flag2, temp2)[
                          0]))  # diags(flag2[0])*sp.dia_matrix((temp2[0], [0]), shape=(nodes_num, nodes_num)))

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
            # for index in idx_purch:
            # time_stamp = t - 1 + purchase_lead[index]
            # if time_stamp < duration:
            # P[time_stamp, index] = purchase_order[0, index]
            # d_P_d_O_item[time_stamp].append(index)
            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                               < equal_tolerance] for i in range(len(idx_mau))]
            d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i])
                            for i in range(len(idx_mau)) if min_rate[i] > 0}
            d_M_d_man_o[t][0, idx_mau] = min_rate + zero
            P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
            """
            M_actual = np_zeros(vector_shape, dtype=data_type)
            for index in idx_mau:
                col = B_indices_list[index]
                min_rate = resource_rate[0, col].min()
                if min_rate > 0:
                    M_actual[0, index] = min_rate * mau_order[0, index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate) < equal_tolerance]
                    k = data_type(1.0 / len(col2)) * mau_order[0, index]
                    d_M_d_r_r[t - 1][index] = (k, col2)
                    d_M_d_man_o[t - 1][0, index] = min_rate
                    time_stamp = t - 1 + manu_lead[index]
                    if time_stamp < duration:
                        W_t[time_stamp, index] = M_actual[0, index]
                        d_W_d_Mqty_item[time_stamp].append(index)
            """
            M_backlog = mau_order - M_actual
            # Mbuf_flag.append(where(M_backlog > 0, one, zero))
            I_t = I_t - np_dot(M_actual, B)
            # W_qty = W_qty + M_actual
            cost = cost + np_sum(np_multiply(I_t, holding_cost)) + np_sum(
                np_multiply(D_backlog, penalty_cost))
        d_S = zeros_like(M_backlog)
        # d_Wqty = self.holding_cost + self.zero
        d_It = holding_cost + zero
        d_Dback = penalty_cost + zero
        d_Ipt = zeros_like(M_backlog)
        d_Mt_backlog = zeros_like(M_backlog)
        # d_O = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
        d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)
        # d_W_d_Mq = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
        d_P_d_Mq = zeros_like(d_O)

        for t in range(duration - 1, -1, -1):
            # t = duration + 1 - tt
            d_Mact = - np_dot(d_It, B_T)  # + d_Wqty
            # temp = np_multiply(d_Mt_backlog, Mbuf_flag[t - 1])
            d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]  # d_Mact - temp + d_W_d_Mq[t - 1]
            d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])  # temp + np_multiply(d_Mq, d_M_d_man_o[t - 1])
            d_res_r = zeros_like(M_backlog)
            # d_M_d_r_r_key=np.array(list(d_M_d_r_r[t - 1].keys()))
            for index in d_M_d_r_r[t]:
                temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
                col2_list = d_M_d_r_r[t][index][1]
                d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k
                # for c_num in d_M_d_r_r[t - 1][index][1]:
                # d_res_r[0, c_num] = d_res_r[0, c_num] + temp_k
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
                # d_Wt = d_Yt + zero  # - d_Wqty
                # d_Wqty = d_Wqty + self.holding_cost
                # d_P = d_Yt + zero
                d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
                d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]
                """
                for index in d_W_d_Mqty_item[t - 1]:
                    #lead = manu_lead[index]
                    d_W_d_Mq[t - 1 - manu_lead[index]][0, index] = d_Yt[0, index]#d_Wt[0, index]
                """
                d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
                d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]
                """
                for index in d_P_d_O_item:#d_P_d_O_item[t - 1]:
                    #lead = purchase_lead[index]
                    d_O[t - 1 - purchase_lead[index]][0, index] = d_Yt[0, index]#d_P[0, index]#d_O[t - 1 - lead][0, index] + d_P[0, index]
                """
            else:
                d_S = d_S + d_Yt
                d_S = d_S + d_Ipt
        # gradient = d_S  # np_array(d_S)
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
        # time_stamp_p = self.time_stamp_p
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
        # t_s = time()
        # initialize
        M_backlog = np.zeros(vector_shape, dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = zeros(vector_shape, dtype=data_type)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_I = E + zero
        d_Dbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        # d_Wqty=csr_matrix(matrix_shape1, dtype=data_type)
        # d_W = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_P = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_Iposition = E + zero
        d_Mbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        d_cost = zeros_like(M_backlog)


        for t in range(duration):
            # print('period: ', t + 1)
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
            temp_I_t = I_t - D_backlog - D[t] + P[t]# + W_t[t]
            d_tempI = d_I - d_Dbacklog + d_P[t].tocsr() #+ d_W[t].tocsr()
            # d_W[t] = nan
            d_P[t] = nan
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_I = diags(flag[0]) * d_tempI
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dbacklog = diags(flag[0]) * d_tempI
            # W_qty = W_qty - W_t[t - 1]
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
                # time_stamp = t - 1 + purchase_lead[index]
                if time_stamp[t, index] < duration:
                    # P[time_stamp, index] = purchase_order[0, index]
                    d_P[time_stamp[t, index]] = set_row_lil(d_P[time_stamp[t, index]], index, d_O_getrow(index))
                    # d_P[time_stamp_p[t,index]].rows[index] = d_O_getrow(index).indices.tolist()
                    # d_P[time_stamp_p[t,index]].data[index] = d_O_getrow(index).data.tolist()

            M_actual = zeros_like(M_backlog)  # zeros(vector_shape, dtype=data_type)
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[minimum(duration, time_stamp[t, idx_mau]), idx_mau] = M_actual[0, idx_mau]
            # range1 = range(len(idx_mau))
            # col2 = [B_indices_list[idx_mau[i]][
            # np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i]) < equal_tolerance] for i in
            # range1]
            # d_min_rate = [csr_matrix(([data_type(1.0 / len(col2_i))] * len(col2_i), ([0] * len(col2_i), col2_i)),
            # shape=vector_shape)*d_res_rate for col2_i in col2]#, dtype=data_type
            # temp = [mau_order[0, idx_mau[i]] * d_min_rate[i] + min_rate[i] * d_mau_o[idx_mau[i]] for i in range1]
            d_Mact = lil_matrix(matrix_shape1, dtype=data_type)
            for i in range(len(idx_mau)):
                # for index in idx_mau:
                # col = B_indices_list[index]
                # min_rate = resource_rate[0, col].min()
                if min_rate[i] > 0:
                    index = idx_mau[i]
                    col = B_indices_list[index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                    k_len = len(col2)
                    d_min_rate = csr_matrix(([data_type(1.0 / k_len)] * k_len, ([0] * k_len, col2)), shape=vector_shape,
                                            dtype=data_type) * d_res_rate
                    # M_actual[0, index] = min_rate * mau_order[0, index]
                    temp = mau_order[0, index] * d_min_rate + min_rate[i] * d_mau_o[index]
                    d_Mact = set_row_lil(d_Mact, index, temp)
                    # d_Mact.rows[index] = temp.indices.tolist()
                    # d_Mact.data[index] = temp.data.tolist()
                    # time_stamp = t - 1 + manu_lead[index]
                    if time_stamp[t, index] < duration:
                        # W_t[time_stamp, index] = M_actual[0, index]
                        d_P[time_stamp[t, index]].rows[index] = d_Mact.rows[index]  # temp[i].indices.tolist()
                        d_P[time_stamp[t, index]].data[index] = d_Mact.data[index]  # temp[i].data.tolist()
            d_Mact = d_Mact.tocsr()
            M_backlog = mau_order - M_actual
            d_Mbacklog = d_mau_o - d_Mact
            I_t = I_t - M_actual * B
            d_I = d_I - B_T * d_Mact
            # W_qty = W_qty + M_actual
            cost = cost + np_sum(multiply(I_t, holding_cost)) + np_sum(
                multiply(D_backlog, penalty_cost))  # np_sum(multiply((I_t + W_qty), holding_cost))
            # d_cost = d_cost + holding_cost * (d_I + d_Wqty) + penalty_cost * d_Dbacklog
            d_cost = d_cost + holding_cost * d_I + penalty_cost * d_Dbacklog
        # gradient = d_cost  # np_array(d_cost)
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
        # E_B_T=self.E_B_T.toarray()
        E_B_T = (E - B).T
        raw_material_node = self.__raw_material_node
        mau_item_diag = self.__mau_item_diag
        B_indices_list = self.__B_indices_list
        equal_tolerance = self.__equal_tole
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp
        # time_stamp_p = self.time_stamp_p
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
        # t_s = time()
        # initialize
        M_backlog = np.zeros(vector_shape, dtype=data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = zeros(vector_shape, dtype=data_type)
        I_t = I_S + zero
        I_position = I_S + zero
        cost = zero

        d_I = E + zero
        d_Dbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        # d_Wqty=csr_matrix(matrix_shape1, dtype=data_type)
        # d_W = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_P = [lil_matrix(matrix_shape1, dtype=data_type) for _ in range(duration)]
        d_Iposition = E + zero
        d_Mbacklog = csr_matrix(matrix_shape1, dtype=data_type)
        d_cost = zeros_like(M_backlog)

        for t in range(duration):
            # print('period: ', t + 1)
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
            temp_I_t = I_t - D_backlog - D[t] + P[t] #+ W_t[t]
            d_tempI = d_I - d_Dbacklog + d_P[t].tocsr() #+ d_W[t].tocsr()
            # d_W[t] = nan
            d_P[t] = nan
            I_t = maximum(zero, temp_I_t)
            flag = where(temp_I_t > 0, one, zero)
            d_I = diags(flag[0]) * d_tempI
            D_backlog = -minimum(zero, temp_I_t)
            flag = where(temp_I_t < 0, one_minus, zero)
            d_Dbacklog = diags(flag[0]) * d_tempI
            # W_qty = W_qty - W_t[t - 1]
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
            # dia_1 = diags(flag2[0]) * dia_matrix((temp1[0], [0]), shape=matrix_shape1)
            # dia_2 = diags(flag2[0]) * dia_matrix((temp2[0], [0]), shape=matrix_shape1)
            d_res_rate = dia_1 * d_I + dia_2 * d_res_n

            P[minimum(duration, time_stamp[t, idx_purch]), idx_purch] = purchase_order[0, idx_purch]
            d_O = csr_matrix(d_O)
            d_O_getrow = d_O.getrow
            for index in idx_purch:
                # time_stamp = t - 1 + purchase_lead[index]
                if time_stamp[t, index] < duration:
                    # P[time_stamp, index] = purchase_order[0, index]
                    d_P[time_stamp[t, index]] = set_row_lil(d_P[time_stamp[t, index]], index, d_O_getrow(index))
                    # d_P[time_stamp_p[t,index]].rows[index] = d_O_getrow(index).indices.tolist()
                    # d_P[time_stamp_p[t,index]].data[index] = d_O_getrow(index).data.tolist()

            M_actual = zeros_like(M_backlog)
            min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
            M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
            P[minimum(duration, time_stamp[t, idx_mau]), idx_mau] = M_actual[0, idx_mau]
            # range1 = range(len(idx_mau))
            # col2 = [B_indices_list[idx_mau[i]][
            # np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i]) < equal_tolerance] for i in
            # range1]
            # d_min_rate = [csr_matrix(([data_type(1.0 / len(col2_i))] * len(col2_i), ([0] * len(col2_i), col2_i)),
            # shape=vector_shape)*d_res_rate for col2_i in col2]#, dtype=data_type
            # temp = [mau_order[0, idx_mau[i]] * d_min_rate[i] + min_rate[i] * d_mau_o[idx_mau[i]] for i in range1]
            d_Mact = lil_matrix(matrix_shape1, dtype=data_type)
            for i in range(len(idx_mau)):
                # for index in idx_mau:
                # col = B_indices_list[index]
                # min_rate = resource_rate[0, col].min()
                if min_rate[i] > 0:
                    index = idx_mau[i]
                    col = B_indices_list[index]
                    col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                    k_len = len(col2)
                    d_min_rate = csr_matrix(([data_type(1.0 / k_len)] * k_len, ([0] * k_len, col2)), shape=vector_shape,
                                            dtype=data_type) * d_res_rate
                    # M_actual[0, index] = min_rate * mau_order[0, index]
                    temp = csr_matrix(mau_order[0, index] * d_min_rate + min_rate[i] * d_mau_o[index])
                    d_Mact = set_row_lil(d_Mact, index, temp)
                    # time_stamp = t - 1 + manu_lead[index]
                    if time_stamp[t, index] < duration:
                        # W_t[time_stamp, index] = M_actual[0, index]
                        d_P[time_stamp[t, index]].rows[index] = d_Mact.rows[index]  # temp[i].indices.tolist()
                        d_P[time_stamp[t, index]].data[index] = d_Mact.data[index]  # temp[i].data.tolist()
            d_Mact = d_Mact.tocsr()
            M_backlog = mau_order - M_actual
            d_Mbacklog = d_mau_o - d_Mact
            I_t = I_t - np_dot(M_actual, B)
            d_I = d_I - B_T * d_Mact
            # W_qty = W_qty + M_actual
            cost = cost + np_sum(multiply(I_t, holding_cost)) + np_sum(
                multiply(D_backlog, penalty_cost))  # np_sum(multiply((I_t + W_qty), holding_cost))
            # d_cost = d_cost + holding_cost * (d_I + d_Wqty) + penalty_cost * d_Dbacklog
            d_cost = d_cost + holding_cost * d_I + penalty_cost * d_Dbacklog
        gradient = np_array(d_cost)
        if print_flag:
            _print_cost_grad_info(cost, gradient)
        return cost, gradient

    def simulate_and_IPA_traditional(self, I_S, rand_seed=None, print_flag=False):
        from itertools import product

        duration = self.__duration
        zero = self.__zero
        stage_num = self.__stage_num
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        equal_tolerance = self.__equal_tole
        raw_material_node = self.__raw_material_node
        time_stamp = self.__time_stamp_truncated
        # time_stamp_p = self.time_stamp_p_limited
        E = self.__E.toarray()
        B = self.__B.toarray()
        node_range = range(self.__nodes_num)
        mat_id = np.array(list(product(node_range, node_range)))
        id0 = mat_id[:, 0]
        id1 = mat_id[:, 1]

        maximum = np.maximum
        minimum = np.minimum
        nonzero = np.nonzero
        zeros_like = np.zeros_like

        D, D_order = self._generate_random_demand(rand_seed)
        # t_s = time()
        # initialize
        M_backlog = np.zeros(self.__nodes_num, dtype=self.__data_type)
        P = np.zeros((duration + 1, self.__nodes_num), dtype=self.__data_type)
        # W_t = zeros_like(P)
        D_backlog = zeros_like(M_backlog)
        # W_qty = np_zeros(vector_shape, dtype=data_type)
        I_t = np.squeeze(I_S) + zero
        I_position = np.squeeze(I_S) + zero
        cost = zero
        O_t = zeros_like(M_backlog)
        temp_I_position = zeros_like(M_backlog)
        purchase_order = zeros_like(M_backlog)
        mau_order = zeros_like(M_backlog)
        temp_I = zeros_like(M_backlog)

        d_Iposition = E + zero
        d_I = E + zero
        d_Dbacklog = zeros_like(B)
        # d_W = [zeros_like(B) for _ in range(duration + 1)]
        d_P = [zeros_like(B) for _ in range(duration + 1)]
        d_O = zeros_like(B)
        d_Mbacklog = zeros_like(B)
        d_cost = zeros_like(M_backlog)
        for t in range(duration):
            print('period: ', t + 1)
            I_position[node_range] = I_position[node_range] - D_order[t, node_range]
            O_t[node_range] = -minimum(zero, (I_position[node_range] - I_S[0, node_range]))
            for i in node_range:
                if I_position[i] - I_S[0, i] < 0:
                    d_O[i, node_range] = E[i, node_range] - d_Iposition[i, node_range]
                else:
                    d_O[i, node_range] = zero
            for _ in range(stage_num - 1):
                temp_I_position[node_range] = I_position[node_range] + zero
                d_temp_Ip = d_Iposition + zero
                for i in node_range:
                    if raw_material_node[0, i] < 1:
                        temp_I_position[node_range] = temp_I_position[node_range] - O_t[i] * B[i, node_range]
                        # d_temp_Ip[id0, id1] = d_temp_Ip[id0, id1] - d_O[i, id1] * B[i, id0]
                        d_temp_Ip[id0, id1] -= d_O[i, id1] * B[i, id0]
                        # for j in node_range:
                        #     # if B[i,j]>0:
                        #     d_temp_Ip[j, node_range] = d_temp_Ip[j, node_range] - d_O[i, node_range] * B[
                        #         i, j]
                O_t[node_range] = -minimum(zero, (temp_I_position[node_range] - I_S[0, node_range]))
                for i in node_range:
                    if temp_I_position[i] - I_S[0, i] < 0:
                        d_O[i, node_range] = E[i, node_range] - d_temp_Ip[i, node_range]
                    else:
                        d_O[i, node_range] = zero
            I_position[node_range] = I_position[node_range] + O_t[node_range]
            d_Iposition[id0, id1] = d_Iposition[id0, id1] + d_O[id0, id1]
            # for i in node_range:
            #     d_Iposition[i,node_range]=d_Iposition[i,node_range] + d_O[i, node_range]
            temp_I[node_range] = I_t[node_range] - D_backlog[node_range] - D[t, node_range]+ P[
                t, node_range]  #+ W_t[t, node_range]
            I_t[node_range] = maximum(zero, temp_I[node_range])
            D_backlog[node_range] = -minimum(zero, temp_I[node_range])
            for i in node_range:
                if temp_I[i] > 0:
                    d_I[i, node_range] = d_I[i, node_range] - d_Dbacklog[i, node_range] + d_P[t][
                        i, node_range] # + d_W[t][i, node_range]
                    d_Dbacklog[i, node_range] = zero
                else:
                    d_Dbacklog[i, node_range] = d_Dbacklog[i, node_range] - d_I[i, node_range] - \
                                              d_P[t][i, node_range]#  - d_W[t][i, node_range]
                    d_I[i, node_range] = zero
            purchase_order[node_range] = O_t[node_range] * raw_material_node[0, node_range]
            mau_order[node_range] = O_t[node_range] - purchase_order[node_range] + M_backlog[node_range]
            d_mau_o = zeros_like(B)
            for i in node_range:
                if raw_material_node[0, i] < 1:
                    d_mau_o[i, node_range] = d_Mbacklog[i, node_range] + d_O[i, node_range]
            idx_purch = nonzero(purchase_order)[0]
            idx_mau = nonzero(mau_order)[0]
            resource_needed = zeros_like(M_backlog)
            d_res_n = zeros_like(B)
            for i in idx_mau:
                I_position[node_range] = I_position[node_range] - O_t[i] * B[i, node_range]
                resource_needed[node_range] = resource_needed[node_range] + mau_order[i] * B[i, node_range]
                # d_Iposition[id0, id1] = d_Iposition[id0, id1] - d_O[i, id1] * B[i, id0]
                # d_res_n[id0, id1] = d_res_n[id0, id1] + d_mau_o[i, id1] * B[i, id0]
                d_Iposition[id0, id1] -= d_O[i, id1] * B[i, id0]
                d_res_n[id0, id1] += d_mau_o[i, id1] * B[i, id0]
                # for j in node_range:
                #     # if B[i, j]>0:
                #     d_Iposition[j, node_range] = d_Iposition[j, node_range] - d_O[i, node_range] * B[i, j]
                #     d_res_n[j, node_range] = d_res_n[j, node_range] + d_mau_o[i, node_range] * B[i, j]
            # for i in idx_mau:
            #     resource_needed[node_range] = resource_needed[node_range] + mau_order[i] * B[i, node_range]
            #     for j in node_range:
            #         if B[i, j] > 0:
            #             d_res_n[j, node_range] = d_res_n[j, node_range] - d_mau_o[i, node_range] * B[i, j]

            P[time_stamp[t, idx_purch], idx_purch] = purchase_order[idx_purch]
            for i in idx_purch:
                d_P[time_stamp[t, i]][i, node_range] = d_O[i, node_range] + zero
            M_actual = zeros_like(M_backlog)
            M_backlog = zeros_like(M_actual)
            d_Mact = zeros_like(B)
            d_Mbacklog = zeros_like(B)
            rate = I_t[node_range] / resource_needed[node_range]
            for index in idx_mau:
                min_rate = 1
                for j in node_range:
                    if B[index, j] > 0:
                        min_rate = min(min_rate, rate[j])
                d_min_rate = zeros_like(M_backlog)
                if min_rate < 1:
                    min_index = [j for j in node_range if
                                 (B[index, j] > 0) and (abs(rate[j] - min_rate) < equal_tolerance)]
                    # for j in node_range:
                    #     if B[index, j] > 0:
                    #         if abs(rate[j] - min_rate) < equal_tolerance:
                    #             min_index.append(j)
                    for index_m in min_index:
                        d_min_rate[node_range] = d_min_rate[node_range] + d_I[index_m, node_range] / resource_needed[
                            index_m] - I_t[index_m] * \
                                                 d_res_n[index_m, node_range] / (
                                                             resource_needed[index_m] * resource_needed[index_m])
                    d_min_rate[node_range] = d_min_rate[node_range] / len(min_index)
                if min_rate > 0:
                    M_actual[index] = min_rate * mau_order[index]
                    d_Mact[index, node_range] = min_rate * d_mau_o[index, node_range] + d_min_rate * mau_order[index]
                    M_backlog[index] = (1 - min_rate) * mau_order[index]
                    d_Mbacklog[index, node_range] = d_mau_o[index, node_range] - d_Mact[index, node_range]
                else:
                    M_backlog[index] = mau_order[index] + zero
                    d_Mbacklog[index, node_range] = d_mau_o[index, node_range] + zero

            for i in idx_mau:
                if M_actual[i] > 0:
                    I_t[node_range] = I_t[node_range] - M_actual[i] * B[i, node_range]
                    # d_I[id0, id1] = d_I[id0, id1] - d_Mact[i, id1] * B[i, id0]
                    d_I[id0, id1] -= d_Mact[i, id1] * B[i, id0]
                    # for j in node_range:
                    #     d_I[j, node_range] = d_I[j, node_range] - d_Mact[i, node_range] * B[i, j]
                    #     # if B[i,j]>0:
                    #     #     d_I[j, node_range] = d_I[j, node_range] - d_Mact[i, node_range] * B[i, j]

            P[time_stamp[t, idx_mau], idx_mau] = M_actual[idx_mau]
            for i in idx_mau:
                d_P[time_stamp[t, i]][i, node_range] = d_Mact[i, node_range] + zero

            cost = cost + sum(
                I_t[node_range] * holding_cost[0, node_range] + D_backlog[node_range] * penalty_cost[0, node_range])
            for i in node_range:
                d_cost[node_range] = d_cost[node_range] + d_I[i, node_range] * holding_cost[0, i] + \
                                     d_Dbacklog[i, node_range] * penalty_cost[0, i]
        # gradient = d_cost
        if print_flag:
            print('total_cost: ', cost)
            delta_S = np.ones_like(d_cost)
            print('gradient of 666: ', d_cost[66])  # 66
            print('cost change: ', np.sum(delta_S * d_cost))
            # print('time used: %6.2f minutes' % ((time() - t_s) / 60))
        return cost, d_cost

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
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
        vector_shape = [1, self.__nodes_num]
        tf_maximum = tf.maximum
        tf_minimum = tf.minimum
        reduce_sum = tf.reduce_sum
        # to_dense=tf.sparse.to_dense
        where = tf.where
        # reorder=tf.sparse.reorder
        reduce_min = tf.reduce_min
        gather_nd = tf.gather_nd
        # concat=tf.concat
        update = tf.tensor_scatter_nd_update
        scatter_nd = tf.scatter_nd

        D, D_order = self._generate_random_demand(rand_seed)
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        P_values = np.zeros((duration + 1, duration, self.__nodes_num), dtype=np.int8)
        # W_values = np.zeros_like(P_values)
        cost = zero

        if GPU_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1,0'
            physical_devices = tf.config.list_physical_devices('GPU')
            for device_gpu in physical_devices:
                tf.config.experimental.set_memory_growth(device_gpu, True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.experimental.set_synchronous_execution(enable=False)
        # print('synchronous_false')

        tf_B, tf_B_indices_list = self._get_tf_B()
        if dense_flag:
            tf_matmul = tf.matmul
            tf_B = tf.sparse.to_dense(tf_B)
        else:
            tf_matmul = tf.sparse.sparse_dense_matmul
        # t_s = time()
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # initialize
        tf_I_S = tf.convert_to_tensor(I_S, self.__data_type)
        M_backlog = tf.zeros(vector_shape, dtype=self.__data_type)
        # P_indices = [[[0, 0]] for _ in range(duration)]
        # P_values = [[zero] for _ in range(duration)]
        D_backlog = tf.zeros_like(M_backlog)
        # W_qty = tf.zeros([1, self.nodes_num], dtype=self.data_type_tensor)
        # M_actual_all = tf.zeros([duration, self.nodes_num], dtype=self.data_type)
        # purchase_order_all = tf.zeros_like(M_actual_all)
        P_history = tf.zeros([duration, self.__nodes_num], dtype=self.__data_type)
        # temp_tensor=[tf_zeros([t, nodes_num], dtype=d_type) for t in range(duration)]
        # temp_one_minus_vector = -tf.ones([1, self.nodes_num], dtype=self.data_type_tensor)
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        I_t = tf_I_S + zero
        I_position = tf_I_S + zero
        for t in range(duration):
            I_position = I_position - D_order[t, :]
            O_t = - tf_minimum(zero, (I_position - tf_I_S))
            for _ in range(stage_num - 1):
                temp_I_position = I_position - tf_matmul(O_t, tf_B)
                O_t = - tf_minimum(zero, (temp_I_position - tf_I_S))
            I_position = I_position + O_t - tf_matmul(O_t, tf_B)
            temp_I_t = I_t - D_backlog - D[t] + reduce_sum(P_values[t] * P_history, axis=0)# + \
                       # reduce_sum(P_values[t] * purchase_order_all, axis=0)
            I_t = tf_maximum(zero, temp_I_t)
            D_backlog = -tf_minimum(zero, temp_I_t)
            # W_qty = W_qty - W_t[t - 1]
            purchase_order = O_t * raw_material_node
            mau_order = O_t - purchase_order + M_backlog
            idx_purch = where(purchase_order > 0)[:, 1].numpy()
            idx_mau = where(mau_order > 0).numpy()  # [:, 1]
            # purchase_order_all = purchase_order_all + concat([temp_tensor[t], purchase_order,
            # temp_tensor[duration_1 - t]], axis=0)
            # purchase_order_all = update(purchase_order_all, [[t]], purchase_order)
            resource_needed = tf_matmul(mau_order, tf_B)
            resource_needed = tf_maximum(equal_tolerance, resource_needed)
            resource_rate = I_t / resource_needed
            resource_rate = tf_minimum(one, resource_rate)
            # min_rate = SparseTensor(indices=idx_mau,
            #                         values=[reduce_min(gather_nd(resource_rate, tf_B_indices_list[index]))
            #                                 for index in idx_mau[:, 1]], dense_shape=vector_shape)
            # min_rate = to_dense(min_rate)
            min_rate = scatter_nd(idx_mau, [reduce_min(gather_nd(resource_rate,
                                                                 tf_B_indices_list[index])) for index in idx_mau[:, 1]],
                                  vector_shape)
            M_act = min_rate * mau_order
            # min_rate = {index: reduce_min(gather_nd(resource_rate, tf_B_indices_list[index])) for index in
            # idx_mau}
            # M_val = {index: min_rate[index] * mau_order[0, index] for index in idx_mau2}
            # M_act = SparseTensor(indices=[[0, index] for index in idx_mau2], values=list(M_val.values()),
            # dense_shape=vector_shape)
            # M_act = to_dense(M_act)
            # M_actual_all = update(M_actual_all, [[t]], M_act)
            P_history = update(P_history, [[t]], purchase_order+M_act)
            # M_actual_all = M_actual_all + concat([temp_tensor[t], M_act,
            # temp_tensor[duration_1 - t]], axis=0)

            idx_mau2 = where(M_act > 0)[:, 1].numpy()  # min_rate
            P_values[time_stamp[t, idx_purch], t, idx_purch] = one_small
            P_values[time_stamp[t, idx_mau2], t, idx_mau2] = one_small
            M_backlog = mau_order - M_act
            # M_backlog = mau_order - tf.where(M_actual > 0, tf.ones_like(M_actual, dtype=self.data_type),
            # tf.zeros_like(M_actual, dtype=self.data_type)) * M_actual
            I_t = I_t - tf_matmul(M_act, tf_B)
            # W_qty = W_qty + M_actual
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
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
        vector_shape = [1, self.__nodes_num]
        tf_maximum = tf.maximum
        tf_minimum = tf.minimum
        reduce_sum = tf.reduce_sum
        # to_dense=tf.sparse.to_dense
        where = tf.where
        # reorder=tf.sparse.reorder
        reduce_min = tf.reduce_min
        gather_nd = tf.gather_nd
        # concat=tf.concat
        update = tf.tensor_scatter_nd_update
        scatter_nd = tf.scatter_nd

        D, D_order = self._generate_random_demand(rand_seed)
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        P_values = np.zeros((duration + 1, duration, self.__nodes_num), dtype=np.int8)
        # W_values = np.zeros_like(P_values)
        cost = zero

        if GPU_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1,0'
            physical_devices = tf.config.list_physical_devices('GPU')
            for device_gpu in physical_devices:
                tf.config.experimental.set_memory_growth(device_gpu, True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.experimental.set_synchronous_execution(enable=False)
        # print('synchronous_false')

        tf_B, tf_B_indices_list = self._get_tf_B()
        if dense_flag:
            tf_matmul = tf.matmul
            tf_B = tf.sparse.to_dense(tf_B)
        else:
            tf_matmul = tf.sparse.sparse_dense_matmul
        # del temp_B, indices, tf_B_sparse_split
        # gc.collect()
        # t_s = time()
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # initialize
        tf_I_S = tf.convert_to_tensor(I_S, self.__data_type)
        M_backlog = tf.zeros(vector_shape, dtype=self.__data_type)
        # P_indices = [[[0, 0]] for _ in range(duration)]
        # P_values = [[zero] for _ in range(duration)]
        D_backlog = tf.zeros_like(M_backlog)  # tf.zeros(vector_shape, dtype=d_type)
        # W_qty = tf.zeros([1, self.nodes_num], dtype=self.data_type_tensor)
        P_history = tf.zeros([duration, self.__nodes_num], dtype=self.__data_type)
        # M_actual_all = tf.zeros_like(purchase_order_all)  # tf.zeros([duration,nodes_num],dtype=d_type)
        # temp_tensor=[tf_zeros([t, nodes_num], dtype=d_type) for t in range(duration)]
        # temp_one_minus_vector = -tf.ones([1, self.nodes_num], dtype=self.data_type_tensor)
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
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
                temp_I_t = I_t - D_backlog - D[t] + reduce_sum(P_values[t] * P_history, axis=0) #+ reduce_sum(W_values[t] * M_actual_all, axis=0)
                I_t = tf_maximum(zero, temp_I_t)
                D_backlog = -tf_minimum(zero, temp_I_t)
                # W_qty = W_qty - W_t[t - 1]
                purchase_order = O_t * raw_material_node
                mau_order = O_t - purchase_order + M_backlog
                with tape.stop_recording():
                    idx_purch = where(purchase_order > 0)[:, 1].numpy()
                    idx_mau = where(mau_order > 0).numpy()  # [:, 1]
                # purchase_order_all = purchase_order_all + concat([temp_tensor[t], purchase_order,
                # temp_tensor[duration_1 - t]], axis=0)
                # purchase_order_all = update(purchase_order_all, [[t]], purchase_order)
                resource_needed = tf_matmul(mau_order, tf_B)
                resource_needed = tf_maximum(equal_tolerance, resource_needed)
                resource_rate = I_t / resource_needed
                resource_rate = tf_minimum(one, resource_rate)
                # min_rate = SparseTensor(indices=idx_mau,
                #                         values=[reduce_min(gather_nd(resource_rate, tf_B_indices_list[index]))
                #                                 for index in idx_mau[:, 1]], dense_shape=vector_shape)
                # min_rate = to_dense(min_rate)
                min_rate = scatter_nd(idx_mau, [reduce_min(gather_nd(resource_rate,
                                                                     tf_B_indices_list[index])) for index in
                                                idx_mau[:, 1]],
                                      vector_shape)
                M_act = min_rate * mau_order
                # min_rate = {index: reduce_min(gather_nd(resource_rate, tf_B_indices_list[index])) for index in
                # idx_mau}
                # M_val = {index: min_rate[index] * mau_order[0, index] for index in idx_mau2}
                # M_act = SparseTensor(indices=[[0, index] for index in idx_mau2], values=list(M_val.values()),
                # dense_shape=vector_shape)
                # M_act = to_dense(M_act)
                # M_actual_all = update(M_actual_all, [[t]], M_act)
                P_history = update(P_history, [[t]], purchase_order+M_act)
                # M_actual_all = M_actual_all + concat([temp_tensor[t], M_act,
                # temp_tensor[duration_1 - t]], axis=0)
                with tape.stop_recording():
                    idx_mau2 = where(M_act > 0)[:, 1].numpy()  # min_rate
                    P_values[time_stamp[t, idx_purch], t, idx_purch] = one_small
                    P_values[time_stamp[t, idx_mau2], t, idx_mau2] = one_small
                M_backlog = mau_order - M_act
                # M_backlog = mau_order - tf.where(M_actual > 0, tf.ones_like(M_actual, dtype=self.data_type),
                # tf.zeros_like(M_actual, dtype=self.data_type)) * M_actual
                I_t = I_t - tf_matmul(M_act, tf_B)
                # W_qty = W_qty + M_actual
                cost = cost + reduce_sum(I_t * holding_cost) + reduce_sum(
                    D_backlog * penalty_cost)

        # print('calculate grad: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        gradient = tape.gradient(cost, tf_I_S)
        gradient = gradient.numpy()
        cost = cost.numpy()
        if print_flag:
            _print_cost_grad_info(cost, gradient)
        return cost, gradient

    def simulate_tf_batch(self, I_S, batch, grad_flag, GPU_flag, dense_flag=False, rand_seed=None, print_flag=False):
        import tensorflow as tf

        d_type = self.__data_type
        duration = self.__duration
        stage_num = self.__stage_num
        nodes_num = self.__nodes_num
        one = self.__one
        one_small = np.int8(1)
        zero = self.__zero
        equal_tolerance = self.__equal_tole
        raw_material_node = self.__raw_material_node
        holding_cost = self.__hold_coef
        penalty_cost = self.__penalty_coef
        time_stamp = self.__time_stamp_truncated  # minimum(self.time_stamp_m, duration)
        # time_stamp_p = self.time_stamp_p_limited  # minimum(self.time_stamp_p, duration)
        SparseTensor = tf.SparseTensor
        tf_maximum = tf.maximum
        tf_minimum = tf.minimum
        reduce_sum = tf.reduce_sum
        to_dense = tf.sparse.to_dense
        where = tf.where
        reduce_min = tf.reduce_min
        gather = tf.gather
        concat = tf.concat
        tf_zeros = tf.zeros
        tf_expand_dims = tf.expand_dims
        vector_shape = [batch, nodes_num]

        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if rand_seed:
            seed(1)
            print('constant seed')
        D = np.zeros((duration, batch, nodes_num), dtype=d_type)
        D_order = np.zeros((duration, batch, nodes_num), dtype=d_type)
        D_mean = self.__D_mean
        std = self.__std
        demand_set = self.__demand_set
        delivery_shift = self.__delivery_shift

        # for ba in range(batch):
        #     for t in range(duration):
        #         for i in demand_set:
        #             D_order[t, ba, i] = normalvariate(D_mean[t, i], std[t, i])
        #             if t >= delivery_cycle[i]:
        #                 D[t, ba, i] = D_order[int(t - delivery_cycle[i]), ba, i]
        for t in range(duration):
            for i in demand_set:
                D_order[t, range(batch), i] = normalvariate(D_mean[t, i], std[t, i])
                D[t, range(batch), i] = D_order[delivery_shift[t, i], range(batch), i]
        D_order = np.maximum(zero, D_order)
        D = np.maximum(zero, D)
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        W_values = np.zeros((duration + 1, batch, duration, nodes_num), dtype=np.int8)
        P_values = np.zeros((duration + 1, batch, duration, nodes_num), dtype=np.int8)
        cost = zero

        if GPU_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1,0'
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

        # t_s = time()
        # print('start! ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # initialize
        tf_I_S = tf.convert_to_tensor(I_S, d_type)
        # tf_I_S = tf.expand_dims(tf_I_S, axis=0)
        # tf_I_S = tf.tile(tf_I_S, [batch, 1])#[batch, 1, 1]
        M_buffer = tf_zeros(vector_shape, dtype=d_type)
        # P_indices = [[[0, 0]] for _ in range(duration)]
        # P_values = [[zero] for _ in range(duration)]
        D_queue = tf_zeros(vector_shape, dtype=d_type)
        # W_qty = tf.zeros([1, self.nodes_num], dtype=self.data_type_tensor)
        M_actual_all = tf_zeros([batch, duration, nodes_num], dtype=d_type)
        purchase_order_all = tf_zeros([batch, duration, nodes_num], dtype=d_type)
        temp_tensor = [tf_zeros([batch, t, nodes_num], dtype=d_type) for t in range(duration)]
        # temp_one_minus_vector = -tf.ones([1, self.nodes_num], dtype=self.data_type_tensor)
        # strategy = tf.distribute.MirroredStrategy()
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # with strategy.scope():

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tf_I_S)
            # I_t = tf_I_S + zero
            I_t = tf.tile(tf_I_S, [batch, 1]) + zero
            I_position = I_t + zero  # tf_I_S + zero#
            for t in range(duration):
                I_position = I_position - D_order[t]
                O_t = - tf_minimum(zero, (I_position - tf_I_S))
                for _ in range(stage_num - 1):
                    temp_I_position = I_position - tf_matmul(O_t, tf_B)
                    O_t = - tf_minimum(zero, (temp_I_position - tf_I_S))
                I_position = I_position + O_t - tf_matmul(O_t, tf_B)
                temp_I_t = I_t - D_queue - D[t] + reduce_sum(W_values[t] * M_actual_all, axis=1) + \
                           reduce_sum(P_values[t] * purchase_order_all, axis=1)
                I_t = tf_maximum(zero, temp_I_t)
                D_queue = -tf_minimum(zero, temp_I_t)
                # W_qty = W_qty - W_t[t - 1]
                purchase_order = O_t * raw_material_node
                mau_order = O_t - purchase_order + M_buffer

                with tape.stop_recording():
                    idx_purch = where(purchase_order > 0).numpy()
                    # idx_b_p = where(purchase_order > 0)[:, 0].numpy()
                    idx_mau = where(mau_order > 0).numpy()  # [:, 1]
                purchase_order_all = purchase_order_all + concat(
                    [temp_tensor[t], tf_expand_dims(purchase_order, axis=1),
                     temp_tensor[duration - t - 1]], axis=1)
                resource_needed = tf_matmul(mau_order, tf_B)
                resource_needed = tf_maximum(equal_tolerance, resource_needed)
                resource_rate = I_t / resource_needed
                resource_rate = tf_minimum(one, resource_rate)
                min_rate = SparseTensor(indices=idx_mau,
                                        values=[reduce_min(
                                            gather(resource_rate[idx_mau[i, 0]], tf_B_indices_list[idx_mau[i, 1]]))
                                            for i in range(idx_mau.shape[0])], dense_shape=vector_shape)
                min_rate = to_dense(min_rate)
                M_act = min_rate * mau_order
                # min_rate = {index: reduce_min(gather_nd(resource_rate, tf_B_indices_list[index])) for index in
                # idx_mau}
                # M_val = {index: min_rate[index] * mau_order[0, index] for index in idx_mau2}
                # M_act = SparseTensor(indices=[[0, index] for index in idx_mau2], values=list(M_val.values()),
                # dense_shape=vector_shape)
                # M_act = to_dense(M_act)
                M_actual_all = M_actual_all + concat([temp_tensor[t], tf_expand_dims(M_act, axis=1),
                                                      temp_tensor[duration - t - 1]], axis=1)
                with tape.stop_recording():
                    # idx_b_mau = where(min_rate > 0)[:, 0].numpy()
                    idx_mau2 = where(min_rate > 0).numpy()
                    P_values[time_stamp[t, idx_purch[:, 1]], idx_purch[:, 0], t, idx_purch[:, 1]] = one_small
                    W_values[time_stamp[t, idx_mau2[:, 1]], idx_mau2[:, 0], t, idx_mau2[:, 1]] = one_small
                M_buffer = mau_order - M_act
                # M_buffer = mau_order - tf.where(M_actual > 0, tf.ones_like(M_actual, dtype=self.data_type),
                # tf.zeros_like(M_actual, dtype=self.data_type)) * M_actual
                I_t = I_t - tf_matmul(M_act, tf_B)
                # W_qty = W_qty + M_actual
                cost = cost + reduce_sum(I_t * holding_cost, axis=1) + reduce_sum(
                    D_queue * penalty_cost, axis=1)
        if grad_flag:
            # print('calculate grad: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            gradient = tape.gradient(cost, tf_I_S)
            gradient = gradient.numpy()
            print(gradient.shape)
            cost = cost.numpy()
            if print_flag:
                _print_cost_grad_info(cost, gradient)
            return cost, gradient
        else:
            cost = cost.numpy()
            if print_flag:
                print('total_cost: ', cost)
            return cost

    def evaluate_cost(self, I_S, eval_num=30):#, print_flag=False
        process_num = min(CORE_NUM, eval_num)
        if self.__nodes_num == 500000: process_num = min(process_num, 30)
        if self.__nodes_num == 100000: process_num = min(process_num, 50)
        # I_S_list = [(I_S, i+self.__seed_num) for i in range(eval_num)]
        I_S_list = [(I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__stage_num
                     , self.__data_type, self.__B_indices_list, self.__hold_coef, self.__penalty_coef,
                     self.__raw_material_node, self.__B, self.__time_stamp_truncated, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_only_parallel, I_S_list)
            # result = pool.map(self.simulate_only, I_S_list)
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
                     , self.__hold_coef, self.__penalty_coef, self.__mau_item_diag, self.__raw_material_node, self.__B
                     , self.__B_T, self.__E_B_T, self.__time_stamp_truncated, self.__D_mean, self.__std,
                     self.__demand_set, self.__delivery_shift, i + self.__seed_num) for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_and_bp_parallel, I_S_list)
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



def _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
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


def _simulate_and_bp_parallel(args):
    (I_S, duration, nodes_num, zero, one, one_minus, stage_num, lead_time, data_type
     , B_indices_list, equal_tole, hold_coef, penalty_coef, mau_item_diag, raw_material_node
     , B, B_T, E_B_T, time_stamp, D_mean, std, demand_set, delivery_shift, rand_seed) = args
    minimum = np.minimum
    maximum = np.maximum
    where = np.where
    # np_isnan = np.isnan
    nonzero = np.nonzero
    np_abs = np.abs
    zeros_like = np.zeros_like
    np_sum = np.sum
    np_multiply = np.multiply
    np_array = np.array

    D, D_order = _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                                  zero, rand_seed)
    # t_s = time()
    # initialize
    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)
    # W_t = zeros_like(P)
    D_backlog = zeros_like(M_backlog)
    # W_qty = np_zeros(vector_shape, dtype=data_type)
    I_t = I_S + zero
    I_position = I_S + zero
    cost = zero

    d_It_d_Yt = []
    d_Dback_d_Yt = []
    d_O_d_Ipformer = [[] for _ in range(duration)]
    # d_W_d_Mqty_item = [[] for _ in range(duration)]
    d_M_d_man_o = [zeros_like(M_backlog) for _ in range(duration)]  # []
    d_M_d_r_r = [{} for _ in range(duration)]
    d_r_r_d_I = []
    d_r_r_d_r_n = []
    # d_P_d_O_item = [[] for _ in range(duration)]
    # Mbuf_flag = []

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

        temp_I_t = I_t - D_backlog - D[t] + P[t] #+ W_t[t]
        I_t = maximum(zero, temp_I_t)
        flag = where(temp_I_t > 0, one, zero)
        d_It_d_Yt.append(diags(flag[0]))
        D_backlog = -minimum(zero, temp_I_t)
        flag = where(temp_I_t <= 0, one_minus, zero)
        d_Dback_d_Yt.append(diags(flag[0]))
        # W_qty = W_qty - W_t[t - 1]
        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_backlog
        idx_purch = nonzero(purchase_order)[1]
        idx_mau = nonzero(mau_order)[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed
        # temp_resource_rate[np.isnan(temp_resource_rate)] = one
        temp_resource_rate[resource_needed == 0] = one

        temp1 = one / resource_needed
        temp1[resource_needed == 0] = one
        temp2 = -np_multiply(temp_resource_rate, temp1)
        resource_rate = minimum(one, temp_resource_rate)
        flag2 = where(temp_resource_rate < 1, one, zero)
        d_r_r_d_I.append(
            diags(np_multiply(flag2, temp1)[
                      0]))  # diags(flag2[0])*sp.dia_matrix((temp1[0], [0]), shape=(nodes_num, nodes_num)))
        d_r_r_d_r_n.append(
            diags(np_multiply(flag2, temp2)[
                      0]))  # diags(flag2[0])*sp.dia_matrix((temp2[0], [0]), shape=(nodes_num, nodes_num)))

        # P[minimum(duration, t + purchase_lead[idx_purch]), idx_purch] = purchase_order[0, idx_purch]
        P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
        # for index in idx_purch:
        # time_stamp = t - 1 + purchase_lead[index]
        # if time_stamp < duration:
        # P[time_stamp, index] = purchase_order[0, index]
        # d_P_d_O_item[time_stamp].append(index)
        M_actual = zeros_like(M_backlog)
        min_rate = np_array([resource_rate[0, B_indices_list[index]].min() for index in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
        col2 = [B_indices_list[idx_mau[i]][np_abs(resource_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i])
                                           < equal_tole] for i in range(len(idx_mau))]
        d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_order[0, idx_mau[i]], col2[i]) for i
                        in range(len(idx_mau)) if min_rate[i] > 0}
        d_M_d_man_o[t][0, idx_mau] = min_rate + zero
        # W_t[minimum(duration, t + manu_lead[idx_mau]), idx_mau] = M_actual[0, idx_mau]
        P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
        """
        M_actual = np_zeros(vector_shape, dtype=data_type)
        for index in idx_mau:
            col = B_indices_list[index]
            min_rate = resource_rate[0, col].min()
            if min_rate > 0:
                M_actual[0, index] = min_rate * mau_order[0, index]
                col2 = col[np_abs(resource_rate[0, col] - min_rate) < equal_tole]
                k = data_type(1.0 / len(col2)) * mau_order[0, index]
                d_M_d_r_r[t - 1][index] = (k, col2)
                d_M_d_man_o[t - 1][0, index] = min_rate
                time_stamp = t - 1 + manu_lead[index]
                if time_stamp < duration:
                    W_t[time_stamp, index] = M_actual[0, index]
                    d_W_d_Mqty_item[time_stamp].append(index)
        """
        M_backlog = mau_order - M_actual
        # Mbuf_flag.append(where(M_backlog > 0, one, zero))
        I_t = I_t - M_actual * B
        # W_qty = W_qty + M_actual
        cost = cost + np_sum(np_multiply(I_t, hold_coef)) + np_sum(
            np_multiply(D_backlog, penalty_coef))  # np_sum(np_multiply((I_t + W_qty), hold_coef))
    d_S = zeros_like(M_backlog)
    # d_Wqty = self.hold_coef + self.zero
    d_It = hold_coef + zero
    d_Dback = penalty_coef + zero
    d_Ipt = zeros_like(M_backlog)
    d_Mt_backlog = zeros_like(M_backlog)
    # d_O = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
    d_O = np.zeros((duration, 1, nodes_num), dtype=data_type)
    # d_W_d_Mq = [np_zeros(vector_shape, dtype=data_type) for _ in range(duration)]#[]
    d_P_d_Mq = zeros_like(d_O)

    for t in range(duration - 1, -1, -1):
        # t = duration - tt
        d_Mact = - d_It * B_T  # + d_Wqty
        # temp = np_multiply(d_Mt_backlog, Mbuf_flag[t - 1])
        d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]  # d_Mact - temp + d_W_d_Mq[t - 1]
        d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t])  # temp + np_multiply(d_Mq, d_M_d_man_o[t - 1])
        d_res_r = zeros_like(M_backlog)
        # d_M_d_r_r_key=np.array(list(d_M_d_r_r[t - 1].keys()))
        for index in d_M_d_r_r[t]:
            temp_k = d_M_d_r_r[t][index][0] * d_Mq[0, index]
            col2_list = d_M_d_r_r[t][index][1]
            d_res_r[0, col2_list] = d_res_r[0, col2_list] + temp_k
            # for c_num in d_M_d_r_r[t - 1][index][1]:
            # d_res_r[0, c_num] = d_res_r[0, c_num] + temp_k
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
            # d_Wt = d_Yt + zero  # - d_Wqty
            # d_Wqty = d_Wqty + self.hold_coef
            # d_P = d_Yt + zero
            d_P_d_Mqty_item = nonzero(P[t]*mau_item_diag)[0]
            d_P_d_Mq[t - lead_time[d_P_d_Mqty_item], 0, d_P_d_Mqty_item] = d_Yt[0, d_P_d_Mqty_item]
            """
            for index in d_W_d_Mqty_item[t - 1]:
                #lead = manu_lead[index]
                d_W_d_Mq[t - 1 - manu_lead[index]][0, index] = d_Yt[0, index]#d_Wt[0, index]
            """
            d_P_d_O_item = nonzero(P[t]*raw_material_node)[0]
            d_O[t - lead_time[d_P_d_O_item], 0, d_P_d_O_item] = d_Yt[0, d_P_d_O_item]
            """
            for index in d_P_d_O_item:#d_P_d_O_item[t - 1]:
                #lead = purchase_lead[index]
                d_O[t - 1 - purchase_lead[index]][0, index] = d_Yt[0, index]#d_P[0, index]#d_O[t - 1 - lead][0, index] + d_P[0, index]
            """
        else:
            d_S = d_S + d_Yt
            d_S = d_S + d_Ipt
    # gradient = d_S  # np_array(d_S)
    # print_cost_grad_info(cost, d_S)
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
    # t_s = time()
    # initialize
    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)
    # W_t = zeros_like(P)
    D_backlog = zeros_like(M_backlog)
    # W_qty = np_zeros(vector_shape, dtype=data_type)
    I_t = I_S + zero
    I_position = I_S + zero
    cost = zero
    # cost2 = zero
    # filled_demand = 0

    for t in range(duration):
        I_position -= D_order[t, :]  # I_position = I_position - D_order[t, :]
        O_t = -minimum(zero, (I_position - I_S))
        for _ in range(stage_num - 1):
            temp_I_position = I_position - O_t * B
            O_t = -minimum(zero, (temp_I_position - I_S))
        I_position += O_t - O_t * B  # I_position = I_position - O_t * B + O_t
        temp_I_t = I_t - D_backlog - D[t] + P[t]# + W_t[t]
        I_t = maximum(zero, temp_I_t)
        D_backlog = -minimum(zero, temp_I_t)
        # W_qty = W_qty - W_t[t - 1]
        purchase_order = O_t * raw_material_node
        mau_order = O_t - purchase_order + M_backlog
        idx_purch = nonzero(purchase_order)[1]
        idx_mau = nonzero(mau_order)[1]

        resource_needed = mau_order * B
        temp_resource_rate = I_t / resource_needed
        temp_resource_rate[np_isnan(temp_resource_rate)] = one
        resource_rate = minimum(one, temp_resource_rate)

        # P[minimum(duration, t - 1 + purchase_lead[idx_purch]), idx_purch] = purchase_order[0, idx_purch]
        P[time_stamp[t, idx_purch], idx_purch] = purchase_order[0, idx_purch]
        # for index in idx_purch:
        # time_stamp = t - 1 + purchase_lead[index]
        # if time_stamp < duration:
        # P[time_stamp, index] = purchase_order[0, index]
        M_actual = zeros_like(M_backlog)
        min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_order[0, idx_mau]
        # W_t[minimum(duration, t - 1 + manu_lead[idx_mau]), idx_mau] = M_actual[0, idx_mau]
        P[time_stamp[t, idx_mau], idx_mau] = M_actual[0, idx_mau]
        """
        for index in idx_mau:
            col = B_indices_list[index]
            min_rate = resource_rate[0, col].min()
            if min_rate > 0:
                M_actual[0, index] = min_rate * mau_order[0, index]
                time_stamp = t - 1 + manu_lead[index]
                if time_stamp < duration:
                    W_t[time_stamp, index] = M_actual[0, index]
        """
        M_backlog = mau_order - M_actual
        I_t -= M_actual * B  # I_t = I_t - M_actual * B
        # W_qty = W_qty + M_actual
        cost = cost + np_sum(np_multiply(I_t, hold_coef)) + np_sum(
            np_multiply(D_backlog, penalty_coef))  # np_multiply((I_t + W_qty), hold_coef)
        # cost2 = cost2 + np_sum(np_multiply(I_t, hold_coef))
        # filled_demand = filled_demand - D_backlog + D[t]
    # D_sum = np_sum(D, axis=0)
    # print('total_cost: ', cost)
    return cost#, cost2, filled_demand, D_sum


def _print_cost_grad_info(cost, gradient):
    print('total_cost: ', cost)
    delta_S = np.ones_like(gradient)
    print('gradient of item 666: ', gradient[0, 666])
    print('cost change: ', np.sum(delta_S * gradient))


















