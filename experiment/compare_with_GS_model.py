"""
This module contains the experiments for the RNN inspired simulation approach for large-scale inventory optimization problems
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

import os
from rnnisa.model import simulation, simu_opt
import numpy as np
from rnnisa.utils.tool_function import my_load




def compare_with_GSM(data_type, nodes_num, sim, opt, I_S0_K, base_stock_GSM_path):
    I_S_GSM = my_load(base_stock_GSM_path)
    print('max: ', np.max(I_S_GSM))
  
    I_S0 = data_type(I_S0_K) * np.ones((1, nodes_num), dtype=data_type)
    I_S0[0, sim.get_demand_set()] = 40
    
    _, I_S = opt.two_stage_procedure(I_S0)
    if I_S.shape[0] < 20:
        print('I_S: ', I_S)
        print('I_S_GSM: ', I_S_GSM)
    optimal_cost = sim.evaluate_cost(I_S=I_S, eval_num=100)  # , print_flag=True
    print('optimal cost of RNN-based method: %.3e' % optimal_cost)
    sim.cut_seed(100)
    optimal_cost = sim.evaluate_cost(I_S=I_S_GSM, eval_num=100)  # , print_flag=True
    print('optimal cost of GSM: %.3e' % optimal_cost)
    sim.reset_seed()


def compare_with_GSM_spanning_tree(data_type, temp_path, nodes_num):
    network_name_dict={10:"bom_kodak.pkl", #a simple example of Kodak digital camera supply chain
                       10000:"bom_spanning_tree_from_real_case_10000-2.pkl", #Larger Spanning Trees with 10000 nodes
                       50000:"bom_spanning_tree_from_real_case_50000.pkl"} #Larger Spanning Trees with 50000 nodes
    delivery_cycle_pkl_dict={10:'delivery_cycle-10nodes-2021-12-17 04-33.pkl',
                             10000:'delivery_cycle-10000nodes-2021-12-21 09-46-2.pkl'
                             ,50000:"delivery_cycle-50000nodes-2021-12-22 03-35.pkl"}
    step_size_dict={10:3.8e-2, 10000:2.53e-6, 50000:1e-5}
    regula_para_dict={10:1.2e4, 10000:1.46e6, 50000:7e5}
    stop_thresh_dict={10:2e-4, 10000:1e-4, 50000:1.11e-4}
    stop_thresh_ratio_dict={10:0.7, 10000:0.48, 50000:0.4}
    step_size_ratio_dict={10:0.08, 10000:0.026, 50000:0.0015}
    step_bound_dict={10:None, 10000:[[4,0.04,-4,-0.04], [1.5, 0.015, -1.5, -0.015]],
                     50000:[[3, 0.04, -3, -0.04], [2.5, 0.025, -2.5, -0.025]]}
    I_S0_K_dict={10:10, 10000:0.01, 50000:0.01}
    base_stock_path_dict={10:"base_stock_GSM-10nodes-2021-12-17 04-33.pkl",
                          10000:"base_stock_GSM-10000nodes-2021-12-21 09-46-2.pkl",
                          50000:"base_stock_GSM-50000nodes-2021-12-22 03-35.pkl"}

    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name=network_name_dict[nodes_num],  # test_bom_spanning_tree50000.pkl
                                delivery_cycle=delivery_cycle_pkl_dict[nodes_num],
                                penalty_factor=2.0)  # "temp_delivery_cycle-10000-20210423.pkl"
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num],
                          stop_thresh=stop_thresh_dict[nodes_num], positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=step_bound_dict[nodes_num],
                          stop_thresh_ratio=stop_thresh_ratio_dict[nodes_num],
                          step_size_ratio=step_size_ratio_dict[nodes_num], decay_mode=2)
    compare_with_GSM(data_type=data_type, nodes_num=nodes_num, sim=sim, opt=opt, I_S0_K=I_S0_K_dict[nodes_num],
                     base_stock_GSM_path=os.path.join(temp_path, base_stock_path_dict[nodes_num]))


def compare_with_GSM_network_with_cycles(data_type, temp_path):
    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name="bom_kodak_with_cycle.pkl",
                                delivery_cycle='delivery_cycle-10nodes-2021-12-17 04-33.pkl',
                                #'delivery_cycle-10nodes-2022-01-09 05-01-2.pkl',
                                penalty_factor=2.0)#"temp_delivery_cycle-10000-20210423.pkl"
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=1e-2, regula_para=1.2e4,
                          stop_thresh= 4e-4, positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=[[3, 0.03, -3, -0.03], [3, 0.03, -3, -0.03]]
                          , stop_thresh_ratio=0.7, step_size_ratio=0.1, decay_mode=2)
    path = os.path.join(temp_path, "base_stock_GSM-10nodes-2021-12-17 04-33.pkl")
    compare_with_GSM(data_type=data_type, nodes_num=10, sim=sim, opt=opt, I_S0_K=10, base_stock_GSM_path=path)


def compare_with_GSM_rand_lead(data_type, data_path):
    from rnnisa.model import simulation_rand_lead
    sim = simulation_rand_lead.Simulation(data_type=np.float32, duration=100, data_path=data_path,
                                          network_name="bom_kodak.pkl",
                                          delivery_cycle='delivery_cycle-10nodes-2021-12-17 04-33.pkl',
                                          # 'delivery_cycle-10nodes-2022-01-09 05-01-2.pkl',
                                          penalty_factor=2.0)  # "temp_delivery_cycle-10000-20210423.pkl"
    opt = simu_opt.SimOpt(data_path=data_path, rep_num=10, step_size=1e-2, regula_para=5e3,
                          stop_thresh= 1e-3, positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=[[3, 0.03, -3, -0.03], [3, 0.03, -3, -0.03]]
                          , stop_thresh_ratio=0.7, step_size_ratio=0.08, decay_mode=2)
    path = os.path.join(data_path, "base_stock_GSM-10nodes-2021-12-17 04-33.pkl")
    compare_with_GSM(data_type=data_type, nodes_num=10, sim=sim, opt=opt, I_S0_K=10, base_stock_GSM_path=path)





if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num_list = [10, 10000, 50000]
    for n in nodes_num_list:
        compare_with_GSM_spanning_tree(data_type, data_path, n)

    compare_with_GSM_network_with_cycles(data_type, data_path)

    compare_with_GSM_rand_lead(data_type, data_path)






