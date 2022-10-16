"""
This module contains the experiments for the RNN inspired simulation approach for large-scale inventory optimization problems
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

import numpy as np
from rnnisa.model import simulation
from rnnisa.utils.tool_function import print_run_time
from time import time



def evaluate_performance_of_ipa(I_S_0, sim, rep_num, print_flag=True):
    t_s = time()
    for i in range(rep_num):
        sim.simulate_and_IPA_dense(I_S=I_S_0, rand_seed=i)
    if print_flag:
        print_run_time('IPA-dense', t_s, 'minute', rep_num)






if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num_list = [1000, 5000, 10000]
    warm_flag = True
    for n in nodes_num_list:
        sim = simulation.Simulation(data_type=data_type, duration=100, data_path=data_path,
                                    network_name='test_bom_' + str(n) + '.pkl',
                                    penalty_factor=2.0)
        I_S_0 = data_type(10) * np.ones((1, n), dtype=data_type)
        if warm_flag:
            evaluate_performance_of_ipa(I_S_0, sim, 1, False)
            warm_flag = False

        evaluate_performance_of_ipa(I_S_0, sim, 1)
