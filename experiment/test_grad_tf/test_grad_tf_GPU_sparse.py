"""
This module contains the experiments for the paper

Author:
    Tan Wang
"""

import numpy as np
from rnnisa.model import simulation
from rnnisa.utils.tool_function import print_run_time
from time import time



def evaluate_performance_of_gradient_computation_tf_GPU(I_S_0, sim, rep_num, print_flag=True):
    # sim.simulate_and_bp_tf(I_S=I_S_0, GPU_flag=True, rand_seed=0)#warm up
    t_s = time()
    for i in range(rep_num):
        sim.simulate_and_bp_tf(I_S=I_S_0, GPU_flag=True, rand_seed=i)
    if print_flag:
        print_run_time('Gradient Computation Using TensorFlow-GPU-sparse', t_s, 'second', rep_num)






if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num_list = [1000, 5000, 10000, 50000]#[1000]#
    warm_flag = True
    for n in nodes_num_list:
        sim = simulation.Simulation(data_type=data_type, duration=100, data_path=data_path,
                                    network_name='test_bom_' + str(n) + '.pkl',
                                    penalty_factor=2.0)
        I_S_0 = data_type(10) * np.ones((1, n), dtype=data_type)
        if warm_flag:
            evaluate_performance_of_gradient_computation_tf_GPU(I_S_0, sim, 1, False)#warm up
            warm_flag = False
        evaluate_performance_of_gradient_computation_tf_GPU(I_S_0, sim, 10)
