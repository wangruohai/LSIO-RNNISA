"""
This module contains the experiments for the RNN inspired simulation approach for large-scale inventory optimization problems
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

from rnnisa.model import simulation
import numpy as np
from rnnisa.utils.tool_function import print_run_time
from time import time




def evaluate_performance_of_multiple_replications(I_S_0, sim):
    t_s = time()
    sim.evaluate_cost(I_S = I_S_0, eval_num=10)
    print_run_time('Simulation Mini-batch', t_s, 'second')
    sim.reset_seed()

    t_s = time()
    sim.evaluate_cost_gradient(I_S = I_S_0, eval_num=10)
    print_run_time('BP Mini-batch', t_s, 'second')
    sim.reset_seed()







if __name__ == "__main__":
    temp_path = "./temp"
    data_type = np.float32
    nodes_num_list = [1000, 5000, 10000, 50000, 100000, 500000]#[1000]#
    for n in nodes_num_list:
        sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                    network_name='test_bom_' + str(n) + '.pkl',
                                    penalty_factor=2.0)
        I_S_0 = data_type(10) * np.ones((1, n), dtype=data_type)
        evaluate_performance_of_multiple_replications(I_S_0, sim)










