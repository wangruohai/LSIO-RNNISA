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




if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num = 1000  # set number of nodes here
    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=data_path,
                                network_name='test_bom_' + str(nodes_num) + '.pkl',
                                penalty_factor=2.0)
    I_S_0 = data_type(10) * np.ones((1, nodes_num), dtype=data_type)
    t_s = time()
    sim.simulate_and_bp_tf(I_S=I_S_0, GPU_flag=False, dense_flag=True, rand_seed=0) #you can use other random seeds
    print_run_time('Gradient Computation Using TensorFlow-CPU-dense', t_s, 'second', 1)

