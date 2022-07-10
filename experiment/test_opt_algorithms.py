"""
This module contains the experiments for the RNN inspired simulation approach for large-scale inventory optimization problems
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

from rnnisa.model import simulation, simu_opt
import numpy as np



def compare_opt_algorithms(sim, opt, I_S0):
    _, max_epoch = opt.FISTA(I_S_0=I_S0)  # init_step_size=0.0000001*0.2
    sim.reset_seed()
    _ = opt.SSGD(I_S_0=I_S0, max_epoch=max_epoch)  # stopping_threshold=5e10
    sim.reset_seed()


def compare_optimization_performance(data_type, temp_path, nodes_num):
    step_size_dict={10000:5e-7, 50000:5e-7, 100000:5e-7, 500000:5e-7}
    regula_para_dict={10000:7e6, 50000:7e6, 100000:7e6, 500000:7e6}
    stop_thresh_dict={10000:2.0e-5, 50000:2.0e-5, 100000:1.0e-5, 500000:1.0e-5}
    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name='test_bom_'+str(nodes_num)+'.pkl',
                                penalty_factor=2.0)  # penalty_factor=10.0, order_time=7,
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num], stop_thresh=stop_thresh_dict[nodes_num],
                          positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, decay_mode=1)
    I_S_0 = data_type(10) * np.ones((1, nodes_num), dtype=data_type)
    I_S_0[0, sim.get_demand_set()] = 320
    compare_opt_algorithms(sim=sim, opt=opt, I_S0=I_S_0)  # step_size=10.0*1e-7


def evaluate_stage2(data_type, temp_path, nodes_num):
    step_size_dict={10000:5e-7, 50000:5e-7, 100000:5e-7, 500000:5e-7}
    regula_para_dict={10000:7e6, 50000:7e6, 100000:7e6, 500000:7e6}
    stop_thresh_dict={10000:2.0e-5, 50000:2.0e-5, 100000:1.0e-5, 500000:1.0e-5}
    stop_thresh_ratio_dict={10000:1.0, 50000:1.0, 100000:1.0, 500000:1.0}
    step_size_ratio_dict={10000:0.2, 50000:0.2, 100000:0.2, 500000:0.2}
    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name='test_bom_'+str(nodes_num)+'.pkl',
                                penalty_factor=2.0) # penalty_factor=10.0, order_time=7,
    I_S_0 = data_type(10) * np.ones((1, nodes_num), dtype=data_type)
    I_S_0[0, sim.get_demand_set()] = 320
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num],
                          stop_thresh=stop_thresh_dict[nodes_num], positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient,
                          stop_thresh_ratio=stop_thresh_ratio_dict[nodes_num],
                          step_size_ratio=step_size_ratio_dict[nodes_num], decay_mode=1)
    I_S_1, I_S_2 = opt.two_stage_procedure(I_S_0)
    optimal_cost1 = sim.evaluate_cost(I_S_1, 100)
    sim.cut_seed(100)
    optimal_cost2 = sim.evaluate_cost(I_S_2, 100)
    print('optimal cost in stage1: %.3e' % optimal_cost1)
    print('optimal cost in stage2: %.3e' % optimal_cost2)
    print('cost improvement in stage2: %.4e' % ((optimal_cost1 - optimal_cost2) / optimal_cost1))
    sim.reset_seed()




if __name__ == "__main__":
    temp_path = "./temp"
    data_type = np.float32
    nodes_num_list = [10000, 50000, 100000, 500000]
    for n in nodes_num_list:
        compare_optimization_performance(data_type, temp_path, n)
        evaluate_stage2(data_type, temp_path, n)







