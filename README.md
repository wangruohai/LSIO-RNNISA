# LSIO-RNNISA 

This package contains the implementation of the RNN inspired simulation approach for large-scale inventory optimization problems discussed in the paper, "Tan Wang, L. Jeff Hong (2023) Large-Scale Inventory Optimization: A Recurrent Neural Networks–Inspired Simulation Approach. INFORMS Journal on Computing 35(1):196-215." (https://doi.org/10.1287/ijoc.2022.1253)


## Data
The `data` folder contains the experimental data for all the numerical experiments in the paper, which are stored in Python **pickle** files (.pkl), for example, `test_bom_100.pkl`. The BOMs are represented by directed networks created by **NetworkX** Python package, and lead time information and holding cost coefficients are represented by node attributes of the networks.
  * An introduction to the Python **pickle** module: https://docs.python.org/3/library/pickle.html
  * A tutorial on **NetworkX** Python package: https://networkx.org/documentation/latest/tutorial.html

## Source code
The `rnnisa` folder contains the source code of our recurrent-neural-networks-inspired simulation approach.

## Experiments  
The `experiment` folder contains the code for all the numerical experiments in the paper.
  * Simulation and Gradient Computation  
    * Performance of the Simulation Algorithms  
      * Run `test_sim_traditional.py`, `test_sim_sparse.py`, and `test_sim_dense.py` to test the traditional simulation algorithm, the tensorized simulation algorithms (i.e., Algorithm 1) with dense matrices or sparse matrices for inventory systems with different number of nodes. 

    * Performance of the Gradient Computation Algorithms  
      * Run `test_ipa_dense.py`, `test_ipa_sparse.py`, `test_bp_dense.py` and `test_bp_sparse.py` to test the tensorized BP algorithm and IPA algorithm with dense matrices or sparse matrices.   
      
    * Simulation and Gradient Computation Using TensorFlow  
      * Run `test_sim_tf_CPU_dense.py`, `test_sim_tf_CPU_sparse.py`, `test_sim_tf_GPU_dense.py` and `test_sim_tf_GPU_sparse.py` to test the run time of simulation using TensorFlow for a single replication. 
      * Run `test_grad_tf_CPU_dense.py`, `test_grad_tf_CPU_sparse.py`, `test_grad_tf_GPU_dense.py` and `test_grad_tf_GPU_sparse.py` to test the run time of gradient computation using TensorFlow for a single replication.
      
    * Performance of Multiple Replications  
      * Run `test_multiple_replications.py` to test the run times of the algorithms when multiple replications are run in parallel.
      
  * Performance of the Optimization Algorithms  
    * Run `test_opt_algorithms.py` to compare the performance of SSGD and FISTA algorithms on the inventory optimization problem with different scales and to test the advantage of a second stage in our optimization procedure.
    
  * Comparison with the Guaranteed Service Model  
    * Run `compare_with_GS_model.py` to conduct a comparison between our method and the Guaranteed Service (GS) model.
  
     
  

## Citing
If you use the code, please cite its corresponding paper:

```
@article{wang2022large,
  title={Large-Scale Inventory Optimization: A Recurrent Neural Networks–Inspired Simulation Approach},
  author={Tan Wang, L. Jeff Hong},
  journal={INFORMS Journal on Computing},
  volume = {35},
  number = {1},
  pages = {196-215},
  year = {2023}
}     
```  
