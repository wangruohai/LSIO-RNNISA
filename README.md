# LSIO-RNNISA
This package contains the implementation of the RNN inspired simulation approach for large-scale inventory optimization problems discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach"


## About the data
Experimental data are stored in Python pickle files (.pkl), for example, "test_bom_100.pkl". The BOMs are represented by directed networks created by NetworkX Python package, and lead time information and holding cost coefficients are represented by node attributes of the networks.





If you use the code, please cite its corresponding paper:

```
@article{wan2022large,
  title={Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach},
  author={Wan, Tan and Hong, L Jeff},
  journal={arXiv preprint arXiv:2201.05868},
  year={2022}
}     
```  
