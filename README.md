## Mechanical-MNIST
Mechanical-MNIST is a benchmark dataset for mechanical meta-models

The Mechanical MNIST dataset contains the results of 70,000 (60,000 training examples + 10,000 test examples) finite element simulation of a heterogeneous material subject to large deformation. Mechanical MNIST is generated by first converting the MNIST bitmap images (http://www.pymvpa.org/datadb/mnist.html) to 2D heterogeneous blocks of a Neo-Hookean material. Consistent with the MNIST bitmap (28 x 28 pixels), the material domain is a 28 x 28 unit square. The bottom of the domain is fixed (Dirichlet boundary condition), the left and right edges of the domain are free, and the top of the domain is moved to a set of given fixed displacements. The results of the simulations include: (1) change in strain energy at each step, (2) total reaction force at the top boundary at each step, and (3) full field displacement at each step. All simulations are conducted with the FEniCS computing platform (https://fenicsproject.org).

## Full dataset

The full dataset is hosted by OpenBU with permanent link https://hdl.handle.net/2144/38693

The dataset can be downloaded with the following commands: 

<pre><code>wget https://open.bu.edu/bitstream/handle/2144/38693/MNIST_input_files.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_psi_results.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_rxnforce_results.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step1.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step2.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step3.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step4.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step5.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step6.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step7.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step8.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step9.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step10.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step11.zip
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step12.zip </code></pre>


## This repository contains the following:

**1) the code used to generate the dataset**

*input_data* -- folder containing 20 example MNIST bitmaps (10 test, 10 train) 

*FEA_train_FEniCS.py* -- code to generate FEA simulation training dataset

*FEA_test_FEniCS.py* -- code to generate FEA simulation test dataset

**2) a subset of the data (full dataset: https://hdl.handle.net/2144/38693)**

*mnist_img_train.txt.zip* -- the MNIST training bitmaps flattened and zipped (use python reshape((60000,28,28))) to get bitmaps

*mnist_img_test.txt.zip* -- the MNIST test bitmaps flattened and zipped (use python reshape((10000,28,28))) to get bitmaps

*summary_psi_train_all.txt* -- total change in strain energy at each step of applied displacement, training dataset, dimension 60K x 13 (call [:,12]) to get final step 

*summary_psi_test_all.txt* -- total change in strain energy at each step of applied displacement, test dataset, dimension 10K x 13 (call [:,12]) to get final step 

**3) the code used to create the metamodels in the paper "Mechanical MNIST: a benchmark dataset for mechanical metamodels" (will link to the paper in the near future)**

*set_up_input_data.py* -- import and save the data to be used to train the neural networks 

*nn_regress_psi_fnn.py* -- train a feedforward neural network to predict total change in strain energy from MNIST bitmap with PyTorch

*nn_regress_psi_cnn.py* -- train a convolutional neural network to predict total change in strain energy from MNIST bitmap with PyTorch

*pytorch_model_make_predictions.py* -- evaluate the PyTorch models on the test and training data 

*predict_disp_MLPR.py* -- train a MLPR model to predict final displacement from tiny initial displacement with scikit-learn

*predict_bitmap_MLPR.py* -- train a MLPR model to predict MNIST bitmap from final displacement with scikit-learn

Additional files for recording model results and plotting: *el_papers.mplstyle*, *plot_full_field.py*, *plot_nn_error.py*, 

