import numpy as np
import os
##########################################################################################
#	SUMMARY
##########################################################################################
#####################################
#	FEA problem statement  
#####################################
# We run a finite element simulation where the bottom of the domain is fixed (Dirichlet 
#	boundary condition), the left and right edges of the domain are free, and the top of 
#	the domain is moved to a set of given fixed displacements. In keeping with the size of 
#	the MNIST bitmap ($28 \times 28$ pixels), the domain is a $28 \times 28$ unit square. 
#	We prescribe displacement at the top of the domain up to $50 \%$ of the initial domain 
#	size. The applied displacements $d$ are:
# 			d = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0 ] 
#	and data is generated at each displacement step. We run all finite element simulations
#	using the FEniCS computing platform. Mesh refinement studies were conducted, and we 
#	determined that a mesh with $39,200$ quadratic triangular elements is sufficient to 
#	capture the converged solution while not needlessly using computational resources. 
#	This mesh corresponds to $50$ quadratic triangular elements per bitmap pixel. 
#
# To convert the MNIST bitmap images to material properties, we divide the material domain 
#	such that it corresponds with the grayscale bitmap and then specify $E$ as 
#			E = \frac{b}{ 255.0} \, (100.0 - 1.0) + 1.0
#	where $b$ is the corresponding value of the grayscale bitmap that can range from 
#	$0-255$. Poisson's ratio is kept fixed at $\nu = 0.3$ throughout the domain. This 
#	strategy means that the Mechanical MNIST material domains contain a soft background 
#	material with ``digits'' that are two orders of magnitude stiffer. 

##########################################################################################
#	contents of https://open.bu.edu/handle/2144/38693
##########################################################################################
# there are 60,000 training examples and 10,000 test examples  
# results are reported for each displacement $d$, there are 12 steps (numbered 1-12)
#	NOTE: step 0, which corresponds to zero displacement, is omitted from the dataset 

############################################
#	contents of MNIST_input_files folder 
############################################
# mnist_img_train.txt			--> input bitmaps, MNIST dataset
#									60K x 784 (use reshape((60000,28,28))) to get images
# mnist_img_test.txt			--> input bitmaps, MNIST dataset
#									10K x 784 (use reshape((10000,28,28))) to get images

######### --> DOWNLOAD at:
# https://open.bu.edu/bitstream/handle/2144/38693/MNIST_input_files.zip

############################################
#	contents of FEA_psi_results folder 
############################################
# summary_psi_train_all.txt	 	--> total strain energy at each displacement step
#									60K x 13 (call [:,12]) to get final step 
# summary_psi_test_all.txt		--> total strain energy at each displacement step
#									10K x 13 (call [:,12]) to get final step 

######### --> DOWNLOAD at:
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_psi_results.zip

############################################
#	contents of FEA_rxnforce_results folder 
############################################
# summary_rxnx_train_all.txt 	--> total x reaction force (upper edge)
#									60K x 13 (call [:,12]) to get final step 
# summary_rxnx_test_all.txt		--> total x reaction force (upper edge)
#									10K x 13 (call [:,12]) to get final step 
# summary_rxny_train_all.txt	--> total y reaction force (upper edge)
#									60K x 13 (call [:,12]) to get final step 
# summary_rxny_test_all.txt		--> total y reaction force (upper edge)
#									10K x 13 (call [:,12]) to get final step 

######### --> DOWNLOAD at:
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_rxnforce_results.zip

############################################
#	contents of FEA_displacement_results_step%i folder  (i = 1-12)
############################################
# summary_dispx_train_step%i.txt	--> x-disp at the center of each `pixel' at step i
#									60K x 784 (use reshape((60000,28,28))) to match images
# summary_dispx_test_step%i.txt 	--> x-disp at the center of each `pixel' at step i
#									10K x 784 (use reshape((60000,28,28))) to match images
# summary_dispy_train_step%i.txt	--> y-disp at the center of each `pixel' at step i
#									60K x 784 (use reshape((60000,28,28))) to match images
# summary_dispy_test_step%i.txt		--> y-disp at the center of each `pixel' at step i
#									10K x 784 (use reshape((60000,28,28))) to match images

######### --> DOWNLOAD at:
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step1.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step2.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step3.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step4.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step5.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step6.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step7.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step8.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step9.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step10.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step11.zip
# https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step12.zip

# --> the initial position at the center of each ``pixel'' can be computed by 
# init_x = np.zeros((28,28)); init_y = np.zeros((28,28))
# for kk in range(0,28):
# 	for jj in range(0,28):
# 		init_x[kk,jj] = jj + 0.5 # x is columns, 0 is lower corner 
# 		init_y[kk,jj] = kk + 0.5 # y is rows 
# --> and the initial positions can be added to displacement to get final position
##########################################################################################

##########################################################################################
# Here we assume that the data is downloaded and in the following directories:
#		data/MNIST_input_files
#		data/FEA_psi_results
#		data/FEA_displacement_results_step1
#		data/FEA_displacement_results_step12

# ~ - ~ - ~ - ~ - UPDATE AS NEEDED / COMMENT OUT UN-USED DATA
##########################################################################################

# --> initial MNIST bitmaps 
data_dir = '../data/MNIST_input_files'

MNIST_bitmap_train = np.loadtxt(data_dir + '/mnist_img_train.txt').astype(np.uint8)
MNIST_bitmap_test  = np.loadtxt(data_dir + '/mnist_img_test.txt').astype(np.uint8)

# --> free energy at the end of the simulation (final step)
data_dir = '../data/FEA_psi_results'

final_psi_train = np.loadtxt(data_dir + '/summary_psi_train_all.txt')[:,12]
final_psi_test  = np.loadtxt(data_dir + '/summary_psi_test_all.txt')[:,12]

# --> displacement at the first step of the simulation (d = 0.001)
data_dir = '../data/FEA_displacement_results_step1'

first_dispx_train = np.loadtxt(data_dir + '/summary_dispx_train_step1.txt')
first_dispx_test  = np.loadtxt(data_dir + '/summary_dispx_test_step1.txt')
first_dispy_train = np.loadtxt(data_dir + '/summary_dispy_train_step1.txt')
first_dispy_test  = np.loadtxt(data_dir + '/summary_dispy_test_step1.txt')

# --> displacement at the final step of the simulation (d = 14.0)
data_dir = '../data/FEA_displacement_results_step12'

final_dispx_train = np.loadtxt(data_dir + '/summary_dispx_train_step12.txt')
final_dispx_test  = np.loadtxt(data_dir + '/summary_dispx_test_step12.txt')
final_dispy_train = np.loadtxt(data_dir + '/summary_dispy_train_step12.txt')
final_dispy_test  = np.loadtxt(data_dir + '/summary_dispy_test_step12.txt')

##########################################################################################
#	SAVE data in a way that's appropriate for the NN that is being trained 
##########################################################################################
# --> create a folder
path = 'NPY_FILES'
if not os.path.exists(path):
	os.mkdir(path)

# --> save everything as .npy files for faster read/write in later steps 
# --> initial MNIST bitmaps
np.save(path + '/MNIST_bitmap_train.npy',MNIST_bitmap_train)
np.save(path + '/MNIST_bitmap_test.npy',MNIST_bitmap_test)

# --> final step free energy 
np.save(path + '/final_psi_train.npy',final_psi_train)
np.save(path + '/final_psi_test.npy',final_psi_test)

# --> group displacements together 
first_disp_train = np.zeros((60000,784*2))
first_disp_train[:,0:784] = first_dispx_train
first_disp_train[:,784:]  = first_dispy_train

first_disp_test  = np.zeros((10000,784*2))
first_disp_test[:,0:784] = first_dispx_test
first_disp_test[:,784:]  = first_dispy_test

final_disp_train = np.zeros((60000,784*2))
final_disp_train[:,0:784] = final_dispx_train
final_disp_train[:,784:]  = final_dispy_train

final_disp_test  = np.zeros((10000,784*2))
final_disp_test[:,0:784] = final_dispx_test
final_disp_test[:,784:]  = final_dispy_test

np.save(path + '/first_disp_train.npy',first_disp_train)
np.save(path + '/first_disp_test.npy',first_disp_test)
np.save(path + '/final_disp_train.npy',final_disp_train)
np.save(path + '/final_disp_test.npy',final_disp_test)
