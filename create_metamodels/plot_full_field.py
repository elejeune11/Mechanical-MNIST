import numpy as np
import matplotlib.pyplot as plt
import os 
##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       DATA IMPORT            ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
# training results are found at:
# 'MLPR_bitmap_from_final_disp_train_predict'
# 'MLPR_disp_first_to_final_train_predict'
##########################################################################################
bitmap_prediction = np.load('MLPR_bitmap_from_final_disp_test_predict.npy')
disp_prediction = np.load('MLPR_disp_first_to_final_test_predict.npy')

bitmap_actual = np.load('NPY_FILES/MNIST_bitmap_test.npy')
disp_actual = np.load('NPY_FILES/final_disp_test.npy')
mini_disp_actual = np.load('NPY_FILES/first_disp_test.npy')

# initial positions: 
init_x = np.zeros((28,28))
init_y = np.zeros((28,28))

for kk in range(0,28):
	for jj in range(0,28):
		init_x[kk,jj] = jj + 0.5 # x is columns, 0 is lower corner 
		init_y[kk,jj] = kk + 0.5 # y is rows 

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~   COMPUTE ERROR -- TOTAL     ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################
# --> compute the mean absolute error, standard deviation of values as well

###################################
# --> displacement prediction 
###################################
disp_error = np.abs(disp_actual - disp_prediction)
mean_disp_error = np.mean(disp_error) #0.44277285740119215
disp_std = np.std(disp_actual) #4.255771193118254

###################################
# --> bitmap prediction 
###################################
bitmap_error = np.abs(bitmap_actual - bitmap_prediction)
mean_bitmap_error = np.mean(bitmap_error) #13.2860128529493
bitmap_std = np.std(bitmap_actual) #79.17246322228644

##########################################################################################

def flip_data(input): 
	data_to_plot_flipped = input
	data_to_plot = np.zeros(data_to_plot_flipped.shape)
	for jj in range(0,data_to_plot.shape[0]):
		for kk in range(0,data_to_plot.shape[1]):
			data_to_plot[kk,jj] = data_to_plot_flipped[int(27.0-kk),jj]
	return data_to_plot

def define_colorfield(data):
	max = np.max(data)
	min = np.min(data)
	color_data = (data - min)/(max-min)
	return color_data


##########################################################################################

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       DATA ARRANGE           ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################
path = 'full_field_plots'
if not os.path.exists(path):
	os.mkdir(path)

for idx in range(0,10):

	idx_bitmap_prediction = bitmap_prediction[idx,:].reshape(28,28)
	idx_bitmap_actual = bitmap_actual[idx,:].reshape(28,28)

	idx_bitmap_error = bitmap_error[idx,:].reshape(28,28)

	idx_disp_x_prediction = disp_prediction[idx,0:784].reshape(28,28)
	idx_disp_x_actual = disp_actual[idx,0:784].reshape(28,28)
	idx_disp_y_prediction = disp_prediction[idx,784:].reshape(28,28)
	idx_disp_y_actual = disp_actual[idx,784:].reshape(28,28)

	idx_disp_x_error = disp_error[idx,0:784].reshape(28,28)
	idx_disp_y_error  = disp_error[idx,784:].reshape(28,28)
	idx_disp_error = ( idx_disp_x_error**2.0 + idx_disp_y_error**2.0 )**(1.0/2.0)

	idx_mini_disp_x_actual = mini_disp_actual[idx,0:784].reshape(28,28)
	idx_mini_disp_y_actual = mini_disp_actual[idx,784:].reshape(28,28)

	##########################################################################################
	# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       PLOT                   ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
	##########################################################################################
	##########################################################################################

	######################################
	fig = plt.figure(figsize=(18,4))
	plt.style.use('el_papers.mplstyle')
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	######################################
	plt.subplot(1,4,1)

	x_positions = init_x + idx_mini_disp_x_actual
	y_positions = init_y + idx_mini_disp_y_actual
	idx_bitmap_actual_flip = flip_data(idx_bitmap_actual)
	color_field = define_colorfield(idx_bitmap_actual_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (color_field[kk,jj],0,0))

	plt.title('input: initial position', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.subplot(1,4,2)

	x_positions = init_x + idx_disp_x_actual
	y_positions = init_y + idx_disp_y_actual
	idx_bitmap_actual_flip = flip_data(idx_bitmap_actual)
	color_field = define_colorfield(idx_bitmap_actual_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (color_field[kk,jj],0,0))

	plt.title('output: final position', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.subplot(1,4,3)

	x_positions = init_x + idx_disp_x_prediction
	y_positions = init_y + idx_disp_y_prediction
	idx_bitmap_actual_flip = flip_data(idx_bitmap_actual)
	color_field = define_colorfield(idx_bitmap_actual_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , '+',markersize=5,color = (0,0,color_field[kk,jj]))

	plt.title('nn prediction', y=-0.01)
	plt.axis('equal')
	plt.axis('off')


	######################################
	plt.subplot(1,4,4)
	plt.plot([-5,33],[-5,45],'w.')

	x_positions = init_x + idx_disp_x_actual
	y_positions = init_y + idx_disp_y_actual
	idx_bitmap_actual_flip = flip_data(idx_bitmap_actual)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (0,0,0))


	x_positions = init_x + idx_disp_x_prediction
	y_positions = init_y + idx_disp_y_prediction
	color_field = define_colorfield(idx_disp_error)

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , '+',markersize=2.5,color = (color_field[kk,jj],0,color_field[kk,jj]))

	plt.title('error', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.savefig(path + '/displacement_prediction_%i.png'%(idx))
	plt.savefig(path + '/displacement_prediction_%i.eps'%(idx))

	##########################################################################################
	##########################################################################################

	######################################
	fig = plt.figure(figsize=(18,4))
	plt.style.use('el_papers.mplstyle')
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	######################################
	plt.subplot(1,4,1)

	x_positions = init_x + idx_disp_x_actual
	y_positions = init_y + idx_disp_y_actual

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (0,0,0))

	plt.title('input: final position', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.subplot(1,4,2)

	x_positions = init_x 
	y_positions = init_y 
	idx_bitmap_actual_flip = flip_data(idx_bitmap_actual)
	color_field = define_colorfield(idx_bitmap_actual_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (color_field[kk,jj],0,0))

	plt.title('output: bitmap', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.subplot(1,4,3)

	x_positions = init_x 
	y_positions = init_y 
	idx_bitmap_prediction_flip = flip_data(idx_bitmap_prediction)
	color_field = define_colorfield(idx_bitmap_prediction_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (0,0,color_field[kk,jj]))

	plt.title('nn prediction', y=-0.01)
	plt.axis('equal')
	plt.axis('off')


	######################################
	plt.subplot(1,4,4)
	plt.plot([-5,33],[-5,45],'w.')

	x_positions = init_x 
	y_positions = init_y
	idx_bitmap_error_flip = flip_data(idx_bitmap_error)
	color_field = define_colorfield(idx_bitmap_error_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (color_field[kk,jj],0,color_field[kk,jj]))


	plt.title('error', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	######################################
	plt.savefig(path + '/bitmap_prediction_%i.png'%(idx))
	plt.savefig(path + '/bitmap_prediction_%i.eps'%(idx))



