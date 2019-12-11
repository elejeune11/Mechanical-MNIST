import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       DATA IMPORT            ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################

x_train = np.load('NPY_FILES/first_disp_train.npy')
y_train = np.load('NPY_FILES/final_disp_train.npy')

x_test = np.load('NPY_FILES/first_disp_test.npy')
y_test = np.load('NPY_FILES/final_disp_test.npy')

########################################
# scale data
########################################
scaler = StandardScaler()  
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)  

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       RUN MODEL              ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################
model = MLPRegressor(hidden_layer_sizes=(500,500,500), activation='relu', solver='adam',\
	learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
model.fit(x_train,y_train)

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~     MAKE PRED, CALC ERR      ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################

########################################
# prediction
########################################
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

########################################
# error
########################################
y_train_err = [] 
y_test_err = [] 

y_train_perc_err = [] 
y_test_perc_err = []

for kk in range(0,y_train_predict.shape[0]):
	all = np.abs(y_train_predict[kk,:] - y_train[kk,:])
	y_train_err.append(np.median(all))
	all_perc = np.abs( all / y_train[kk,:]) * 100 
	y_train_perc_err.append( np.median(all_perc) )

for kk in range(0,y_test_predict.shape[0]):
	all = np.abs(y_test_predict[kk,:] - y_test[kk,:])
	y_test_err.append(np.median(all))
	all_perc = np.abs( all / y_test[kk,:]) * 100 
	y_test_perc_err.append( np.median(all_perc) )

median_error_train = np.median(y_train_err)
median_error_test = np.median(y_test_err)

median_percent_error_train = np.median(y_train_perc_err)
median_percent_error_test  = np.median(y_test_perc_err)

print('MAE train:',median_error_train)
print('MAE test:',median_error_test)

print('MPE train:', median_percent_error_train )
print('MPE train:', median_percent_error_test )

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       SAVE RESULTS           ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################
np.save('MLPR_disp_first_to_final_train_predict',y_train_predict)
np.save('MLPR_disp_first_to_final_test_predict',y_test_predict)

# MAE train: 0.2909284932445627
# MAE test: 0.28764190090640535
# MPE train: 14.020119652006859
# MPE train: 13.913811209865486


