import numpy as np
import matplotlib.pyplot as plt
import os 
##########################################################################################
# --> compute training error 
##########################################################################################
all_test_cnn = np.loadtxt('cnn_psitrain_predict.txt')
all_target_cnn = np.loadtxt('cnn_psitrain_correct.txt')

all_test_lnn = np.loadtxt('fnn_psitrain_predict.txt')
all_target_lnn = np.loadtxt('fnn_psitrain_correct.txt')

# compute percent error 
perc_err_CNN_train = np.mean(np.abs((all_test_cnn - all_target_cnn) / all_target_cnn)) * 100  #1.8968585466864325
perc_err_LNN_train = np.mean(np.abs((all_test_lnn - all_target_lnn) / all_target_lnn)) * 100  #2.3877460329830593
##########################################################################################

##########################################################################################
# --> compute and plot test error 
##########################################################################################
path = 'delta_psi_plots/'
if not os.path.exists(path):
	os.mkdir(path)


all_test_cnn = np.loadtxt('cnn_psitest_predict.txt')
all_target_cnn = np.loadtxt('cnn_psitest_correct.txt')

all_test_lnn = np.loadtxt('fnn_psitest_predict.txt')
all_target_lnn = np.loadtxt('fnn_psitest_correct.txt')

# compute percent error 
perc_err_CNN = np.mean(np.abs((all_test_cnn - all_target_cnn) / all_target_cnn)) * 100  
perc_err_LNN = np.mean(np.abs((all_test_lnn - all_target_lnn) / all_target_lnn)) * 100  

# sort CNN
argsort_cnn = np.argsort(all_target_cnn)

sort_cnn_target = all_target_cnn[argsort_cnn]
sort_cnn_test = all_test_cnn[argsort_cnn] 

if False:
	plt.figure()
	plt.plot(range(0,10000),sort_cnn_test,'.',markersize=1)
	plt.plot(range(0,10000),sort_cnn_target,'-')
	plt.savefig(path + '/cnn sorted')

# sort LNN
argsort_lnn = np.argsort(all_target_lnn)
sort_lnn_target = all_target_lnn[argsort_lnn]
sort_lnn_test = all_test_lnn[argsort_lnn] 


if False:
	plt.figure()
	plt.plot(range(0,10000),sort_lnn_test,'.',markersize=1)
	plt.plot(range(0,10000),sort_lnn_target,'-')
	plt.savefig(path + '/lnn sorted')

#### -- make an actual nice plot 
mi = np.min([np.min(sort_lnn_test),np.min(sort_lnn_target),np.min(sort_cnn_test),np.min(sort_cnn_target)])
ma = np.max([np.max(sort_lnn_test),np.max(sort_lnn_target),np.max(sort_cnn_test),np.max(sort_cnn_target)])

if False:
	plt.figure()
	plt.plot(all_target_lnn,all_test_lnn,'k.')
	plt.plot([mi, ma],[mi, ma],'r--')
	plt.savefig(path + '/compare_line')
	
### -- standard case
fig = plt.figure()
plt.style.use('el_papers.mplstyle')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.set_figheight(10)
fig.set_figwidth(5)

plt.subplot(2,1,1)
plt.plot(all_target_lnn,all_test_lnn,'k.')
plt.plot([mi, ma],[mi, ma],'r--')
plt.xlim((100,225))
plt.ylim((100,225))
plt.xlabel(r'target $\Delta \psi$')
plt.ylabel(r'predicted $\Delta \psi$')
str = 'FNN: %.1f'%(perc_err_LNN) + ' MPE'
plt.title(str)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(range(0,10000),sort_lnn_test,'k.',label=r'prediction')
plt.plot(range(0,10000),sort_lnn_target,'r--',label=r'target')
plt.legend()
plt.xlabel(r'sorted target index')
plt.ylabel(r'$\Delta \psi$')
str = 'FNN: %.1f'%(perc_err_LNN) + ' MPE'
plt.title(str)
plt.tight_layout()

plt.savefig(path + '/plot_lnn_error.png')
plt.savefig(path + '/plot_lnn_error.eps')

### -- CNN case


fig = plt.figure()
plt.style.use('el_papers.mplstyle')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.set_figheight(10)
fig.set_figwidth(5)

plt.subplot(2,1,1)
plt.plot(all_target_cnn,all_test_cnn,'k.')
plt.plot([mi, ma],[mi, ma],'r--')
plt.xlim((100,225))
plt.ylim((100,225))
plt.xlabel(r'target $\Delta \psi$')
plt.ylabel(r'predicted $\Delta \psi$')
str = 'CNN: %.1f'%(perc_err_CNN ) + ' MPE'
plt.title(str)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(range(0,10000),sort_cnn_test,'k.',label=r'prediction')
plt.plot(range(0,10000),sort_cnn_target,'r--',label=r'target')
plt.legend()
plt.xlabel(r'sorted target index')
plt.ylabel(r'$\Delta \psi$')
str = 'CNN: %.1f'%(perc_err_CNN) + ' MPE'
plt.title(str)
plt.tight_layout()

plt.savefig(path + '/plot_cnn_error.png')
plt.savefig(path + '/plot_cnn_error.eps')

