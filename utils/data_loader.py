import pickle as pkl
import numpy as np

def load_experimental_data(path,animal,day_name,n1,n2,t1=0,t2=1):
	'''
		Loads experimental data
	'''
	def data_from_n(n):
		if n>=0:
			data = signals[n]
		elif n==-1:
			data = behaviour_pkl['transformed_velocity']
		elif n==-2:
			data = behaviour_pkl['transformed_licks']
		elif n==-3:
			data = (behaviour_pkl['transformed_early_reward'] + behaviour_pkl['transformed_late_reward'])/2
		elif n==-4:
			data = behaviour_pkl['transformed_early_reward']
		elif n==-5:
			data = behaviour_pkl['transformed_late_reward']
		else:
			raise ValueError('n is out of range')
		return data

	with open(f"{path}/{animal}_{day_name}_signals.pkl",'rb') as f:
	    signal_pkl = pkl.load(f)
	with open(f"{path}/{animal}_{day_name}_behaviour.pkl",'rb') as f:
	    behaviour_pkl = pkl.load(f)
	for s in ['ROIsN','trialStart','maxTrialNum','trials']:
	    assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))

	if (t1!=0) or (t2!=1):
		assert t1>=0
		assert t2<=1
		t_min = signal_pkl['trialStart']
		t_max = signal_pkl['trials'][-1]
		T1 = int(t_min + t1 * (t_max-t_min))
		T2 = int(t_min + t2 * (t_max-t_min))
		mask = (signal_pkl['trials']>=T1) & (signal_pkl['trials']<=T2)
		print(f'Loading trials {T1} to {T2} out of {t_max}')
	else:
		mask = signal_pkl['trials']>=0

	signals = signal_pkl['signals_transformed']

	data1 = data_from_n(n1)[mask]
	data2 = data_from_n(n2)[mask]

	Y_all = np.array([data1,data2]).T
	X_all = np.array(behaviour_pkl['position'][mask])#local_time

	rule = (Y_all[:,0]>0) & (Y_all[:,1]>0)  \
	        & (Y_all[:,0]<1) & (Y_all[:,1]<1)
	 
	X = np.reshape(X_all[rule],(-1,1))
	X[X<0] = 160.+X[X<0]
	X[X>160] = X[X>160]-160.
	X = X/160.
	Y = Y_all[rule]
	
	return X, Y

def get_likelihoods(summary_path,n1,n2):
	'''
	Looks up the likelihoods of the best selected model in summary.pkl
	Parameters
	----------
	summary_path: str
		A path to a summary of model selection
	n1: int
		The number of the first variable
	n2: int
		The number of the second variable
	Returns
	-------
	likelihoods: list
		A list of copula likelihood objects, corresponding to the
		best selected model for a given pair of variables.
	'''
	with open(summary_path,'rb') as f:
		data = pkl.load(f)	
	return data[n1+5,n2+5][0]

def get_model(weights_file,likelihoods,device):
	'''
	Loads the weights of the best selected model and returns
	the bvcopula.Mixed_GPInferenceModel object
	Parameters
	----------
	weights_file: str
		A path to the folder, containing the results of the model selection

	'''
	import glob
	from bvcopula import load_model
	try:
		model = load_model(weights_file, likelihoods, device)
		return model
	except FileNotFoundError:
		print('Weights file {} not found.'.format(weights_file))
		return 0
