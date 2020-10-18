import time
import pickle as pkl
import numpy as np
import sys
import multiprocessing
import os
from torch import device, tensor
from . import conf
sys.path.insert(0, conf.path2code)

import utils
import select_copula
import train
from bvcopula import MixtureCopula_Likelihood

def worker(X, Y0, Y1, idxs, layer):
	# get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = (int(cpu_name[cpu_name.find('-') + 1:]) - 1)%len(gpu_id_list) # ids will be 8 consequent numbers
	gpu_id = gpu_id_list[cpu_id]

	device_str = f'cuda:{gpu_id}'

	unique_id = unique_id_list[cpu_id]

	Y = np.stack([Y1,Y0]).T # order!
	n0, n1, n_out = idxs[0] + layer, idxs[1]+layer, idxs[1]-1 # substitute this to get other (not C) vines

	train_x = tensor(X).float().to(device=device(device_str))
	train_y = tensor(Y).float().to(device=device(device_str))

	print(f'Selecting {n0}-{n1} on {device_str}')
	try:
		t_start = time.time()
		# (likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),exp_pref,out_dir,layer,n+layer)
		(likelihoods, waic) = select_copula.select_light(X,Y,device(device_str),
							exp_pref,out_dir,n0,n1,train_x=train_x,train_y=train_y)
		t_end = time.time()
		print(f'Selection took {int((t_end-t_start)/60)} min')
	except RuntimeError as error:
		print(error)
		# logging.error(error, exc_info=True)
		return -1
	finally:
		print(f"{n0}-{n1}",utils.get_copula_name_string(likelihoods),waic)
		# save textual info into model list
		with open(out_dir+'_model_list.txt','a') as f:
			f.write(f"{n0}-{n1} {utils.get_copula_name_string(likelihoods)}\t{waic:.4f}\t{int(t_end-t_start)} sec\n")

		mix_lik = MixtureCopula_Likelihood(likelihoods)
		dump = mix_lik.serialize()	

		if utils.get_copula_name_string(likelihoods)!='Independence':
			weights_file = f"{out_dir}/model_{exp_pref}_{n0}-{n1}.pth"
			model = utils.get_model(weights_file, likelihoods, device(device_str))
			copula = model.marginalize(train_x)
			y = copula.ccdf(train_y).cpu().numpy()
		else:
			y = Y1

		return (dump, y)

# def setup(x, y, z):
#     """
#     	Sets up the worker processes of the pool. 
#     """
#     global exp_pref
#     exp_pref = Adder()

def train_next_layer(X, Y, exp, layer, gpus):

	global exp_pref, out_dir
	exp_pref = exp
	out_dir = f'{conf.path2outputs}/{exp_pref}/layer{layer}'

	global gpu_id_list, unique_id_list
	gpu_id_list = gpus
	unique_id_list = np.random.randint(0,10000,len(gpu_id_list)) #TODO: make truely unique

	if layer==0:
		try:
			os.mkdir(f'{conf.path2outputs}/{exp_pref}')
		except FileExistsError as error:
			print(f"Error:{error}")

	try:
		os.mkdir(out_dir)
	except FileExistsError as error:
		print(f"Error:{error}")

	NN = Y.shape[-1]-1

	results = np.empty(NN,dtype=object)
	pool = multiprocessing.Pool(len(gpu_id_list))
		# initializer=setup, initargs=["some arg", "another", 2])

	for i in np.arange(1,NN+1): 
		results[i-1] = pool.apply_async(worker, (X, Y[:,0], Y[:,i], [0,i],  layer, ))

	pool.close()
	pool.join()  # block at this line until all processes are done
	print(f"Layer {layer} completed")

	models, Y_next = [], []
	for result in results:
		m, y = result.get()
		models.append(m)
		Y_next.append(y)

	Y_next = np.array(Y_next).T

	return models, Y_next